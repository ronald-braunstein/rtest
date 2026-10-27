//! Resolves compile-time constants from Python AST.
//!
//! This module builds a scope tree from a Python module's AST to resolve
//! constant expressions like `X`, `Color.RED`, or `Outer.Inner.VALUE`.
//! When a [`ModuleResolver`] is provided, imported names from other project
//! modules are resolved as well.

use ruff_python_ast::name::UnqualifiedName;
use ruff_python_ast::{helpers::map_subscript, Expr, ModModule, Stmt};
use ruff_python_stdlib::sys::is_known_standard_library;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use super::cases::LiteralValue;
use super::module_resolver::{resolve_relative_import, ModuleResolver};

/// Maximum import-chain depth when loading modules on demand.
const MAX_IMPORT_DEPTH: usize = 8;

/// A scope containing constants (module-level or class body).
///
/// Supports nested classes via recursive structure.
#[derive(Debug, Default)]
struct ConstantScope {
    /// Each assignment has exactly one resolution state.
    bindings: HashMap<String, ConstantBinding>,
    /// Nested class scopes.
    children: HashMap<String, ConstantScope>,
    /// Base class paths (`class Foo(Bar)` → `"Bar"`).
    bases: Vec<String>,
    /// Undecorated functions, whose `__name__` is their definition name.
    functions: std::collections::HashSet<String>,
    /// Return annotations of functions whose decorators keep the annotated return type.
    returns: HashMap<String, Expr>,
    /// Runtime fixture parametrization cannot be represented by decorator-only IDs.
    runtime_parameters: bool,
}

#[derive(Debug)]
struct ConstantBinding {
    /// Repeated class assignments need decorator evaluation order and use fallback.
    reassigned: bool,
    value: RefCell<BindingValue>,
}

#[derive(Debug)]
enum BindingValue {
    Known(LiteralValue),
    Pending(Rc<PendingValue>),
    Unknown,
}

impl ConstantBinding {
    fn known_value(&self) -> Option<LiteralValue> {
        match &*self.value.borrow() {
            BindingValue::Known(value) => Some(value.clone()),
            BindingValue::Pending(_) | BindingValue::Unknown => None,
        }
    }
}

/// A value whose names are resolved after every module scope is built.
#[derive(Debug, Clone)]
enum PendingValue {
    Literal(LiteralValue),
    Path(Vec<String>),
    Sequence(Vec<PendingValue>),
    Mapping(Vec<(PendingValue, Option<PendingValue>)>),
    /// Iterating a sequence or mapping: `list(x)`, `tuple(x)`, `sorted(x)`, `x.keys()`, `x.values()`.
    Elements {
        source: Box<PendingValue>,
        view: ElementView,
        sorted: bool,
    },
    BinOp(
        Box<PendingValue>,
        ruff_python_ast::Operator,
        Box<PendingValue>,
    ),
    Param {
        values: Vec<PendingValue>,
        id: Option<String>,
    },
    Slice {
        source: Box<PendingValue>,
        start: Option<i64>,
        stop: Option<i64>,
    },
    /// A call whose result is IDed positionally when its return annotation is an object type.
    CallResult(Vec<String>),
}

/// Return annotations that may produce values pytest IDs by value or `__name__`.
const VALUE_TYPE_NAMES: &[&str] = &[
    "Any",
    "Callable",
    "Enum",
    "Final",
    "IntEnum",
    "Literal",
    "LiteralString",
    "None",
    "Optional",
    "Pattern",
    "Self",
    "StrEnum",
    "Type",
    "TypeVar",
    "Union",
    "bool",
    "bytes",
    "complex",
    "float",
    "int",
    "object",
    "str",
    "type",
];

/// Container annotations; containers have no `__name__`, so pytest IDs them positionally.
const CONTAINER_TYPE_NAMES: &[&str] = &[
    "DefaultDict",
    "Dict",
    "FrozenSet",
    "List",
    "OrderedDict",
    "Set",
    "Tuple",
    "defaultdict",
    "dict",
    "frozenset",
    "list",
    "set",
    "tuple",
];

/// Stdlib classes whose instances pytest IDs positionally.
const STDLIB_OBJECT_TYPES: &[&str] = &[
    "Counter",
    "Decimal",
    "Fraction",
    "OrderedDict",
    "Path",
    "PosixPath",
    "PurePath",
    "UUID",
    "date",
    "datetime",
    "defaultdict",
    "deque",
    "time",
    "timedelta",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ElementView {
    Keys,
    Values,
}

/// Builtins that produce containers; pytest IDs container values positionally.
const CONTAINER_BUILTINS: &[&str] = &["dict", "frozenset", "list", "set", "tuple"];

/// Builtin classes and functions whose `__name__` matches the builtin name.
const NAMED_BUILTINS: &[&str] = &[
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "BaseException",
    "BufferError",
    "ConnectionError",
    "DeprecationWarning",
    "EOFError",
    "Exception",
    "FileExistsError",
    "FileNotFoundError",
    "FloatingPointError",
    "ImportError",
    "IndexError",
    "IsADirectoryError",
    "KeyError",
    "KeyboardInterrupt",
    "LookupError",
    "MemoryError",
    "ModuleNotFoundError",
    "NameError",
    "NotADirectoryError",
    "NotImplementedError",
    "OSError",
    "OverflowError",
    "PermissionError",
    "RecursionError",
    "ReferenceError",
    "RuntimeError",
    "RuntimeWarning",
    "StopIteration",
    "SyntaxError",
    "SystemError",
    "TimeoutError",
    "TypeError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "UnicodeError",
    "UserWarning",
    "ValueError",
    "Warning",
    "ZeroDivisionError",
    "abs",
    "all",
    "any",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "complex",
    "dict",
    "float",
    "frozenset",
    "int",
    "len",
    "list",
    "max",
    "min",
    "object",
    "print",
    "repr",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
];

/// Names a module binds at top level; builtins are only trusted when not shadowed.
#[derive(Debug, Default)]
struct ModuleNames {
    bound: std::collections::HashSet<String>,
    /// Spellings of `pytest.param` in this module (`pytest.param`, `param`, ...).
    pytest_param: std::collections::HashSet<String>,
}

impl ModuleNames {
    fn from_module(module: &ModModule, imports: &HashMap<String, ImportBinding>) -> Self {
        let pytest_param = imports
            .iter()
            .filter(|(_, import)| import.module_path == ["pytest"])
            .filter_map(|(local, import)| match import.imported_name.as_deref() {
                None => Some(format!("{local}.param")),
                Some("param") => Some(local.clone()),
                Some(_) => None,
            })
            .collect();
        let mut names: std::collections::HashSet<String> = imports.keys().cloned().collect();
        for stmt in &module.body {
            match stmt {
                Stmt::Assign(assign) => names.extend(
                    assign
                        .targets
                        .iter()
                        .filter_map(|t| t.as_name_expr().map(|n| n.id.to_string())),
                ),
                Stmt::AnnAssign(ann) => {
                    names.extend(ann.target.as_name_expr().map(|n| n.id.to_string()))
                }
                Stmt::ClassDef(class_def) => {
                    names.insert(class_def.name.to_string());
                }
                Stmt::FunctionDef(func) => {
                    names.insert(func.name.to_string());
                }
                _ => {}
            }
        }
        Self {
            bound: names,
            pytest_param,
        }
    }

    fn is_pytest_param(&self, func: &Expr) -> bool {
        UnqualifiedName::from_expr(func)
            .is_some_and(|name| self.pytest_param.contains(&name.to_string()))
    }

    fn builtin(&self, expr: &Expr) -> Option<&'static str> {
        let name = expr.as_name_expr()?.id.as_str();
        if self.bound.contains(name) {
            return None;
        }
        NAMED_BUILTINS
            .iter()
            .chain(CONTAINER_BUILTINS)
            .chain(&["range"])
            .find(|builtin| **builtin == name)
            .copied()
    }
}

/// Values bound by an enclosing comprehension, which shadow module names.
pub(crate) type LocalValues = HashMap<String, LiteralValue>;

/// Build a pending value from an expression without resolving names.
fn pending_value(expr: &Expr, names: &ModuleNames, locals: &LocalValues) -> Option<PendingValue> {
    if let Some(literal) = try_extract_literal(expr) {
        return Some(PendingValue::Literal(literal));
    }
    match expr {
        Expr::Name(name) => {
            if let Some(value) = locals.get(name.id.as_str()) {
                return Some(PendingValue::Literal(value.clone()));
            }
            if let Some(builtin) = names.builtin(expr).filter(|b| NAMED_BUILTINS.contains(b)) {
                return Some(PendingValue::Literal(LiteralValue::Named(builtin.into())));
            }
            Some(PendingValue::Path(vec![name.id.to_string()]))
        }
        Expr::Attribute(_) => {
            let name = UnqualifiedName::from_expr(expr)?;
            let segments = name.segments();
            if locals.contains_key(segments[0]) {
                return None;
            }
            Some(PendingValue::Path(
                segments.iter().map(|s| (*s).to_string()).collect(),
            ))
        }
        Expr::List(ruff_python_ast::ExprList { elts, .. })
        | Expr::Tuple(ruff_python_ast::ExprTuple { elts, .. }) => Some(PendingValue::Sequence(
            elts.iter()
                .map(|elt| pending_value(elt, names, locals))
                .collect::<Option<_>>()?,
        )),
        Expr::Dict(dict) => Some(PendingValue::Mapping(
            dict.items
                .iter()
                .map(|item| {
                    let key = pending_value(item.key.as_ref()?, names, locals)?;
                    Some((key, pending_value(&item.value, names, locals)))
                })
                .collect::<Option<_>>()?,
        )),
        // Set iteration order depends on hashing, so sets are only countable as single values.
        Expr::Set(_) => Some(PendingValue::Literal(LiteralValue::Opaque)),
        Expr::BinOp(binop) => Some(PendingValue::BinOp(
            Box::new(pending_value(&binop.left, names, locals)?),
            binop.op,
            Box::new(pending_value(&binop.right, names, locals)?),
        )),
        Expr::Subscript(subscript) => {
            let Expr::Slice(slice) = subscript.slice.as_ref() else {
                return None;
            };
            if slice.step.is_some() {
                return None;
            }
            let bound = |bound: &Option<Box<Expr>>| match bound.as_deref() {
                None => Some(None),
                Some(expr) => match try_extract_literal(expr)? {
                    LiteralValue::Int(i) => Some(Some(i)),
                    _ => None,
                },
            };
            Some(PendingValue::Slice {
                source: Box::new(pending_value(&subscript.value, names, locals)?),
                start: bound(&slice.lower)?,
                stop: bound(&slice.upper)?,
            })
        }
        Expr::Call(call) => pending_call(call, names, locals),
        _ => None,
    }
}

fn pending_call(
    call: &ruff_python_ast::ExprCall,
    names: &ModuleNames,
    locals: &LocalValues,
) -> Option<PendingValue> {
    let args = &call.arguments;
    if names.is_pytest_param(&call.func) {
        let mut id = None;
        for keyword in &args.keywords {
            match (keyword.arg.as_deref(), try_extract_literal(&keyword.value)) {
                (Some("id"), Some(LiteralValue::String(value))) => id = Some(value),
                (Some("id"), Some(LiteralValue::None)) | (Some("marks"), _) => {}
                _ => return None,
            }
        }
        return Some(PendingValue::Param {
            values: args
                .args
                .iter()
                .map(|arg| pending_value(arg, names, locals))
                .collect::<Option<_>>()?,
            id,
        });
    }
    if let Expr::Attribute(attr) = call.func.as_ref() {
        let view = match attr.attr.as_str() {
            "keys" => ElementView::Keys,
            "values" => ElementView::Values,
            _ => return constructor_value(call, locals),
        };
        if !args.is_empty() {
            return None;
        }
        return Some(PendingValue::Elements {
            source: Box::new(pending_value(&attr.value, names, locals)?),
            view,
            sorted: false,
        });
    }
    let Some(builtin) = names.builtin(&call.func) else {
        return constructor_value(call, locals);
    };
    match (builtin, args.args.as_ref(), args.keywords.is_empty()) {
        ("range", bounds, true) if (1..=3).contains(&bounds.len()) => {
            let bounds: Vec<i64> = bounds
                .iter()
                .map(|b| match try_extract_literal(b)? {
                    LiteralValue::Int(i) => Some(i),
                    _ => None,
                })
                .collect::<Option<_>>()?;
            let (start, stop, step) = match bounds.as_slice() {
                [stop] => (0, *stop, 1),
                [start, stop] => (*start, *stop, 1),
                [start, stop, step] if *step != 0 => (*start, *stop, *step),
                _ => return None,
            };
            let values: Vec<_> = std::iter::successors(Some(start), |i| i.checked_add(step))
                .take_while(|i| if step > 0 { *i < stop } else { *i > stop })
                .take(MAX_RANGE_LEN + 1)
                .map(|i| PendingValue::Literal(LiteralValue::Int(i)))
                .collect();
            (values.len() <= MAX_RANGE_LEN).then_some(PendingValue::Sequence(values))
        }
        ("list" | "tuple" | "sorted", [source], true) => Some(PendingValue::Elements {
            source: Box::new(pending_value(source, names, locals)?),
            view: ElementView::Keys,
            sorted: builtin == "sorted",
        }),
        (container, _, _) if CONTAINER_BUILTINS.contains(&container) => {
            Some(PendingValue::Literal(LiteralValue::Opaque))
        }
        _ => None,
    }
}

/// Objects built by constructors are IDed positionally, so their arguments are irrelevant.
/// Other calls depend on the callee's return annotation.
fn constructor_value(
    call: &ruff_python_ast::ExprCall,
    locals: &LocalValues,
) -> Option<PendingValue> {
    if super::cases::is_constructor_call(&call.func) {
        return Some(PendingValue::Literal(LiteralValue::Opaque));
    }
    let callee = UnqualifiedName::from_expr(&call.func)?;
    if locals.contains_key(*callee.segments().first()?) {
        return None;
    }
    Some(PendingValue::CallResult(
        callee.segments().iter().map(|s| (*s).to_string()).collect(),
    ))
}

const MAX_RANGE_LEN: usize = 10_000;

/// `sorted()` for homogeneous strings or ints; other mixes need Python's comparison rules.
fn sort_literals(items: &mut [LiteralValue]) -> Option<()> {
    if items
        .iter()
        .all(|item| matches!(item, LiteralValue::String(_)))
    {
        items.sort_by(|a, b| match (a, b) {
            (LiteralValue::String(a), LiteralValue::String(b)) => a.cmp(b),
            _ => std::cmp::Ordering::Equal,
        });
    } else if items
        .iter()
        .all(|item| matches!(item, LiteralValue::Int(_)))
    {
        items.sort_by_key(|item| match item {
            LiteralValue::Int(i) => *i,
            _ => 0,
        });
    } else {
        return None;
    }
    Some(())
}

/// A class name, as opposed to a CONSTANT or function name.
fn is_class_like_name(name: &str) -> bool {
    name.starts_with(|c: char| c.is_ascii_uppercase())
        && name.chars().any(|c| c.is_ascii_lowercase())
}

/// Binding created by `import` / `from ... import`.
#[derive(Debug)]
struct ImportBinding {
    module_path: Vec<String>,
    /// None for a module import; Some(name) for an imported member.
    imported_name: Option<String>,
}

#[derive(Debug)]
struct LoadedModule {
    scope: ConstantScope,
    imports: HashMap<String, ImportBinding>,
    names: ModuleNames,
}

impl LoadedModule {
    fn from_module(module: &ModModule, module_path: &[String], is_package: bool) -> Self {
        // Relative imports start from the file's package, including in __init__.py.
        let context: Vec<_> = module_path
            .iter()
            .cloned()
            .chain(is_package.then(|| "__init__".to_string()))
            .collect();
        let imports = collect_imports(module, &context);
        let names = ModuleNames::from_module(module, &imports);
        Self {
            scope: ConstantScope::from_statements(&module.body, &imports, &names),
            imports,
            names,
        }
    }
}

/// A constant resolved from a name or attribute expression.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedConstant {
    pub value: LiteralValue,
    /// Source path as written (`["Color", "RED"]` or `["DATA"]`).
    pub source_path: Vec<String>,
}

/// Resolver for constant expressions in a module.
pub struct ConstantResolver {
    current_path: Vec<String>,
    modules: RefCell<HashMap<Vec<String>, Rc<LoadedModule>>>,
    /// Present when imported modules may be loaded on demand.
    loader: Option<ModuleResolver>,
}

impl ConstantScope {
    /// Build a scope tree from a list of statements.
    fn from_statements(
        stmts: &[Stmt],
        imports: &HashMap<String, ImportBinding>,
        names: &ModuleNames,
    ) -> Self {
        let mut scope = Self::default();

        for stmt in stmts {
            let (target, value) = match stmt {
                Stmt::Assign(assign) if assign.targets.len() == 1 => {
                    (&assign.targets[0], Some(assign.value.as_ref()))
                }
                Stmt::AnnAssign(ann) => (ann.target.as_ref(), ann.value.as_deref()),
                Stmt::ClassDef(class_def) => {
                    let mut child = Self::from_statements(&class_def.body, imports, names);
                    scope.runtime_parameters |= child.runtime_parameters;
                    child.bases = class_def
                        .bases()
                        .iter()
                        .map(map_subscript)
                        .filter_map(|base| {
                            imported_expression_name(base, imports)
                                .filter(|name| matches!(name.as_str(), "enum.Enum" | "enum.Flag"))
                                .or_else(|| {
                                    UnqualifiedName::from_expr(base).map(|name| name.to_string())
                                })
                        })
                        .collect();
                    scope.children.insert(class_def.name.to_string(), child);
                    continue;
                }
                Stmt::FunctionDef(func) => {
                    if func.decorator_list.is_empty() {
                        scope.functions.insert(func.name.to_string());
                    }
                    let keeps_return = func.decorator_list.iter().all(|decorator| {
                        matches!(
                            UnqualifiedName::from_expr(&decorator.expression)
                                .map(|name| name.to_string())
                                .as_deref(),
                            Some("staticmethod" | "classmethod")
                        )
                    });
                    if let (Some(returns), true, false) =
                        (func.returns.as_deref(), keeps_return, func.is_async)
                    {
                        scope.returns.insert(func.name.to_string(), returns.clone());
                    }
                    scope.runtime_parameters |= func.name.as_str() == "pytest_generate_tests"
                        || func.decorator_list.iter().any(|decorator| {
                            let Expr::Call(call) = &decorator.expression else {
                                return false;
                            };
                            matches!(
                                imported_expression_name(&call.func, imports).as_deref(),
                                Some("pytest.fixture" | "pytest.yield_fixture")
                            ) && call.arguments.find_keyword("params").is_some()
                        });
                    continue;
                }
                _ => continue,
            };
            if let (Expr::Name(name), Some(value_expr)) = (target, value) {
                // Global markers may be imported or computed; let pytest apply them.
                scope.runtime_parameters |= name.id.as_str() == "pytestmark";
                record_assignment(&mut scope, name.id.as_str(), value_expr, names);
            }
        }

        scope
    }

    /// Resolve a path like `["Outer", "Inner", "X"]` to a value.
    fn resolve_path(&self, path: &[&str]) -> Option<LiteralValue> {
        let (member, parents) = path.split_last()?;
        match parents.split_last() {
            None => self.bindings.get(*member)?.known_value(),
            Some((class_name, ancestors)) => {
                let scope = ancestors
                    .iter()
                    .try_fold(self, |scope, name| scope.children.get(*name))?;
                scope.lookup_class_member(class_name, member)
            }
        }
    }

    fn lookup_class_member(&self, class_name: &str, member: &str) -> Option<LiteralValue> {
        let class = self.children.get(class_name)?;
        if let Some(binding) = class.bindings.get(member) {
            return binding.known_value();
        }
        if class.children.contains_key(member) || class.functions.contains(member) {
            return Some(LiteralValue::Named(member.into()));
        }
        // Multiple inheritance needs Python's C3 order; leave it to fallback.
        let [base] = class.bases.as_slice() else {
            return None;
        };
        self.lookup_class_member(base, member)
    }

    fn class_is_enum(&self, class_name: &str) -> bool {
        let Some(class) = self.children.get(class_name) else {
            return false;
        };
        class.bases.iter().any(|base| is_enum_class_name(base))
            || class.bases.iter().any(|base| self.class_is_enum(base))
    }

    fn collect_pending<'a>(
        &'a self,
        module_path: &'a [String],
        pending: &mut Vec<PendingAssignment<'a>>,
    ) {
        for binding in self.bindings.values() {
            if let BindingValue::Pending(pending_value) = &*binding.value.borrow() {
                pending.push(PendingAssignment {
                    module_path,
                    value: &binding.value,
                    pending: Rc::clone(pending_value),
                });
            }
        }
        for child in self.children.values() {
            child.collect_pending(module_path, pending);
        }
    }
}

fn record_assignment(scope: &mut ConstantScope, name: &str, value: &Expr, names: &ModuleNames) {
    let value = match pending_value(value, names, &LocalValues::new()) {
        Some(PendingValue::Literal(literal)) => BindingValue::Known(literal),
        Some(pending) => BindingValue::Pending(Rc::new(pending)),
        None => BindingValue::Unknown,
    };
    let reassigned = scope.bindings.contains_key(name);
    scope.bindings.insert(
        name.into(),
        ConstantBinding {
            reassigned,
            value: RefCell::new(value),
        },
    );
}

fn imported_expression_name(
    expr: &Expr,
    imports: &HashMap<String, ImportBinding>,
) -> Option<String> {
    let name = UnqualifiedName::from_expr(expr)?;
    let (first, rest) = name.segments().split_first()?;
    let import = imports.get(*first)?;
    Some(
        import
            .module_path
            .iter()
            .map(String::as_str)
            .chain(import.imported_name.iter().map(String::as_str))
            .chain(rest.iter().copied())
            .collect::<Vec<_>>()
            .join("."),
    )
}

fn is_enum_class_name(name: &str) -> bool {
    matches!(
        name,
        "Enum" | "IntEnum" | "StrEnum" | "Flag" | "IntFlag" | "enum.Enum" | "enum.Flag"
    )
}

/// Enum identity uses the access path, not the discarded underlying value.
#[derive(PartialEq)]
enum ResolvedValue {
    Literal(LiteralValue),
    EnumMember,
}

impl ResolvedValue {
    fn into_literal(self, path: &[String]) -> LiteralValue {
        match self {
            Self::Literal(value) => value,
            Self::EnumMember => {
                LiteralValue::EnumMember(path[path.len().saturating_sub(2)..].join("."))
            }
        }
    }
}

impl ConstantResolver {
    /// Build a resolver from a parsed module (same-module constants only).
    pub fn from_module(module: &ModModule) -> Self {
        Self::new(module, Vec::new(), None, false)
    }

    /// Build a resolver that follows project-local imports on demand.
    ///
    /// Imported modules are parsed only when a name lookup actually needs them,
    /// so collecting a file that imports a large graph stays cheap.
    pub fn from_module_with_imports(
        module: &ModModule,
        module_path: &[String],
        module_resolver: ModuleResolver,
        file_path: &Path,
    ) -> Self {
        Self::new(
            module,
            module_path.to_vec(),
            Some(module_resolver),
            file_path.ends_with("__init__.py"),
        )
    }

    fn new(
        module: &ModModule,
        current_path: Vec<String>,
        loader: Option<ModuleResolver>,
        is_package: bool,
    ) -> Self {
        let loaded = LoadedModule::from_module(module, &current_path, is_package);
        let resolver = Self {
            modules: RefCell::new(HashMap::from([(current_path.clone(), Rc::new(loaded))])),
            current_path,
            loader,
        };
        resolver.resolve_pending_assignments();
        resolver
    }

    pub fn imported_expression_name(&self, expr: &Expr) -> Option<String> {
        imported_expression_name(
            expr,
            &self.modules.borrow().get(&self.current_path)?.imports,
        )
    }

    pub(crate) fn has_runtime_parameters(&self) -> bool {
        self.modules
            .borrow()
            .values()
            .any(|module| module.scope.runtime_parameters)
    }

    /// Resolve an expression, optionally looking up names on an enclosing class first.
    pub fn resolve_in_class(
        &self,
        enclosing_class: Option<&str>,
        expr: &Expr,
    ) -> Option<ResolvedConstant> {
        let name = UnqualifiedName::from_expr(expr)?;
        let value = self.lookup_path_at(&self.current_path, enclosing_class, name.segments(), 0)?;
        let path: Vec<String> = name
            .segments()
            .iter()
            .map(|segment| (*segment).into())
            .collect();
        Some(ResolvedConstant {
            value: value.into_literal(&path),
            source_path: path,
        })
    }

    /// Parse `module_path` if it is not already in `self.modules`.
    fn ensure_loaded(&self, module_path: &[String], depth: usize) {
        if self.modules.borrow().contains_key(module_path) || depth > MAX_IMPORT_DEPTH {
            return;
        }
        if is_skipped_module(module_path) {
            return;
        }
        let Some(loader) = &self.loader else {
            return;
        };
        let loaded = match loader.resolve_and_load(module_path) {
            Ok(parsed) => LoadedModule::from_module(
                &parsed.module,
                module_path,
                parsed.path.ends_with("__init__.py"),
            ),
            Err(_) => return,
        };
        self.modules
            .borrow_mut()
            .insert(module_path.to_vec(), Rc::new(loaded));
    }

    fn resolve_pending_assignments(&self) {
        for _ in 0..8 {
            // Hold stable modules rather than copying every scope. Freeze the worklist
            // before resolving: lookups may cache values and load additional modules.
            let modules: Vec<_> = self
                .modules
                .borrow()
                .iter()
                .map(|(path, module)| (path.clone(), Rc::clone(module)))
                .collect();
            let mut pending = Vec::new();
            for (path, module) in &modules {
                module.scope.collect_pending(path, &mut pending);
            }
            let updates: Vec<_> = pending
                .into_iter()
                .filter_map(|assignment| {
                    self.resolve_pending(assignment.module_path, None, &assignment.pending, 0)
                        .map(|value| (assignment, value))
                })
                .collect();
            if updates.is_empty() {
                break;
            }
            for (assignment, value) in updates {
                *assignment.value.borrow_mut() = BindingValue::Known(value);
            }
        }
    }

    fn lookup_path_at(
        &self,
        module_path: &[String],
        enclosing_class: Option<&str>,
        path: &[&str],
        depth: usize,
    ) -> Option<ResolvedValue> {
        if depth > MAX_IMPORT_DEPTH {
            return None;
        }
        let (&first, rest) = path.split_first()?;
        let module = Rc::clone(self.modules.borrow().get(module_path)?);
        if enclosing_class
            .and_then(|name| module.scope.children.get(name))
            .and_then(|scope| scope.bindings.get(first))
            .is_some_and(|binding| binding.reassigned)
        {
            return None;
        }
        if rest.is_empty() {
            if let Some(found) = enclosing_class
                .and_then(|class| self.lookup_class_member(module_path, class, first, depth))
            {
                return Some(found);
            }
            if enclosing_class
                .and_then(|name| module.scope.children.get(name))
                .is_some_and(|scope| scope.bindings.contains_key(first))
            {
                return None;
            }
            if let Some(binding) = module.scope.bindings.get(first) {
                let pending = match &*binding.value.borrow() {
                    BindingValue::Known(value) => {
                        return Some(ResolvedValue::Literal(value.clone()))
                    }
                    BindingValue::Pending(pending) => Rc::clone(pending),
                    BindingValue::Unknown => return None,
                };
                let value = self.resolve_pending(module_path, None, &pending, depth + 1)?;
                *binding.value.borrow_mut() = BindingValue::Known(value.clone());
                return Some(ResolvedValue::Literal(value));
            }
            if module.scope.children.contains_key(first) || module.scope.functions.contains(first) {
                return Some(ResolvedValue::Literal(LiteralValue::Named(first.into())));
            }
        } else if module.scope.children.contains_key(first) {
            return self.lookup_class_path(module_path, first, rest, depth);
        }

        let Some(import) = module.imports.get(first) else {
            let builtin = rest.is_empty() && NAMED_BUILTINS.contains(&first);
            return builtin.then(|| ResolvedValue::Literal(LiteralValue::Named(first.into())));
        };
        if import.imported_name.is_none() && rest.is_empty() {
            return None;
        }
        let abs = &import.module_path;
        self.ensure_loaded(abs, depth + 1);
        let imported_path: Vec<&str> = import
            .imported_name
            .iter()
            .map(String::as_str)
            .chain(rest.iter().copied())
            .collect();
        let found = self.lookup_path_at(abs, None, &imported_path, depth + 1);
        // Stdlib and third-party modules are not parsed; their classes still ID by `__name__`.
        match (found, imported_path.as_slice()) {
            (None, [name])
                if is_class_like_name(name) && !self.modules.borrow().contains_key(abs) =>
            {
                Some(ResolvedValue::Literal(LiteralValue::Named((*name).into())))
            }
            (found, _) => found,
        }
    }

    /// Find a called function's return annotation and the module it is written in.
    fn function_return(
        &self,
        module_path: &[String],
        enclosing_class: Option<&str>,
        path: &[String],
        depth: usize,
    ) -> Option<(Vec<String>, Expr)> {
        if depth > MAX_IMPORT_DEPTH {
            return None;
        }
        let module = Rc::clone(self.modules.borrow().get(module_path)?);
        let follow = |import: &ImportBinding, path: Vec<String>| {
            self.ensure_loaded(&import.module_path, depth + 1);
            self.function_return(&import.module_path, None, &path, depth + 1)
        };
        match path {
            [name] => {
                if let Some(returns) = enclosing_class
                    .and_then(|class| module.scope.children.get(class))
                    .and_then(|class| class.returns.get(name))
                {
                    return Some((module_path.to_vec(), returns.clone()));
                }
                if let Some(returns) = module.scope.returns.get(name) {
                    return Some((module_path.to_vec(), returns.clone()));
                }
                if module.scope.bindings.contains_key(name)
                    || module.scope.children.contains_key(name)
                {
                    return None;
                }
                let import = module.imports.get(name)?;
                follow(import, vec![import.imported_name.clone()?])
            }
            [owner, function] => {
                if let Some(class) = module.scope.children.get(owner) {
                    let returns = class.returns.get(function)?;
                    return Some((module_path.to_vec(), returns.clone()));
                }
                let import = module.imports.get(owner)?;
                match &import.imported_name {
                    Some(class) => follow(import, vec![class.clone(), function.clone()]),
                    None => follow(import, vec![function.clone()]),
                }
            }
            _ => None,
        }
    }

    /// Whether every value of an annotated type is an object pytest IDs positionally.
    fn annotation_is_object(
        &self,
        module_path: &[String],
        annotation: &Expr,
        depth: usize,
    ) -> bool {
        let (base, subscripted) = match annotation {
            Expr::Subscript(subscript) => (subscript.value.as_ref(), true),
            other => (other, false),
        };
        let Some(name) = UnqualifiedName::from_expr(base) else {
            return false;
        };
        let segments: Vec<String> = name.segments().iter().map(|s| (*s).to_string()).collect();
        let last = segments.last().map(String::as_str).unwrap_or_default();
        if VALUE_TYPE_NAMES.contains(&last) {
            return false;
        }
        if CONTAINER_TYPE_NAMES.contains(&last) {
            return true;
        }
        !subscripted && self.is_object_class(module_path, &segments, depth)
    }

    fn is_object_class(&self, module_path: &[String], path: &[String], depth: usize) -> bool {
        if depth > MAX_IMPORT_DEPTH {
            return false;
        }
        let Some(module) = self.modules.borrow().get(module_path).map(Rc::clone) else {
            return false;
        };
        let (import, name) = match path {
            [name] => {
                if module.scope.bindings.contains_key(name) {
                    return false;
                }
                if module.scope.children.contains_key(name) {
                    return !self.class_is_value_like(module_path, name, depth);
                }
                let Some(import) = module.imports.get(name) else {
                    return false;
                };
                let Some(imported) = &import.imported_name else {
                    return false;
                };
                (import, imported)
            }
            [module_name, name] => match module.imports.get(module_name) {
                Some(import) if import.imported_name.is_none() => (import, name),
                _ => return false,
            },
            _ => return false,
        };
        if is_skipped_module(&import.module_path) {
            return STDLIB_OBJECT_TYPES.contains(&name.as_str());
        }
        self.ensure_loaded(&import.module_path, depth + 1);
        self.is_object_class(&import.module_path, std::slice::from_ref(name), depth + 1)
    }

    /// Enums, `str`/`int` subclasses, and classes derived from them have value-like pytest IDs.
    fn class_is_value_like(&self, module_path: &[String], class_name: &str, depth: usize) -> bool {
        if depth > MAX_IMPORT_DEPTH {
            return true;
        }
        let Some(module) = self.modules.borrow().get(module_path).map(Rc::clone) else {
            return true;
        };
        let Some(class) = module.scope.children.get(class_name) else {
            return true;
        };
        class.bases.iter().any(|base| {
            let last = base.rsplit('.').next().unwrap_or(base);
            if is_enum_class_name(base)
                || is_enum_class_name(last)
                || VALUE_TYPE_NAMES.contains(&last)
            {
                return true;
            }
            if module.scope.children.contains_key(base.as_str()) {
                return self.class_is_value_like(module_path, base, depth + 1);
            }
            match module.imports.get(base.as_str()) {
                Some(import) if !is_skipped_module(&import.module_path) => {
                    let Some(imported) = &import.imported_name else {
                        return true;
                    };
                    self.ensure_loaded(&import.module_path, depth + 1);
                    let loaded = self.modules.borrow().contains_key(&import.module_path);
                    loaded && self.class_is_value_like(&import.module_path, imported, depth + 1)
                }
                _ => false,
            }
        })
    }

    /// Resolve an expression the way module constants are resolved, with comprehension locals.
    pub(crate) fn resolve_expr(
        &self,
        enclosing_class: Option<&str>,
        expr: &Expr,
        locals: &LocalValues,
    ) -> Option<LiteralValue> {
        let pending = {
            let modules = self.modules.borrow();
            pending_value(expr, &modules.get(&self.current_path)?.names, locals)?
        };
        self.resolve_pending(&self.current_path, enclosing_class, &pending, 0)
    }

    fn resolve_pending(
        &self,
        module_path: &[String],
        enclosing_class: Option<&str>,
        pending: &PendingValue,
        depth: usize,
    ) -> Option<LiteralValue> {
        let resolve =
            |value: &PendingValue| self.resolve_pending(module_path, enclosing_class, value, depth);
        match pending {
            PendingValue::Literal(value) => Some(value.clone()),
            PendingValue::Path(path) => {
                let name: UnqualifiedName = path.iter().map(String::as_str).collect();
                let value =
                    self.lookup_path_at(module_path, enclosing_class, name.segments(), depth)?;
                Some(value.into_literal(path))
            }
            PendingValue::Sequence(items) => Some(LiteralValue::Sequence(
                items.iter().map(resolve).collect::<Option<_>>()?,
            )),
            PendingValue::Mapping(items) => {
                let mut entries: Vec<(LiteralValue, Option<LiteralValue>)> = Vec::new();
                for (key, value) in items {
                    let key = resolve(key)?;
                    if matches!(key, LiteralValue::Opaque | LiteralValue::Mapping(_)) {
                        return Some(LiteralValue::Opaque);
                    }
                    let value = value.as_ref().and_then(resolve);
                    match entries.iter_mut().find(|(existing, _)| *existing == key) {
                        Some(entry) => entry.1 = value,
                        None => entries.push((key, value)),
                    }
                }
                Some(LiteralValue::Mapping(entries))
            }
            PendingValue::Elements {
                source,
                view,
                sorted,
            } => {
                let mut items = match (resolve(source)?, view) {
                    (LiteralValue::Sequence(items), ElementView::Keys) => items,
                    (LiteralValue::Mapping(entries), ElementView::Keys) => {
                        entries.into_iter().map(|(key, _)| key).collect()
                    }
                    (LiteralValue::Mapping(entries), ElementView::Values) => entries
                        .into_iter()
                        .map(|(_, value)| value)
                        .collect::<Option<_>>()?,
                    _ => return None,
                };
                if *sorted {
                    sort_literals(&mut items)?;
                }
                Some(LiteralValue::Sequence(items))
            }
            PendingValue::BinOp(left, op, right) => {
                super::cases::eval_literal_binop(*op, &resolve(left)?, &resolve(right)?)
            }
            PendingValue::Param { values, id } => Some(LiteralValue::Param {
                values: values.iter().map(resolve).collect::<Option<_>>()?,
                id: id.clone(),
            }),
            PendingValue::CallResult(callee) => {
                let (def_module, annotation) =
                    self.function_return(module_path, enclosing_class, callee, depth)?;
                self.annotation_is_object(&def_module, &annotation, depth)
                    .then_some(LiteralValue::Opaque)
            }
            PendingValue::Slice {
                source,
                start,
                stop,
            } => {
                let LiteralValue::Sequence(items) = resolve(source)? else {
                    return None;
                };
                let len = items.len() as i64;
                let clamp = |bound: Option<i64>, default: i64| {
                    let index = bound.map_or(default, |i| if i < 0 { len + i } else { i });
                    index.clamp(0, len) as usize
                };
                let (start, stop) = (clamp(*start, 0), clamp(*stop, len));
                Some(LiteralValue::Sequence(
                    items
                        .get(start..stop.max(start))
                        .unwrap_or_default()
                        .to_vec(),
                ))
            }
        }
    }

    fn lookup_class_path(
        &self,
        module_path: &[String],
        class_name: &str,
        rest: &[&str],
        depth: usize,
    ) -> Option<ResolvedValue> {
        match rest {
            [member] => self.lookup_class_member(module_path, class_name, member, depth),
            _ => {
                let modules = self.modules.borrow();
                let module = modules.get(module_path)?;
                let path: Vec<&str> = std::iter::once(class_name)
                    .chain(rest.iter().copied())
                    .collect();
                module.scope.resolve_path(&path).map(ResolvedValue::Literal)
            }
        }
    }

    fn lookup_class_member(
        &self,
        module_path: &[String],
        class_name: &str,
        member: &str,
        depth: usize,
    ) -> Option<ResolvedValue> {
        let module = Rc::clone(self.modules.borrow().get(module_path)?);
        if let Some(value) = module.scope.lookup_class_member(class_name, member) {
            let is_member = !matches!(value, LiteralValue::Named(_));
            return Some(if is_member && module.scope.class_is_enum(class_name) {
                ResolvedValue::EnumMember
            } else {
                ResolvedValue::Literal(value)
            });
        }
        let class = module.scope.children.get(class_name)?;
        if class.bindings.contains_key(member) || depth > MAX_IMPORT_DEPTH {
            return None;
        }
        // Without computing the MRO, accept a member only when every base defining it agrees.
        let mut found = None;
        for base in &class.bases {
            let value = if module.scope.children.contains_key(base.as_str()) {
                self.lookup_class_member(module_path, base, member, depth + 1)
            } else {
                let import = module.imports.get(base.as_str())?;
                let imported_name = import.imported_name.as_deref()?;
                self.ensure_loaded(&import.module_path, depth + 1);
                self.lookup_class_member(&import.module_path, imported_name, member, depth + 1)
            };
            match (value, &found) {
                (Some(value), None) => found = Some(value),
                (Some(value), Some(existing)) if value == *existing => {}
                (Some(_), Some(_)) => return None,
                (None, _) => {}
            }
        }
        found
    }
}

struct PendingAssignment<'a> {
    module_path: &'a [String],
    value: &'a RefCell<BindingValue>,
    pending: Rc<PendingValue>,
}

fn collect_imports(module: &ModModule, context: &[String]) -> HashMap<String, ImportBinding> {
    let mut imports = HashMap::new();

    for stmt in &module.body {
        match stmt {
            Stmt::ImportFrom(import_from) => {
                let level = import_from.level as usize;
                let module_path = if let Some(module_name) = &import_from.module {
                    module_name.split('.').map(String::from).collect()
                } else {
                    Vec::new()
                };

                let Some(module_path) = resolve_relative_import(context, level, &module_path)
                else {
                    continue;
                };
                for alias in &import_from.names {
                    let imported_name = alias.name.to_string();
                    if imported_name == "*" {
                        continue;
                    }
                    let local_name = alias
                        .asname
                        .as_ref()
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| imported_name.clone());
                    imports.insert(
                        local_name,
                        ImportBinding {
                            module_path: module_path.clone(),
                            imported_name: Some(imported_name),
                        },
                    );
                }
            }
            Stmt::Import(import) => {
                for alias in &import.names {
                    let parts: Vec<String> = alias.name.split('.').map(String::from).collect();
                    if let Some(asname) = &alias.asname {
                        imports.insert(
                            asname.to_string(),
                            ImportBinding {
                                module_path: parts,
                                imported_name: None,
                            },
                        );
                    } else if let Some(first) = parts.first() {
                        imports.insert(
                            first.clone(),
                            ImportBinding {
                                module_path: vec![first.clone()],
                                imported_name: None,
                            },
                        );
                    }
                }
            }
            _ => {}
        }
    }

    imports
}

fn is_skipped_module(module_path: &[String]) -> bool {
    module_path.first().is_some_and(|name| {
        is_known_standard_library(11, name) || name == "pytest" || name == "rtest"
    })
}

/// Try to extract a literal value from an expression.
///
/// This is intentionally limited to simple literals to avoid complexity.
pub(crate) fn try_extract_literal(expr: &Expr) -> Option<LiteralValue> {
    match expr {
        Expr::NumberLiteral(num) => {
            use ruff_python_ast::Number;
            match &num.value {
                Number::Int(i) => Some(
                    i.as_i64()
                        .map_or_else(|| LiteralValue::String(i.to_string()), LiteralValue::Int),
                ),
                Number::Float(f) => Some(LiteralValue::Float(*f)),
                Number::Complex { .. } => None,
            }
        }
        Expr::UnaryOp(unary) => {
            use ruff_python_ast::UnaryOp;
            if !matches!(unary.op, UnaryOp::USub) {
                return None;
            }
            match try_extract_literal(&unary.operand) {
                Some(LiteralValue::Int(i)) => i.checked_neg().map(LiteralValue::Int),
                Some(LiteralValue::Float(f)) => Some(LiteralValue::Float(-f)),
                _ => None,
            }
        }
        Expr::StringLiteral(s) => Some(LiteralValue::String(s.value.to_str().to_string())),
        Expr::BytesLiteral(bytes) => Some(LiteralValue::String(
            bytes.value.bytes().map(char::from).collect(),
        )),
        Expr::BooleanLiteral(b) => Some(LiteralValue::Bool(b.value)),
        Expr::NoneLiteral(_) => Some(LiteralValue::None),
        Expr::List(ruff_python_ast::ExprList { elts, .. })
        | Expr::Tuple(ruff_python_ast::ExprTuple { elts, .. }) => {
            let values: Option<Vec<_>> = elts.iter().map(try_extract_literal).collect();
            Some(LiteralValue::Sequence(values?))
        }
        _ => None,
    }
}
