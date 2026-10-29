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
    /// Every function defined in this scope, whatever its decorators.
    defined_functions: std::collections::HashSet<String>,
    /// Bodies of parameterless functions, static methods and `cls`-only class methods that
    /// only `return` an expression; `None` when the name is defined more than once.
    return_values: HashMap<String, Option<ReturnedExpression>>,
    /// Assigned expressions that may be type aliases (`X = Literal[...]`); `None` when reassigned.
    aliases: HashMap<String, Option<Expr>>,
    /// For a class scope: instances may not be plain objects (`__new__`, a metaclass,
    /// attribute hooks, an own `__name__`, or a decorator that may replace the class).
    custom_construction: bool,
    /// Runtime fixture parametrization cannot be represented by decorator-only IDs.
    runtime_parameters: bool,
    /// For a class scope: how its constructor sets attributes, when that is plain.
    record: Option<RecordShape>,
}

/// Constructor of a class whose instances only hold their arguments as attributes: a
/// `@dataclass` without `__post_init__`, a `NamedTuple`, or an `__init__` that only runs
/// `self.attribute = parameter`.
#[derive(Debug)]
struct RecordShape {
    /// Constructor parameters in signature order, after those of `base`.
    params: Vec<RecordParam>,
    dataclass: bool,
    /// The single base of a dataclass, whose fields must also come from a record dataclass.
    base: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
struct RecordParam {
    name: String,
    keyword_only: bool,
    /// `None` when required; `Some(None)` when the default is not statically known.
    default: Option<Option<PendingValue>>,
    /// Instance attributes assigned from this parameter.
    attributes: Vec<String>,
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
    /// `a | b`: an operator on values, or a union of type forms.
    TypeOr {
        union: Box<Expr>,
        value: Box<PendingValue>,
    },
    Param {
        values: Vec<PendingValue>,
        id: Option<String>,
    },
    Slice {
        source: Box<PendingValue>,
        start: Option<i128>,
        stop: Option<i128>,
    },
    /// `seq[i]` or `mapping[key]`.
    Index {
        source: Box<PendingValue>,
        key: Box<PendingValue>,
    },
    /// An f-string whose interpolations use `str()` without a format spec.
    FString(Vec<PendingValue>),
    /// A call whose result is IDed positionally when it constructs an object or its
    /// return annotation is an object type.
    CallResult(Vec<String>),
    /// `*X` inside a list or tuple display; `X` must resolve to a sequence.
    Unpack(Box<PendingValue>),
    /// A call that builds a record when the callee has a [`RecordShape`], and `otherwise`
    /// when it does not. Arguments are `None` when not statically known.
    Construct {
        callee: Vec<String>,
        args: Vec<Option<PendingValue>>,
        keywords: Vec<(String, Option<PendingValue>)>,
        otherwise: Option<Box<PendingValue>>,
    },
    /// `get_args(X)`: the values of a `Literal[...]` type when the callee is `typing.get_args`.
    LiteralArgs {
        callee: Vec<String>,
        argument: Box<Expr>,
    },
    /// A call whose result depends on the callee's configuration: a one-argument
    /// literal-values function returns the `Literal` type's values, a passthrough function
    /// returns its first argument, and any other callee gives `otherwise`.
    ConfiguredCall {
        callee: Vec<String>,
        /// The first positional argument, as an expression for literal-values functions.
        argument: Box<Expr>,
        /// The first positional argument as a value, for passthrough functions.
        argument_value: Option<Box<PendingValue>>,
        single_argument: bool,
        otherwise: Option<Box<PendingValue>>,
    },
    /// A list comprehension or generator expression, with the enclosing comprehension's
    /// variables already bound.
    Comprehension {
        elt: Box<Expr>,
        generators: Vec<ruff_python_ast::Comprehension>,
        locals: LocalValues,
    },
    /// `set(x)`, `frozenset(x)`, `{a, b}` or a set comprehension over the elements of `x`.
    SetOf(Box<PendingValue>),
    /// `Base[args]` where `Base` may be a generic type: a builtin like `list` or a typing
    /// name, whose aliases pytest IDs by `__name__`; otherwise it is the `index` lookup.
    Generic {
        base: Vec<String>,
        builtin: Option<&'static str>,
        args: Box<Expr>,
        index: Option<Box<PendingValue>>,
    },
    /// `enumerate(x, start)`.
    Enumerate {
        source: Box<PendingValue>,
        start: i128,
    },
    /// `zip(a, b, ...)`; with `strict=True` the sources must have equal lengths.
    Zip {
        sources: Vec<PendingValue>,
        strict: bool,
    },
}

/// Imported callables known to return objects that pytest IDs positionally.
///
/// Matched by fully qualified import name, so a local function with the same name is not affected.
const OBJECT_RETURNING_CALLABLES: &[&str] = &[
    "contextlib.nullcontext",
    "dataclasses.replace",
    "datetime.timezone",
    "dateutil.relativedelta.relativedelta",
    "pyarrow.array",
    "pytest.raises",
    "uuid.UUID",
];

const GET_ARGS_CALLABLES: &[&str] = &["typing.get_args", "typing_extensions.get_args"];
const LITERAL_TYPES: &[&str] = &["typing.Literal", "typing_extensions.Literal"];
const UNION_TYPES: &[&str] = &["typing.Union", "typing_extensions.Union"];

/// Members of `http.HTTPStatus` and `pytest.ExitCode`. Both are `IntEnum`s, which pytest IDs
/// by their integer value.
const INT_ENUM_MEMBERS: &[(&str, i64)] = &[
    ("http.HTTPStatus.ACCEPTED", 202),
    ("http.HTTPStatus.ALREADY_REPORTED", 208),
    ("http.HTTPStatus.BAD_GATEWAY", 502),
    ("http.HTTPStatus.BAD_REQUEST", 400),
    ("http.HTTPStatus.CONFLICT", 409),
    ("http.HTTPStatus.CONTENT_TOO_LARGE", 413),
    ("http.HTTPStatus.CONTINUE", 100),
    ("http.HTTPStatus.CREATED", 201),
    ("http.HTTPStatus.EARLY_HINTS", 103),
    ("http.HTTPStatus.EXPECTATION_FAILED", 417),
    ("http.HTTPStatus.FAILED_DEPENDENCY", 424),
    ("http.HTTPStatus.FORBIDDEN", 403),
    ("http.HTTPStatus.FOUND", 302),
    ("http.HTTPStatus.GATEWAY_TIMEOUT", 504),
    ("http.HTTPStatus.GONE", 410),
    ("http.HTTPStatus.HTTP_VERSION_NOT_SUPPORTED", 505),
    ("http.HTTPStatus.IM_A_TEAPOT", 418),
    ("http.HTTPStatus.IM_USED", 226),
    ("http.HTTPStatus.INSUFFICIENT_STORAGE", 507),
    ("http.HTTPStatus.INTERNAL_SERVER_ERROR", 500),
    ("http.HTTPStatus.LENGTH_REQUIRED", 411),
    ("http.HTTPStatus.LOCKED", 423),
    ("http.HTTPStatus.LOOP_DETECTED", 508),
    ("http.HTTPStatus.METHOD_NOT_ALLOWED", 405),
    ("http.HTTPStatus.MISDIRECTED_REQUEST", 421),
    ("http.HTTPStatus.MOVED_PERMANENTLY", 301),
    ("http.HTTPStatus.MULTIPLE_CHOICES", 300),
    ("http.HTTPStatus.MULTI_STATUS", 207),
    ("http.HTTPStatus.NETWORK_AUTHENTICATION_REQUIRED", 511),
    ("http.HTTPStatus.NON_AUTHORITATIVE_INFORMATION", 203),
    ("http.HTTPStatus.NOT_ACCEPTABLE", 406),
    ("http.HTTPStatus.NOT_EXTENDED", 510),
    ("http.HTTPStatus.NOT_FOUND", 404),
    ("http.HTTPStatus.NOT_IMPLEMENTED", 501),
    ("http.HTTPStatus.NOT_MODIFIED", 304),
    ("http.HTTPStatus.NO_CONTENT", 204),
    ("http.HTTPStatus.OK", 200),
    ("http.HTTPStatus.PARTIAL_CONTENT", 206),
    ("http.HTTPStatus.PAYMENT_REQUIRED", 402),
    ("http.HTTPStatus.PERMANENT_REDIRECT", 308),
    ("http.HTTPStatus.PRECONDITION_FAILED", 412),
    ("http.HTTPStatus.PRECONDITION_REQUIRED", 428),
    ("http.HTTPStatus.PROCESSING", 102),
    ("http.HTTPStatus.PROXY_AUTHENTICATION_REQUIRED", 407),
    ("http.HTTPStatus.RANGE_NOT_SATISFIABLE", 416),
    ("http.HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE", 416),
    ("http.HTTPStatus.REQUEST_ENTITY_TOO_LARGE", 413),
    ("http.HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE", 431),
    ("http.HTTPStatus.REQUEST_TIMEOUT", 408),
    ("http.HTTPStatus.REQUEST_URI_TOO_LONG", 414),
    ("http.HTTPStatus.RESET_CONTENT", 205),
    ("http.HTTPStatus.SEE_OTHER", 303),
    ("http.HTTPStatus.SERVICE_UNAVAILABLE", 503),
    ("http.HTTPStatus.SWITCHING_PROTOCOLS", 101),
    ("http.HTTPStatus.TEMPORARY_REDIRECT", 307),
    ("http.HTTPStatus.TOO_EARLY", 425),
    ("http.HTTPStatus.TOO_MANY_REQUESTS", 429),
    ("http.HTTPStatus.UNAUTHORIZED", 401),
    ("http.HTTPStatus.UNAVAILABLE_FOR_LEGAL_REASONS", 451),
    ("http.HTTPStatus.UNPROCESSABLE_CONTENT", 422),
    ("http.HTTPStatus.UNPROCESSABLE_ENTITY", 422),
    ("http.HTTPStatus.UNSUPPORTED_MEDIA_TYPE", 415),
    ("http.HTTPStatus.UPGRADE_REQUIRED", 426),
    ("http.HTTPStatus.URI_TOO_LONG", 414),
    ("http.HTTPStatus.USE_PROXY", 305),
    ("http.HTTPStatus.VARIANT_ALSO_NEGOTIATES", 506),
    ("pytest.ExitCode.INTERNAL_ERROR", 3),
    ("pytest.ExitCode.INTERRUPTED", 2),
    ("pytest.ExitCode.MAX_WARNINGS_ERROR", 6),
    ("pytest.ExitCode.NO_TESTS_COLLECTED", 5),
    ("pytest.ExitCode.OK", 0),
    ("pytest.ExitCode.TESTS_FAILED", 1),
    ("pytest.ExitCode.USAGE_ERROR", 4),
];

/// Builtin classes whose subscription (`list[int]`) is a generic alias named like the class.
const GENERIC_BUILTINS: &[&str] = &["dict", "frozenset", "list", "set", "tuple", "type"];

/// Typing generics whose subscription keeps the generic's `__name__`.
const TYPING_GENERICS: &[&str] = &[
    "Annotated",
    "Callable",
    "DefaultDict",
    "Dict",
    "FrozenSet",
    "Iterable",
    "List",
    "Literal",
    "Mapping",
    "Sequence",
    "Set",
    "Tuple",
    "Type",
];

/// Class decorators that return the decorated class itself.
const TRANSPARENT_CLASS_DECORATORS: &[&str] = &["dataclass", "final", "total_ordering"];

/// Class members that change how instances are created or which `__name__` they expose.
const CUSTOM_CONSTRUCTION_MEMBERS: &[&str] =
    &["__new__", "__getattr__", "__getattribute__", "__name__"];

/// A callee reached by following module names and imports.
#[derive(Debug, Clone, PartialEq)]
enum Callee {
    /// A class defined in a parsed module, by its path inside that module.
    Class {
        module: Vec<String>,
        path: Vec<String>,
    },
    /// A name from a module that is not parsed (stdlib or third-party), fully qualified.
    External(String),
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
    /// `(key, value)` pairs of a mapping.
    Items,
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
            .chain(&["range", "enumerate", "zip"])
            .find(|builtin| **builtin == name)
            .copied()
    }
}

/// Values bound by an enclosing comprehension, which shadow module names.
pub(crate) type LocalValues = HashMap<String, LiteralValue>;

/// A member of a type union, as far as it decides the union's `__name__`.
enum TypeMember {
    None,
    /// A class, by `__name__`; `key` identifies the class object.
    Class {
        name: String,
        key: String,
    },
    /// `Literal[...]`, which compares equal to another holding the same set of values.
    Literal(Vec<LiteralValue>),
    /// A subscripted generic such as `list[int]`; `key` identifies its spelling.
    Generic {
        name: String,
        key: String,
    },
}

fn generic_key(module_path: &[String], expr: &Expr) -> String {
    format!(
        "{}:{:?}",
        module_path.join("."),
        ruff_python_ast::comparable::ComparableExpr::from(expr)
    )
}

/// The value of a union of these members, which drops duplicates. A single remaining type
/// is that type itself. Otherwise a `typing.Union` is named `Union`, or `Optional` for one
/// type plus `None`, and a `types.UnionType` (`int | None`) has no `__name__`. Members that
/// may or may not be equal (two classes sharing a name, two spellings of one generic) have
/// no static value.
fn union_value(members: Vec<TypeMember>, typing: bool) -> Option<LiteralValue> {
    let mut distinct: Vec<TypeMember> = Vec::new();
    let mut has_none = false;
    for member in members {
        let duplicate = match &member {
            TypeMember::None => {
                has_none = true;
                continue;
            }
            TypeMember::Class { name, key } => {
                let mut duplicate = false;
                for other in &distinct {
                    if let TypeMember::Class { name: other_name, key: other_key } = other {
                        if other_key == key {
                            duplicate = true;
                        } else if other_name == name {
                            return None;
                        }
                    }
                }
                duplicate
            }
            TypeMember::Literal(values) => distinct.iter().any(|other| {
                matches!(other, TypeMember::Literal(other)
                    if other.len() == values.len() && other.iter().all(|value| values.contains(value)))
            }),
            TypeMember::Generic { name, key } => {
                let mut duplicate = false;
                for other in &distinct {
                    if let TypeMember::Generic { name: other_name, key: other_key } = other {
                        if other_key == key {
                            duplicate = true;
                        } else if other_name == name {
                            return None;
                        }
                    }
                }
                duplicate
            }
        };
        if !duplicate {
            distinct.push(member);
        }
    }
    let name = match (distinct.as_slice(), has_none) {
        ([single], false) => match single {
            TypeMember::Class { name, .. } | TypeMember::Generic { name, .. } => name.clone(),
            TypeMember::Literal(_) => "Literal".to_string(),
            TypeMember::None => return None,
        },
        ([], _) => return None,
        _ if !typing => return Some(LiteralValue::Opaque),
        ([_], true) => "Optional".to_string(),
        _ => "Union".to_string(),
    };
    Some(LiteralValue::Named(name))
}

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
            if let Some(value) = locals.get(segments[0]) {
                return match segments[1..] {
                    [attribute] => attribute_value(value, attribute).map(PendingValue::Literal),
                    _ => None,
                };
            }
            Some(PendingValue::Path(
                segments.iter().map(|s| (*s).to_string()).collect(),
            ))
        }
        Expr::List(ruff_python_ast::ExprList { elts, .. })
        | Expr::Tuple(ruff_python_ast::ExprTuple { elts, .. }) => {
            pending_sequence(elts, names, locals)
        }
        Expr::Dict(dict) => Some(PendingValue::Mapping(
            dict.items
                .iter()
                .map(|item| match item.key.as_ref() {
                    Some(key) => Some((
                        pending_value(key, names, locals)?,
                        pending_value(&item.value, names, locals),
                    )),
                    // `**mapping` is stored as an unpacked key without a value.
                    None => Some((
                        PendingValue::Unpack(Box::new(pending_value(&item.value, names, locals)?)),
                        None,
                    )),
                })
                .collect::<Option<_>>()?,
        )),
        // A set with unknown members still counts as one positionally IDed value.
        Expr::Set(set) => Some(
            pending_sequence(&set.elts, names, locals)
                .map_or(PendingValue::Literal(LiteralValue::Opaque), |items| {
                    PendingValue::SetOf(Box::new(items))
                }),
        ),
        Expr::ListComp(comp) => Some(comprehension(&comp.elt, &comp.generators, locals)),
        Expr::Generator(comp) => Some(comprehension(&comp.elt, &comp.generators, locals)),
        Expr::SetComp(comp) => Some(PendingValue::SetOf(Box::new(comprehension(
            &comp.elt,
            &comp.generators,
            locals,
        )))),
        Expr::BinOp(binop) if binop.op == ruff_python_ast::Operator::BitOr && locals.is_empty() => {
            Some(PendingValue::TypeOr {
                union: Box::new(expr.clone()),
                value: Box::new(PendingValue::BinOp(
                    Box::new(pending_value(&binop.left, names, locals)?),
                    binop.op,
                    Box::new(pending_value(&binop.right, names, locals)?),
                )),
            })
        }
        Expr::BinOp(binop) => Some(PendingValue::BinOp(
            Box::new(pending_value(&binop.left, names, locals)?),
            binop.op,
            Box::new(pending_value(&binop.right, names, locals)?),
        )),
        Expr::FString(fstring) => {
            let mut parts = Vec::new();
            for part in fstring.value.iter() {
                match part {
                    ruff_python_ast::FStringPartRef::Literal(literal) => parts.push(
                        PendingValue::Literal(LiteralValue::String(literal.value.to_string())),
                    ),
                    ruff_python_ast::FStringPartRef::FString(fstring) => {
                        for element in fstring.elements.iter() {
                            match element {
                                ruff_python_ast::InterpolatedStringElement::Literal(literal) => {
                                    parts.push(PendingValue::Literal(LiteralValue::String(
                                        literal.value.to_string(),
                                    )))
                                }
                                ruff_python_ast::InterpolatedStringElement::Interpolation(
                                    interpolation,
                                ) => {
                                    if interpolation.debug_text.is_some()
                                        || interpolation.format_spec.is_some()
                                        || !matches!(
                                            interpolation.conversion,
                                            ruff_python_ast::ConversionFlag::None
                                                | ruff_python_ast::ConversionFlag::Str
                                        )
                                    {
                                        return None;
                                    }
                                    parts.push(pending_value(
                                        &interpolation.expression,
                                        names,
                                        locals,
                                    )?);
                                }
                            }
                        }
                    }
                }
            }
            Some(PendingValue::FString(parts))
        }
        Expr::Subscript(subscript) => {
            let Expr::Slice(slice) = subscript.slice.as_ref() else {
                let index = (|| {
                    Some(PendingValue::Index {
                        source: Box::new(pending_value(&subscript.value, names, locals)?),
                        key: Box::new(pending_value(&subscript.slice, names, locals)?),
                    })
                })();
                return match expr_segments(&subscript.value) {
                    Some(base) if !locals.contains_key(&base[0]) => Some(PendingValue::Generic {
                        builtin: names
                            .builtin(&subscript.value)
                            .filter(|name| GENERIC_BUILTINS.contains(name)),
                        base,
                        args: subscript.slice.clone(),
                        index: index.map(Box::new),
                    }),
                    _ => index,
                };
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

fn pending_sequence(
    elts: &[Expr],
    names: &ModuleNames,
    locals: &LocalValues,
) -> Option<PendingValue> {
    Some(PendingValue::Sequence(
        elts.iter()
            .map(|elt| match elt {
                Expr::Starred(starred) => Some(PendingValue::Unpack(Box::new(pending_value(
                    &starred.value,
                    names,
                    locals,
                )?))),
                elt => pending_value(elt, names, locals),
            })
            .collect::<Option<_>>()?,
    ))
}

fn comprehension(
    elt: &Expr,
    generators: &[ruff_python_ast::Comprehension],
    locals: &LocalValues,
) -> PendingValue {
    PendingValue::Comprehension {
        elt: Box::new(elt.clone()),
        generators: generators.to_vec(),
        locals: locals.clone(),
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
    if let (Some(callee), [argument], true) = (
        UnqualifiedName::from_expr(&call.func),
        args.args.as_ref(),
        args.keywords.is_empty(),
    ) {
        if callee.segments().last() == Some(&"get_args")
            && !locals.contains_key(callee.segments()[0])
        {
            return Some(PendingValue::LiteralArgs {
                callee: callee.segments().iter().map(|s| (*s).to_string()).collect(),
                argument: Box::new(argument.clone()),
            });
        }
    }
    if let Expr::Attribute(attr) = call.func.as_ref() {
        let view = match attr.attr.as_str() {
            "keys" => ElementView::Keys,
            "values" => ElementView::Values,
            "items" => ElementView::Items,
            _ => {
                let otherwise = record_call(call, names, locals, constructor_value(call, locals));
                return configured_call(call, otherwise, names, locals);
            }
        };
        if !args.is_empty() {
            return configured_call(call, None, names, locals);
        }
        return Some(PendingValue::Elements {
            source: Box::new(pending_value(&attr.value, names, locals)?),
            view,
            sorted: false,
        });
    }
    let Some(builtin) = names.builtin(&call.func) else {
        let otherwise = record_call(call, names, locals, constructor_value(call, locals));
        return configured_call(call, otherwise, names, locals);
    };
    match (builtin, args.args.as_ref(), args.keywords.is_empty()) {
        ("range", bounds, true) if (1..=3).contains(&bounds.len()) => {
            let bounds: Vec<i128> = bounds
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
        ("enumerate", [source, rest @ ..], _) if rest.len() + args.keywords.len() <= 1 => {
            let start = match (rest, args.keywords.as_ref()) {
                ([], []) => None,
                ([start], []) => Some(start),
                ([], [keyword]) if keyword.arg.as_deref() == Some("start") => Some(&keyword.value),
                _ => return None,
            };
            let start = match start.map(try_extract_literal) {
                None => 0,
                Some(Some(LiteralValue::Int(start))) => start,
                Some(_) => return None,
            };
            Some(PendingValue::Enumerate {
                source: Box::new(pending_value(source, names, locals)?),
                start,
            })
        }
        ("zip", sources, _) if !sources.iter().any(Expr::is_starred_expr) => {
            let mut strict = false;
            for keyword in &args.keywords {
                match (keyword.arg.as_deref(), try_extract_literal(&keyword.value)) {
                    (Some("strict"), Some(LiteralValue::Bool(value))) => strict = value,
                    _ => return None,
                }
            }
            Some(PendingValue::Zip {
                sources: sources
                    .iter()
                    .map(|source| pending_value(source, names, locals))
                    .collect::<Option<_>>()?,
                strict,
            })
        }
        ("set" | "frozenset", [], true) => {
            Some(PendingValue::Literal(LiteralValue::Set(Vec::new())))
        }
        ("set" | "frozenset", [source], true) => Some(
            pending_value(source, names, locals)
                .map_or(PendingValue::Literal(LiteralValue::Opaque), |source| {
                    PendingValue::SetOf(Box::new(source))
                }),
        ),
        (container, _, _) if CONTAINER_BUILTINS.contains(&container) => {
            Some(PendingValue::Literal(LiteralValue::Opaque))
        }
        ("object", [], true) => Some(PendingValue::Literal(LiteralValue::Opaque)),
        ("float", [value], true) => {
            let value = match try_extract_literal(value)? {
                LiteralValue::String(text) => parse_python_float(&text)?,
                LiteralValue::Float(value) => value,
                LiteralValue::Int(value) if value.unsigned_abs() <= 1 << f64::MANTISSA_DIGITS => {
                    value as f64
                }
                _ => return None,
            };
            Some(PendingValue::Literal(LiteralValue::Float(value)))
        }
        _ => None,
    }
}

/// `float(text)` for the spellings Python accepts that Rust parses identically: decimal and
/// exponent notation, `nan`, and `inf`/`infinity`, case-insensitively and surrounded by
/// whitespace. Underscore digit grouping is left to Python.
fn parse_python_float(text: &str) -> Option<f64> {
    let text = text.trim().to_ascii_lowercase();
    let (negative, magnitude) = match text.as_bytes().first()? {
        b'-' => (true, &text[1..]),
        b'+' => (false, &text[1..]),
        _ => (false, text.as_str()),
    };
    let value = match magnitude {
        "nan" => f64::NAN,
        "inf" | "infinity" => f64::INFINITY,
        digits
            if digits.starts_with(|c: char| c.is_ascii_digit() || c == '.')
                && digits
                    .chars()
                    .all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | '+' | '-')) =>
        {
            digits.parse::<f64>().ok()?
        }
        _ => return None,
    };
    Some(if negative { -value } else { value })
}

/// `Class(...)`: possibly a record construction, decided once names are resolved.
fn record_call(
    call: &ruff_python_ast::ExprCall,
    names: &ModuleNames,
    locals: &LocalValues,
    otherwise: Option<PendingValue>,
) -> Option<PendingValue> {
    let arguments = &call.arguments;
    let Some(callee) = expr_segments(&call.func) else {
        return otherwise;
    };
    if locals.contains_key(&callee[0])
        || arguments.args.iter().any(Expr::is_starred_expr)
        || arguments
            .keywords
            .iter()
            .any(|keyword| keyword.arg.is_none())
    {
        return otherwise;
    }
    Some(PendingValue::Construct {
        callee,
        args: arguments
            .args
            .iter()
            .map(|arg| pending_value(arg, names, locals))
            .collect(),
        keywords: arguments
            .keywords
            .iter()
            .map(|keyword| {
                (
                    keyword
                        .arg
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_default(),
                    pending_value(&keyword.value, names, locals),
                )
            })
            .collect(),
        otherwise: otherwise.map(Box::new),
    })
}

/// The constructor shape of a plain record class, if it is one.
fn record_shape(
    class_def: &ruff_python_ast::StmtClassDef,
    scope: &ConstantScope,
    imports: &HashMap<String, ImportBinding>,
    names: &ModuleNames,
) -> Option<RecordShape> {
    if scope.custom_construction || !class_def.keywords().is_empty() {
        return None;
    }
    let qualified = |expr: &Expr| imported_expression_name(expr, imports);
    match (class_def.decorator_list.as_slice(), class_def.bases()) {
        ([decorator], bases) => {
            // `Generic[T]` adds no fields; a subscripted base is its class.
            let mut base = None;
            for class in bases {
                let class = match class {
                    Expr::Subscript(subscript) => subscript.value.as_ref(),
                    class => class,
                };
                if matches!(
                    qualified(class).as_deref(),
                    Some("typing.Generic" | "typing_extensions.Generic")
                ) {
                    continue;
                }
                if base.replace(expr_segments(class)?).is_some() {
                    return None;
                }
            }
            let (func, keywords) = match &decorator.expression {
                Expr::Call(call) if call.arguments.args.is_empty() => {
                    (call.func.as_ref(), &call.arguments.keywords[..])
                }
                Expr::Call(_) => return None,
                func => (func, &[][..]),
            };
            let plain_options = keywords.iter().all(|keyword| {
                matches!(
                    keyword.arg.as_deref(),
                    Some(
                        "frozen" | "eq" | "order" | "unsafe_hash" | "repr" | "slots" | "match_args"
                    )
                )
            });
            let custom_init = ["__init__", "__post_init__", "__setattr__"]
                .iter()
                .any(|name| scope.defined_functions.contains(*name));
            let mut shape = (qualified(func).as_deref() == Some("dataclasses.dataclass")
                && plain_options
                && !custom_init)
                .then(|| annotated_fields(class_def, imports, names, true))??;
            shape.base = base;
            Some(shape)
        }
        ([], [base])
            if matches!(
                qualified(base).as_deref(),
                Some("typing.NamedTuple" | "typing_extensions.NamedTuple")
            ) =>
        {
            annotated_fields(class_def, imports, names, false)
        }
        ([], []) => init_fields(class_def, scope, names),
        _ => None,
    }
}

/// Fields of a dataclass or `NamedTuple`: annotated class attributes in definition order.
fn annotated_fields(
    class_def: &ruff_python_ast::StmtClassDef,
    imports: &HashMap<String, ImportBinding>,
    names: &ModuleNames,
    dataclass: bool,
) -> Option<RecordShape> {
    let locals = LocalValues::new();
    let mut params: Vec<RecordParam> = Vec::new();
    for stmt in &class_def.body {
        let Stmt::AnnAssign(ann) = stmt else {
            continue;
        };
        let Expr::Name(target) = ann.target.as_ref() else {
            return None;
        };
        let annotation = imported_expression_name(map_subscript(&ann.annotation), imports);
        match annotation.as_deref() {
            Some("typing.ClassVar" | "typing_extensions.ClassVar") if dataclass => continue,
            Some(
                "typing.ClassVar"
                | "typing_extensions.ClassVar"
                | "dataclasses.InitVar"
                | "dataclasses.KW_ONLY",
            ) => return None,
            _ => {}
        }
        let default = match ann.value.as_deref() {
            None => None,
            Some(Expr::Call(call))
                if dataclass
                    && imported_expression_name(&call.func, imports).as_deref()
                        == Some("dataclasses.field") =>
            {
                if !call.arguments.args.is_empty() {
                    return None;
                }
                let mut default = None;
                for keyword in &call.arguments.keywords {
                    match keyword.arg.as_deref() {
                        Some("default") => {
                            default = Some(pending_value(&keyword.value, names, &locals));
                        }
                        Some("default_factory") => default = Some(None),
                        Some("repr" | "hash" | "compare" | "metadata") => {}
                        _ => return None,
                    }
                }
                default
            }
            Some(value) => Some(pending_value(value, names, &locals)),
        };
        let name = target.id.to_string();
        if params.iter().any(|param| param.name == name) {
            return None;
        }
        params.push(RecordParam {
            attributes: vec![name.clone()],
            name,
            keyword_only: false,
            default,
        });
    }
    Some(RecordShape {
        params,
        dataclass,
        base: None,
    })
}

/// Fields of a class whose only `__init__` statements are `self.attribute = parameter`.
fn init_fields(
    class_def: &ruff_python_ast::StmtClassDef,
    scope: &ConstantScope,
    names: &ModuleNames,
) -> Option<RecordShape> {
    let mut inits = class_def.body.iter().filter_map(|stmt| match stmt {
        Stmt::FunctionDef(func) if func.name.as_str() == "__init__" => Some(func),
        _ => None,
    });
    let (Some(init), None) = (inits.next(), inits.next()) else {
        return None;
    };
    let parameters = &init.parameters;
    if !init.decorator_list.is_empty()
        || init.is_async
        || parameters.vararg.is_some()
        || parameters.kwarg.is_some()
        || !parameters.posonlyargs.is_empty()
    {
        return None;
    }
    let (receiver, positional) = parameters.args.split_first()?;
    if receiver.default.is_some() {
        return None;
    }
    let locals = LocalValues::new();
    let mut params: Vec<RecordParam> = positional
        .iter()
        .map(|param| (param, false))
        .chain(parameters.kwonlyargs.iter().map(|param| (param, true)))
        .map(|(param, keyword_only)| RecordParam {
            name: param.parameter.name.to_string(),
            keyword_only,
            default: param
                .default
                .as_deref()
                .map(|default| pending_value(default, names, &locals)),
            attributes: Vec::new(),
        })
        .collect();
    let receiver = receiver.parameter.name.as_str();
    let body = match init.body.split_first() {
        Some((Stmt::Expr(docstring), rest)) if docstring.value.is_string_literal_expr() => rest,
        _ => &init.body[..],
    };
    for stmt in body {
        let (target, value) = match stmt {
            Stmt::Pass(_) => continue,
            Stmt::Assign(assign) => match assign.targets.as_slice() {
                [target] => (target, assign.value.as_ref()),
                _ => return None,
            },
            Stmt::AnnAssign(ann) => (ann.target.as_ref(), ann.value.as_deref()?),
            _ => return None,
        };
        let (Expr::Attribute(target), Expr::Name(value)) = (target, value) else {
            return None;
        };
        let attribute = target.attr.to_string();
        // A class attribute of the same name may be a descriptor that intercepts the store.
        if target.value.as_name_expr()?.id.as_str() != receiver
            || scope.bindings.contains_key(&attribute)
            || scope.defined_functions.contains(&attribute)
        {
            return None;
        }
        for param in &mut params {
            param.attributes.retain(|existing| *existing != attribute);
        }
        params
            .iter_mut()
            .find(|param| param.name == value.id.as_str())?
            .attributes
            .push(attribute);
    }
    Some(RecordShape {
        params,
        dataclass: false,
        base: None,
    })
}

/// `f(X, ...)`, `Owner.f(X, ...)` or `Owner[T].f(X, ...)`: possibly a configured
/// literal-values or passthrough function, decided once names are resolved.
fn configured_call(
    call: &ruff_python_ast::ExprCall,
    otherwise: Option<PendingValue>,
    names: &ModuleNames,
    locals: &LocalValues,
) -> Option<PendingValue> {
    let Some(argument) = call
        .arguments
        .args
        .first()
        .filter(|arg| !arg.is_starred_expr())
    else {
        return otherwise;
    };
    let callee = match call.func.as_ref() {
        Expr::Attribute(attr) => {
            let owner = match attr.value.as_ref() {
                Expr::Subscript(subscript) => subscript.value.as_ref(),
                owner => owner,
            };
            let mut segments = expr_segments(owner)?;
            segments.push(attr.attr.to_string());
            segments
        }
        Expr::Name(_) => expr_segments(&call.func)?,
        _ => return otherwise,
    };
    if locals.contains_key(&callee[0]) {
        return otherwise;
    }
    Some(PendingValue::ConfiguredCall {
        callee,
        argument: Box::new(argument.clone()),
        argument_value: pending_value(argument, names, locals).map(Box::new),
        single_argument: call.arguments.args.len() == 1 && call.arguments.keywords.is_empty(),
        otherwise: otherwise.map(Box::new),
    })
}

/// Objects built by constructors are IDed positionally, so their arguments are irrelevant.
/// Other calls depend on the callee's return annotation.
/// An attribute of a resolved value: a record field, or the `__name__` of a class or function.
fn attribute_value(value: &LiteralValue, attribute: &str) -> Option<LiteralValue> {
    match (value, attribute) {
        (LiteralValue::Record(fields), attribute) => fields
            .iter()
            .find(|(name, _)| name == attribute)
            .and_then(|(_, value)| value.clone()),
        (LiteralValue::Named(name), "__name__") => Some(LiteralValue::String(name.clone())),
        _ => None,
    }
}

#[derive(Debug, Clone)]
struct ReturnedExpression {
    value: Expr,
    /// The `cls` parameter of a class method, whose attributes are not tracked.
    class_parameter: Option<String>,
}

/// The expression a function returns when its body is only that `return` (after an
/// optional docstring): a parameterless function or `@staticmethod`, or a `@classmethod`
/// whose only parameter is `cls`.
fn returned_expression(func: &ruff_python_ast::StmtFunctionDef) -> Option<ReturnedExpression> {
    let decorator = match func.decorator_list.as_slice() {
        [] => None,
        [decorator] => match &decorator.expression {
            Expr::Name(name) => Some(name.id.as_str()),
            _ => return None,
        },
        _ => return None,
    };
    let parameters = &func.parameters;
    let class_parameter = match decorator {
        _ if func.is_async => return None,
        None | Some("staticmethod") if parameters.is_empty() => None,
        Some("classmethod")
            if parameters.args.len() == 1
                && parameters.posonlyargs.is_empty()
                && parameters.kwonlyargs.is_empty()
                && parameters.vararg.is_none()
                && parameters.kwarg.is_none()
                && parameters.args[0].default.is_none() =>
        {
            Some(parameters.args[0].parameter.name.to_string())
        }
        _ => return None,
    };
    let body = match func.body.as_slice() {
        [Stmt::Expr(doc), rest @ ..] if doc.value.is_string_literal_expr() => rest,
        body => body,
    };
    match body {
        [Stmt::Return(ruff_python_ast::StmtReturn {
            value: Some(value), ..
        })] => Some(ReturnedExpression {
            value: value.as_ref().clone(),
            class_parameter,
        }),
        _ => None,
    }
}

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

/// A set of same-typed hashable scalars; mixed types could compare equal (`1 == True`).
fn set_value(items: Vec<LiteralValue>) -> Option<LiteralValue> {
    let mut members: Vec<LiteralValue> = Vec::with_capacity(items.len());
    for item in items {
        let same_kind = members
            .first()
            .is_none_or(|first| std::mem::discriminant(first) == std::mem::discriminant(&item));
        if !same_kind
            || !matches!(
                item,
                LiteralValue::String(_)
                    | LiteralValue::Int(_)
                    | LiteralValue::Bool(_)
                    | LiteralValue::EnumMember(_)
            )
        {
            return None;
        }
        if !members.contains(&item) {
            members.push(item);
        }
    }
    Some(LiteralValue::Set(members))
}

/// `sorted()` for homogeneous strings or ints; other mixes need Python's comparison rules.
fn sort_literals(items: &mut [LiteralValue]) -> Option<()> {
    if items
        .iter()
        .all(|item| matches!(item, LiteralValue::Sequence(pair) if pair.len() == 2))
    {
        // `sorted(d.items())`: dict keys are unique, so tuple order is decided by the key alone.
        let mut keys: Vec<LiteralValue> = items
            .iter()
            .map(|item| match item {
                LiteralValue::Sequence(pair) => pair[0].clone(),
                _ => LiteralValue::None,
            })
            .collect();
        sort_literals(&mut keys)?;
        if keys.windows(2).any(|w| w[0] == w[1]) {
            return None;
        }
        let mut sorted = Vec::with_capacity(items.len());
        for key in &keys {
            let index = items
                .iter()
                .position(|item| matches!(item, LiteralValue::Sequence(pair) if pair[0] == *key))?;
            sorted.push(items[index].clone());
        }
        items.clone_from_slice(&sorted);
        return Some(());
    }
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
    /// For `import a.b.c` binding `a`: the dotted modules imported through this name.
    submodules: Vec<Vec<String>>,
}

#[derive(Debug)]
struct LoadedModule {
    scope: ConstantScope,
    imports: HashMap<String, ImportBinding>,
    names: ModuleNames,
    /// Modules named by `from ... import *`, in statement order.
    star_imports: Vec<Vec<String>>,
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
            star_imports: collect_star_imports(module, &context),
        }
    }

    fn defines(&self, name: &str) -> bool {
        self.scope.bindings.contains_key(name)
            || self.scope.children.contains_key(name)
            || self.scope.defined_functions.contains(name)
            || self.imports.contains_key(name)
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
    /// Qualified functions that return the values of a `Literal` type or union of them.
    literal_values_functions: Vec<String>,
    /// Qualified functions that return their first argument unchanged, such as validators.
    passthrough_functions: Vec<String>,
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
                    child.custom_construction = class_def
                        .keywords()
                        .iter()
                        .any(|keyword| keyword.arg.as_deref() == Some("metaclass"))
                        || !class_def
                            .decorator_list
                            .iter()
                            .all(|decorator| is_transparent_class_decorator(&decorator.expression))
                        || CUSTOM_CONSTRUCTION_MEMBERS.iter().any(|member| {
                            child.defined_functions.contains(*member)
                                || child.bindings.contains_key(*member)
                        });
                    child.record = record_shape(class_def, &child, imports, names);
                    scope.children.insert(class_def.name.to_string(), child);
                    continue;
                }
                Stmt::FunctionDef(func) => {
                    let body = (!scope.defined_functions.contains(func.name.as_str())
                        && !scope.bindings.contains_key(func.name.as_str()))
                    .then(|| returned_expression(func))
                    .flatten();
                    scope.return_values.insert(func.name.to_string(), body);
                    scope.defined_functions.insert(func.name.to_string());
                    // These decorators keep the function's `__name__`, which pytest uses as its ID.
                    if func.decorator_list.iter().all(|decorator| {
                        matches!(&decorator.expression, Expr::Name(name)
                            if matches!(name.id.as_str(), "staticmethod" | "classmethod"))
                    }) {
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
                let alias = ((matches!(
                    value_expr,
                    Expr::Subscript(_) | Expr::Name(_) | Expr::Attribute(_)
                ) || matches!(
                    value_expr,
                    Expr::BinOp(binop) if binop.op == ruff_python_ast::Operator::BitOr
                )) && !scope.bindings.contains_key(name.id.as_str()))
                .then(|| value_expr.clone());
                scope.aliases.insert(name.id.to_string(), alias);
                record_assignment(&mut scope, name.id.as_str(), value_expr, names);
            }
        }

        scope
    }

    fn class_is_enum(&self, class_name: &str) -> bool {
        let Some(class) = self.children.get(class_name) else {
            return false;
        };
        class.bases.iter().any(|base| is_enum_class_name(base))
            || class.bases.iter().any(|base| self.class_is_enum(base))
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

fn expr_segments(expr: &Expr) -> Option<Vec<String>> {
    Some(
        UnqualifiedName::from_expr(expr)?
            .segments()
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
    )
}

/// Walk nested class scopes: `["Outer", "Inner"]`.
fn class_scope<'a>(scope: &'a ConstantScope, path: &[String]) -> Option<&'a ConstantScope> {
    path.iter()
        .try_fold(scope, |scope, name| scope.children.get(name))
}

/// A builtin base such as `Exception`, whose methods are not parsed.
fn segments_are_builtin(base: &str) -> bool {
    base == "object" || NAMED_BUILTINS.contains(&base)
}

fn is_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    chars.next().is_some_and(|c| c == '_' || c.is_alphabetic())
        && chars.all(|c| c == '_' || c.is_alphanumeric())
}

/// Scalars and enum members are the values `Literal[...]` can hold here.
fn literal_member(value: LiteralValue) -> Option<LiteralValue> {
    matches!(
        value,
        LiteralValue::String(_)
            | LiteralValue::Int(_)
            | LiteralValue::Bool(_)
            | LiteralValue::None
            | LiteralValue::EnumMember(_)
    )
    .then_some(value)
}

fn is_transparent_class_decorator(decorator: &Expr) -> bool {
    let decorator = match decorator {
        Expr::Call(call) => call.func.as_ref(),
        other => other,
    };
    UnqualifiedName::from_expr(decorator)
        .and_then(|name| name.segments().last().copied())
        .is_some_and(|last| TRANSPARENT_CLASS_DECORATORS.contains(&last))
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
        let resolver = Self {
            modules: RefCell::new(HashMap::new()),
            current_path,
            loader,
            literal_values_functions: Vec::new(),
            passthrough_functions: Vec::new(),
        };
        let loaded = resolver.expand_star_imports(
            LoadedModule::from_module(module, &resolver.current_path, is_package),
            0,
        );
        resolver
            .modules
            .borrow_mut()
            .insert(resolver.current_path.clone(), Rc::new(loaded));
        resolver
    }

    /// Treat calls to these qualified functions like `typing.get_args`, flattening unions.
    pub fn with_literal_values_functions(mut self, functions: &[String]) -> Self {
        self.literal_values_functions = functions.to_vec();
        self
    }

    /// Treat calls to these qualified functions as returning their first argument.
    pub fn with_passthrough_functions(mut self, functions: &[String]) -> Self {
        self.passthrough_functions = functions.to_vec();
        self
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
        let loaded = self.expand_star_imports(loaded, depth);
        self.modules
            .borrow_mut()
            .insert(module_path.to_vec(), Rc::new(loaded));
    }

    /// Bind each name a `from M import *` brings in as if imported explicitly. Names come
    /// from a statically known `__all__`, or are M's public names when it has none; a star
    /// import whose names cannot be listed adds nothing, so lookups through it fall back.
    fn expand_star_imports(&self, mut loaded: LoadedModule, depth: usize) -> LoadedModule {
        for star in std::mem::take(&mut loaded.star_imports) {
            self.ensure_loaded(&star, depth + 1);
            for name in self.exported_names(&star, depth + 1).unwrap_or_default() {
                if !loaded.defines(&name) {
                    loaded.imports.insert(
                        name.clone(),
                        ImportBinding {
                            module_path: star.clone(),
                            imported_name: Some(name),
                            submodules: Vec::new(),
                        },
                    );
                }
            }
        }
        loaded
    }

    fn exported_names(&self, module_path: &[String], depth: usize) -> Option<Vec<String>> {
        let module = Rc::clone(self.modules.borrow().get(module_path)?);
        if module.defines("__all__") {
            let ResolvedValue::Literal(LiteralValue::Sequence(items)) =
                self.lookup_path_at(module_path, None, &["__all__"], depth)?
            else {
                return None;
            };
            return items
                .into_iter()
                .map(|item| match item {
                    LiteralValue::String(name) => Some(name),
                    _ => None,
                })
                .collect();
        }
        Some(
            module
                .scope
                .bindings
                .keys()
                .chain(module.scope.children.keys())
                .chain(&module.scope.defined_functions)
                .chain(module.imports.keys())
                .filter(|name| !name.starts_with('_'))
                .cloned()
                .collect(),
        )
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
                return self
                    .binding_value(module_path, binding, depth)
                    .map(ResolvedValue::Literal);
            }
            if module.scope.children.contains_key(first) || module.scope.functions.contains(first) {
                return Some(ResolvedValue::Literal(LiteralValue::Named(first.into())));
            }
        } else if module.scope.children.contains_key(first) {
            return self.lookup_class_path(module_path, first, rest, depth);
        } else if let (Some(binding), [attribute]) = (module.scope.bindings.get(first), rest) {
            if let Some(record @ LiteralValue::Record(_)) =
                self.binding_value(module_path, binding, depth)
            {
                return attribute_value(&record, attribute).map(ResolvedValue::Literal);
            }
        }

        let Some(import) = module.imports.get(first) else {
            let builtin = rest.is_empty() && NAMED_BUILTINS.contains(&first);
            return builtin.then(|| ResolvedValue::Literal(LiteralValue::Named(first.into())));
        };
        if import.imported_name.is_none() && rest.is_empty() {
            return None;
        }
        // `import a.b.c` then `a.b.c.X`: the longest imported submodule prefixing the path.
        if let Some(submodule) = import
            .submodules
            .iter()
            .filter(|submodule| {
                submodule.len() < path.len()
                    && submodule
                        .iter()
                        .zip(path)
                        .all(|(part, segment)| part == segment)
            })
            .max_by_key(|submodule| submodule.len())
        {
            self.ensure_loaded(submodule, depth + 1);
            return self.lookup_path_at(submodule, None, &path[submodule.len()..], depth + 1);
        }
        let abs = &import.module_path;
        self.ensure_loaded(abs, depth + 1);
        // `from pkg import sub` names the submodule when the package does not define `sub`.
        if let (Some(name), false) = (&import.imported_name, rest.is_empty()) {
            let defined = self
                .modules
                .borrow()
                .get(abs)
                .is_some_and(|package| package.defines(name));
            if !defined {
                let submodule: Vec<String> =
                    abs.iter().chain(std::iter::once(name)).cloned().collect();
                self.ensure_loaded(&submodule, depth + 1);
                if self.modules.borrow().contains_key(&submodule) {
                    return self.lookup_path_at(&submodule, None, rest, depth + 1);
                }
            }
        }
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
                if (is_class_like_name(name)
                    || (is_skipped_module(abs) && STDLIB_OBJECT_TYPES.contains(name)))
                    && !self.modules.borrow().contains_key(abs) =>
            {
                Some(ResolvedValue::Literal(LiteralValue::Named((*name).into())))
            }
            (None, [class, member]) if !self.modules.borrow().contains_key(abs) => {
                let qualified = format!("{}.{class}.{member}", abs.join("."));
                INT_ENUM_MEMBERS
                    .iter()
                    .find(|(name, _)| *name == qualified)
                    .map(|(_, value)| ResolvedValue::Literal(LiteralValue::Int(i128::from(*value))))
            }
            (found, _) => found,
        }
    }

    fn binding_value(
        &self,
        module_path: &[String],
        binding: &ConstantBinding,
        depth: usize,
    ) -> Option<LiteralValue> {
        let pending = match &*binding.value.borrow() {
            BindingValue::Known(value) => return Some(value.clone()),
            BindingValue::Pending(pending) => Rc::clone(pending),
            BindingValue::Unknown => return None,
        };
        let value = self.resolve_pending(module_path, None, &pending, depth + 1)?;
        *binding.value.borrow_mut() = BindingValue::Known(value.clone());
        Some(value)
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

    /// Follow a dotted callee through module scopes and imports to a class or an external name.
    fn find_callee(
        &self,
        module_path: &[String],
        enclosing_class: Option<&str>,
        path: &[String],
        depth: usize,
    ) -> Option<Callee> {
        if depth > MAX_IMPORT_DEPTH {
            return None;
        }
        let module = Rc::clone(self.modules.borrow().get(module_path)?);
        let (first, rest) = path.split_first()?;
        if let Some(class) = enclosing_class.and_then(|name| module.scope.children.get(name)) {
            if class.children.contains_key(first) {
                let nested: Vec<String> = std::iter::once(enclosing_class?.to_string())
                    .chain(path.iter().cloned())
                    .collect();
                return self.find_callee(module_path, None, &nested, depth);
            }
            if class.bindings.contains_key(first) || class.defined_functions.contains(first) {
                return None;
            }
        }
        if module.scope.children.contains_key(first) {
            class_scope(&module.scope, path)?;
            return Some(Callee::Class {
                module: module_path.to_vec(),
                path: path.to_vec(),
            });
        }
        if module.scope.bindings.contains_key(first)
            || module.scope.defined_functions.contains(first)
        {
            return None;
        }
        let import = module.imports.get(first)?;
        let target: Vec<String> = import.imported_name.iter().chain(rest).cloned().collect();
        if target.is_empty() {
            return None;
        }
        if !is_skipped_module(&import.module_path) {
            self.ensure_loaded(&import.module_path, depth + 1);
        }
        if !self.modules.borrow().contains_key(&import.module_path) {
            let qualified: Vec<&str> = import
                .module_path
                .iter()
                .chain(&target)
                .map(String::as_str)
                .collect();
            return Some(Callee::External(qualified.join(".")));
        }
        self.find_callee(&import.module_path, None, &target, depth + 1)
    }

    /// Whether a call produces an object that pytest IDs positionally.
    fn call_returns_object(
        &self,
        module_path: &[String],
        enclosing_class: Option<&str>,
        callee: &[String],
        depth: usize,
    ) -> bool {
        match self.find_callee(module_path, enclosing_class, callee, depth) {
            Some(Callee::Class { module, path }) => {
                return self.instance_is_object(&module, &path, true, depth)
            }
            Some(Callee::External(name)) => {
                return OBJECT_RETURNING_CALLABLES.contains(&name.as_str())
            }
            None => {}
        }
        if let Some((method, owner)) = callee.split_last().filter(|(_, owner)| !owner.is_empty()) {
            if let Some(Callee::Class { module, path }) =
                self.find_callee(module_path, enclosing_class, owner, depth)
            {
                return self
                    .method_return(&module, &path, method, depth)
                    .is_some_and(|(def_module, annotation)| {
                        self.return_is_object(
                            &def_module,
                            &annotation,
                            Some((&module, &path)),
                            depth,
                        )
                    });
            }
        }
        self.function_return(module_path, enclosing_class, callee, depth)
            .is_some_and(|(def_module, annotation)| {
                self.return_is_object(&def_module, &annotation, None, depth)
            })
    }

    /// Whether instances of a parsed class are plain objects rather than values pytest IDs
    /// by value or `__name__`. `constructed` also rejects classes that customize construction.
    fn instance_is_object(
        &self,
        module_path: &[String],
        class_path: &[String],
        constructed: bool,
        depth: usize,
    ) -> bool {
        if depth > MAX_IMPORT_DEPTH {
            return false;
        }
        let Some(module) = self.modules.borrow().get(module_path).map(Rc::clone) else {
            return false;
        };
        let Some(class) = class_scope(&module.scope, class_path) else {
            return false;
        };
        if constructed && class.custom_construction {
            return false;
        }
        class.bases.iter().all(|base| {
            let last = base.rsplit('.').next().unwrap_or(base);
            if is_enum_class_name(base)
                || is_enum_class_name(last)
                || VALUE_TYPE_NAMES.contains(&last)
            {
                return false;
            }
            let segments: Vec<String> = base.split('.').map(String::from).collect();
            match self.find_callee(module_path, None, &segments, depth + 1) {
                Some(Callee::Class { module, path }) => {
                    self.instance_is_object(&module, &path, constructed, depth + 1)
                }
                // Unparsed bases (stdlib, pydantic, ORM models, ...) are trusted unless their
                // names mark them as enums or value types, matching return-annotation checks.
                Some(Callee::External(_)) => true,
                None => segments_are_builtin(base),
            }
        })
    }

    /// Find a method's return annotation on a class or the one base that defines it.
    fn method_return(
        &self,
        module_path: &[String],
        class_path: &[String],
        method: &str,
        depth: usize,
    ) -> Option<(Vec<String>, Expr)> {
        if depth > MAX_IMPORT_DEPTH {
            return None;
        }
        let module = Rc::clone(self.modules.borrow().get(module_path)?);
        let class = class_scope(&module.scope, class_path)?;
        if let Some(returns) = class.returns.get(method) {
            return Some((module_path.to_vec(), returns.clone()));
        }
        if class.defined_functions.contains(method)
            || class.bindings.contains_key(method)
            || class.children.contains_key(method)
        {
            return None;
        }
        // Without computing the MRO, accept a method only when exactly one base defines it.
        let mut found = None;
        for base in &class.bases {
            if segments_are_builtin(base) {
                continue;
            }
            let segments: Vec<String> = base.split('.').map(String::from).collect();
            let Some(Callee::Class { module, path }) =
                self.find_callee(module_path, None, &segments, depth + 1)
            else {
                return None;
            };
            if let Some(returns) = self.method_return(&module, &path, method, depth + 1) {
                if found.is_some() {
                    return None;
                }
                found = Some(returns);
            }
        }
        found
    }

    /// Whether a return annotation names an object type. `Self` refers to `owner`.
    fn return_is_object(
        &self,
        def_module: &[String],
        annotation: &Expr,
        owner: Option<(&[String], &[String])>,
        depth: usize,
    ) -> bool {
        let segments: Vec<String> = match annotation {
            Expr::StringLiteral(text) => {
                let text = text.value.to_str();
                if !text.split('.').all(is_identifier) {
                    return false;
                }
                text.split('.').map(String::from).collect()
            }
            Expr::Name(_) | Expr::Attribute(_) => match UnqualifiedName::from_expr(annotation) {
                Some(name) => name.segments().iter().map(|s| (*s).to_string()).collect(),
                None => return false,
            },
            _ => return self.annotation_is_object(def_module, annotation, depth),
        };
        if segments.last().map(String::as_str) == Some("Self") {
            return owner.is_some_and(|(module, class)| {
                self.instance_is_object(module, class, false, depth)
            });
        }
        if let Some(Callee::Class { module, path }) =
            self.find_callee(def_module, None, &segments, depth)
        {
            return self.instance_is_object(&module, &path, false, depth);
        }
        match annotation {
            Expr::StringLiteral(_) => self.is_object_class(def_module, &segments, depth),
            _ => self.annotation_is_object(def_module, annotation, depth),
        }
    }

    /// The values `typing.get_args` returns for a `Literal[...]` type, flattened and
    /// de-duplicated like `typing.Literal` does.
    fn literal_type_values(
        &self,
        module_path: &[String],
        expr: &Expr,
        depth: usize,
    ) -> Option<Vec<LiteralValue>> {
        if depth > MAX_IMPORT_DEPTH {
            return None;
        }
        let subscript = match expr {
            Expr::Subscript(subscript) => subscript,
            Expr::Name(_) | Expr::Attribute(_) => {
                let (alias_module, alias) =
                    self.find_alias(module_path, &expr_segments(expr)?, depth)?;
                return self.literal_type_values(&alias_module, &alias, depth + 1);
            }
            _ => return None,
        };
        let base: Vec<String> = UnqualifiedName::from_expr(&subscript.value)?
            .segments()
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        match self.find_callee(module_path, None, &base, depth)? {
            Callee::External(name) if LITERAL_TYPES.contains(&name.as_str()) => {}
            _ => return None,
        }
        let elements = match subscript.slice.as_ref() {
            Expr::Tuple(tuple) => tuple.elts.as_slice(),
            single => std::slice::from_ref(single),
        };
        let mut values: Vec<LiteralValue> = Vec::new();
        for element in elements {
            let resolved = match element {
                Expr::Subscript(_) => self.literal_type_values(module_path, element, depth + 1)?,
                Expr::Name(_) | Expr::Attribute(_) => self
                    .find_alias(module_path, &expr_segments(element)?, depth)
                    .and_then(|(alias_module, alias)| {
                        self.literal_type_values(&alias_module, &alias, depth + 1)
                    })
                    .or_else(|| {
                        Some(vec![self.resolve_literal_member(
                            module_path,
                            element,
                            depth,
                        )?])
                    })?,
                Expr::StringLiteral(_)
                | Expr::NumberLiteral(_)
                | Expr::UnaryOp(_)
                | Expr::BooleanLiteral(_)
                | Expr::NoneLiteral(_) => vec![literal_member(try_extract_literal(element)?)?],
                _ => return None,
            };
            for value in resolved {
                if !values.contains(&value) {
                    values.push(value);
                }
            }
        }
        Some(values)
    }

    /// Bind call arguments to a record class's constructor parameters.
    fn construct_record(
        &self,
        module_path: &[String],
        enclosing_class: Option<&str>,
        callee: &[String],
        args: &[Option<PendingValue>],
        keywords: &[(String, Option<PendingValue>)],
        depth: usize,
    ) -> Option<LiteralValue> {
        let Callee::Class { module, path } =
            self.find_callee(module_path, enclosing_class, callee, depth)?
        else {
            return None;
        };
        let params = self.record_params(&module, &path, depth)?;
        let argument = |value: &Option<PendingValue>| {
            value
                .as_ref()
                .and_then(|value| self.resolve_pending(module_path, enclosing_class, value, depth))
        };
        let mut supplied: Vec<Option<Option<LiteralValue>>> = vec![None; params.len()];
        let positional = (0..params.len()).filter(|&index| !params[index].1.keyword_only);
        let mut positional = positional.peekable();
        for arg in args {
            supplied[positional.next()?] = Some(argument(arg));
        }
        for (name, value) in keywords {
            let index = params.iter().position(|(_, param)| param.name == *name)?;
            if supplied[index].replace(argument(value)).is_some() {
                return None;
            }
        }
        let mut fields = Vec::new();
        for ((param_module, param), value) in params.iter().zip(supplied) {
            let value = match (value, &param.default) {
                (Some(value), _) => value,
                (None, None) => return None,
                (None, Some(default)) => default.as_ref().and_then(|default| {
                    self.resolve_pending(param_module, None, default, depth + 1)
                }),
            };
            fields.extend(
                param
                    .attributes
                    .iter()
                    .map(|attribute| (attribute.clone(), value.clone())),
            );
        }
        Some(LiteralValue::Record(fields))
    }

    /// The `__name__` of a subscripted typing generic. `Union` collapses duplicate members
    /// and is named `Optional` for exactly one type plus `None`, so its members must be
    /// distinct named types or `None`.
    fn typing_alias_name(
        &self,
        module_path: &[String],
        enclosing_class: Option<&str>,
        typing_name: &str,
        args: &Expr,
        depth: usize,
    ) -> Option<String> {
        if TYPING_GENERICS.contains(&typing_name) {
            return Some(typing_name.to_string());
        }
        let mut members = Vec::new();
        match (typing_name, args) {
            ("Optional", Expr::Tuple(_)) => return None,
            ("Optional", member) => {
                self.collect_type_members(module_path, member, depth, &mut members)?;
                members.push(TypeMember::None);
            }
            ("Union", Expr::Tuple(tuple)) => {
                for member in &tuple.elts {
                    self.collect_type_members(module_path, member, depth, &mut members)?;
                }
            }
            ("Union", member) => {
                self.collect_type_members(module_path, member, depth, &mut members)?;
            }
            _ => return None,
        }
        let _ = enclosing_class;
        match union_value(members, true)? {
            LiteralValue::Named(name) => Some(name),
            _ => None,
        }
    }

    /// Flatten a union's members, following aliases. Returns whether any member is a typing
    /// form, which makes `a | b` a `typing.Union` rather than a `types.UnionType`.
    fn collect_type_members(
        &self,
        module_path: &[String],
        expr: &Expr,
        depth: usize,
        members: &mut Vec<TypeMember>,
    ) -> Option<bool> {
        if depth > MAX_IMPORT_DEPTH {
            return None;
        }
        match expr {
            Expr::NoneLiteral(_) => {
                members.push(TypeMember::None);
                Some(false)
            }
            Expr::BinOp(binop) if binop.op == ruff_python_ast::Operator::BitOr => {
                let left = self.collect_type_members(module_path, &binop.left, depth, members)?;
                let right = self.collect_type_members(module_path, &binop.right, depth, members)?;
                Some(left || right)
            }
            Expr::Subscript(subscript) => {
                let base = expr_segments(&subscript.value)?;
                if base.len() == 1
                    && GENERIC_BUILTINS.contains(&base[0].as_str())
                    && self.find_alias(module_path, &base, depth).is_none()
                    && !self
                        .modules
                        .borrow()
                        .get(module_path)?
                        .scope
                        .bindings
                        .contains_key(&base[0])
                {
                    members.push(TypeMember::Generic {
                        name: base[0].clone(),
                        key: generic_key(module_path, expr),
                    });
                    return Some(false);
                }
                let Callee::External(name) = self.find_callee(module_path, None, &base, depth)?
                else {
                    return None;
                };
                if LITERAL_TYPES.contains(&name.as_str()) {
                    members.push(TypeMember::Literal(self.literal_type_values(
                        module_path,
                        expr,
                        depth,
                    )?));
                    return Some(true);
                }
                let typing_name = name
                    .strip_prefix("typing.")
                    .or_else(|| name.strip_prefix("typing_extensions."))?;
                match (typing_name, subscript.slice.as_ref()) {
                    ("Optional", Expr::Tuple(_)) => return None,
                    ("Optional", member) => {
                        self.collect_type_members(module_path, member, depth, members)?;
                        members.push(TypeMember::None);
                    }
                    ("Union", Expr::Tuple(tuple)) => {
                        for member in &tuple.elts {
                            self.collect_type_members(module_path, member, depth, members)?;
                        }
                    }
                    ("Union", member) => {
                        self.collect_type_members(module_path, member, depth, members)?;
                    }
                    (generic, _) if TYPING_GENERICS.contains(&generic) => {
                        members.push(TypeMember::Generic {
                            name: generic.to_string(),
                            key: generic_key(module_path, expr),
                        });
                    }
                    _ => return None,
                }
                Some(true)
            }
            Expr::Name(_) | Expr::Attribute(_) => {
                let path = expr_segments(expr)?;
                if let Some((alias_module, alias)) = self.find_alias(module_path, &path, depth) {
                    if matches!(
                        alias,
                        Expr::Name(_) | Expr::Attribute(_) | Expr::Subscript(_) | Expr::BinOp(_)
                    ) {
                        return self.collect_type_members(
                            &alias_module,
                            &alias,
                            depth + 1,
                            members,
                        );
                    }
                    return None;
                }
                let pending = {
                    let modules = self.modules.borrow();
                    pending_value(expr, &modules.get(module_path)?.names, &LocalValues::new())?
                };
                match self.resolve_pending(module_path, None, &pending, depth)? {
                    LiteralValue::Named(name) => {
                        let key = match self.find_callee(module_path, None, &path, depth) {
                            Some(Callee::Class { module, path }) => {
                                format!("{}:{}", module.join("."), path.join("."))
                            }
                            Some(Callee::External(qualified)) => qualified,
                            None => format!("{}:{}", module_path.join("."), path.join(".")),
                        };
                        members.push(TypeMember::Class { name, key });
                        Some(false)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Constructor parameters of a record class, with the module each was declared in.
    /// A dataclass extends its base's fields; a redefined field keeps the base's position.
    fn record_params(
        &self,
        module: &[String],
        path: &[String],
        depth: usize,
    ) -> Option<Vec<(Vec<String>, RecordParam)>> {
        if depth > MAX_IMPORT_DEPTH {
            return None;
        }
        let class_module = Rc::clone(self.modules.borrow().get(module)?);
        let shape = class_scope(&class_module.scope, path)?.record.as_ref()?;
        let mut params = match &shape.base {
            None => Vec::new(),
            Some(base) => {
                let enclosing = match path {
                    [.., outer, _] => Some(outer.as_str()),
                    _ => None,
                };
                let Callee::Class {
                    module: base_module,
                    path: base_path,
                } = self.find_callee(module, enclosing, base, depth + 1)?
                else {
                    return None;
                };
                let base_class = Rc::clone(self.modules.borrow().get(&base_module)?);
                if !class_scope(&base_class.scope, &base_path)?
                    .record
                    .as_ref()?
                    .dataclass
                {
                    return None;
                }
                self.record_params(&base_module, &base_path, depth + 1)?
            }
        };
        for param in &shape.params {
            let declared = (module.to_vec(), param.clone());
            match params
                .iter_mut()
                .find(|(_, existing)| existing.name == param.name)
            {
                Some(existing) => *existing = declared,
                None => params.push(declared),
            }
        }
        Some(params)
    }

    /// The value of a parameterless call to a function that only returns an expression,
    /// evaluated in the function's own module.
    fn call_return_value(
        &self,
        module_path: &[String],
        enclosing_class: Option<&str>,
        callee: &[String],
        depth: usize,
    ) -> Option<LiteralValue> {
        if depth > MAX_IMPORT_DEPTH {
            return None;
        }
        let (module, scope_path, name) = match callee.split_last()? {
            (method, owner) if !owner.is_empty() => {
                match self.find_callee(module_path, enclosing_class, owner, depth)? {
                    Callee::Class { module, path } => (module, path, method.clone()),
                    Callee::External(_) => return None,
                }
            }
            (name, _) => self.function_definition(module_path, enclosing_class, name, depth)?,
        };
        let pending = {
            let loaded = Rc::clone(self.modules.borrow().get(&module)?);
            let scope = class_scope(&loaded.scope, &scope_path)?;
            let body = scope.return_values.get(&name)?.as_ref()?;
            let locals: LocalValues = body
                .class_parameter
                .iter()
                .map(|parameter| (parameter.clone(), LiteralValue::Opaque))
                .collect();
            pending_value(&body.value, &loaded.names, &locals)?
        };
        self.resolve_pending(&module, None, &pending, depth + 1)
    }

    /// Where a called name is defined: its module, enclosing class path and name.
    fn function_definition(
        &self,
        module_path: &[String],
        enclosing_class: Option<&str>,
        name: &str,
        depth: usize,
    ) -> Option<(Vec<String>, Vec<String>, String)> {
        if depth > MAX_IMPORT_DEPTH {
            return None;
        }
        let module = Rc::clone(self.modules.borrow().get(module_path)?);
        if let Some(class_name) = enclosing_class {
            let class = module.scope.children.get(class_name)?;
            if class.defined_functions.contains(name) {
                return Some((
                    module_path.to_vec(),
                    vec![class_name.to_string()],
                    name.to_string(),
                ));
            }
            if class.bindings.contains_key(name) || class.children.contains_key(name) {
                return None;
            }
        }
        if module.scope.defined_functions.contains(name) {
            return Some((module_path.to_vec(), Vec::new(), name.to_string()));
        }
        if module.scope.bindings.contains_key(name) || module.scope.children.contains_key(name) {
            return None;
        }
        let import = module.imports.get(name)?;
        let imported = import.imported_name.clone()?;
        self.ensure_loaded(&import.module_path, depth + 1);
        if !self.modules.borrow().contains_key(&import.module_path) {
            return None;
        }
        self.function_definition(&import.module_path, None, &imported, depth + 1)
    }

    /// The qualified name a configured function list would use for `callee`.
    fn qualified_callee(
        &self,
        module_path: &[String],
        enclosing_class: Option<&str>,
        callee: &[String],
        depth: usize,
    ) -> Option<String> {
        Some(match callee.split_last() {
            Some((method, owner)) if !owner.is_empty() => {
                match self.find_callee(module_path, enclosing_class, owner, depth)? {
                    Callee::Class { module, path } => module
                        .iter()
                        .chain(&path)
                        .chain(std::iter::once(method))
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("."),
                    Callee::External(name) => format!("{name}.{method}"),
                }
            }
            _ => match self.find_callee(module_path, enclosing_class, callee, depth) {
                Some(Callee::External(name)) => name,
                _ => self.function_module(module_path, callee, depth)?,
            },
        })
    }

    /// The qualified name of a module-level function, following `from x import f`.
    fn function_module(
        &self,
        module_path: &[String],
        path: &[String],
        depth: usize,
    ) -> Option<String> {
        if depth > MAX_IMPORT_DEPTH {
            return None;
        }
        let [name] = path else {
            return None;
        };
        let module = Rc::clone(self.modules.borrow().get(module_path)?);
        if module.scope.defined_functions.contains(name) {
            return Some(
                module_path
                    .iter()
                    .chain(std::iter::once(name))
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("."),
            );
        }
        if module.scope.bindings.contains_key(name) {
            return None;
        }
        let import = module.imports.get(name)?;
        let imported = import.imported_name.clone()?;
        self.ensure_loaded(&import.module_path, depth + 1);
        if !self.modules.borrow().contains_key(&import.module_path) {
            return Some(
                import
                    .module_path
                    .iter()
                    .chain(std::iter::once(&imported))
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("."),
            );
        }
        self.function_module(&import.module_path, &[imported], depth + 1)
    }

    /// String values of a `Literal` type or a union of them, as literal-values
    /// helpers return them: unions flatten left to right, union members equal as types are
    /// dropped like `typing.Union` does, and each `Literal` keeps its own de-duplication.
    fn union_literal_values(
        &self,
        module_path: &[String],
        expr: &Expr,
        depth: usize,
    ) -> Option<Vec<LiteralValue>> {
        let mut members: Vec<Vec<LiteralValue>> = Vec::new();
        self.collect_union_members(module_path, expr, depth, &mut members)?;
        let mut values = Vec::new();
        let mut seen: Vec<Vec<LiteralValue>> = Vec::new();
        for member in members {
            if !member
                .iter()
                .all(|value| matches!(value, LiteralValue::String(_)))
            {
                return None;
            }
            // `Literal` types compare equal when they hold the same set of values.
            let same_type = |other: &Vec<LiteralValue>| {
                other.len() == member.len() && other.iter().all(|value| member.contains(value))
            };
            if seen.iter().any(same_type) {
                continue;
            }
            values.extend(member.iter().cloned());
            seen.push(member);
        }
        Some(values)
    }

    fn collect_union_members(
        &self,
        module_path: &[String],
        expr: &Expr,
        depth: usize,
        members: &mut Vec<Vec<LiteralValue>>,
    ) -> Option<()> {
        if depth > MAX_IMPORT_DEPTH {
            return None;
        }
        match expr {
            Expr::BinOp(binop) if binop.op == ruff_python_ast::Operator::BitOr => {
                self.collect_union_members(module_path, &binop.left, depth, members)?;
                self.collect_union_members(module_path, &binop.right, depth, members)
            }
            Expr::Subscript(subscript) => {
                let base = expr_segments(&subscript.value)?;
                match self.find_callee(module_path, None, &base, depth)? {
                    Callee::External(name) if LITERAL_TYPES.contains(&name.as_str()) => {
                        members.push(self.literal_type_values(module_path, expr, depth)?);
                        Some(())
                    }
                    Callee::External(name) if UNION_TYPES.contains(&name.as_str()) => {
                        let elements = match subscript.slice.as_ref() {
                            Expr::Tuple(tuple) => tuple.elts.as_slice(),
                            single => std::slice::from_ref(single),
                        };
                        for element in elements {
                            self.collect_union_members(module_path, element, depth, members)?;
                        }
                        Some(())
                    }
                    _ => None,
                }
            }
            Expr::Name(_) | Expr::Attribute(_) => {
                let (alias_module, alias) =
                    self.find_alias(module_path, &expr_segments(expr)?, depth)?;
                self.collect_union_members(&alias_module, &alias, depth + 1, members)
            }
            _ => None,
        }
    }

    /// A name used as a `Literal[...]` member: an enum member or a scalar constant.
    fn resolve_literal_member(
        &self,
        module_path: &[String],
        expr: &Expr,
        depth: usize,
    ) -> Option<LiteralValue> {
        let name = UnqualifiedName::from_expr(expr)?;
        let path: Vec<String> = name.segments().iter().map(|s| (*s).to_string()).collect();
        let value = self.lookup_path_at(module_path, None, name.segments(), depth)?;
        literal_member(value.into_literal(&path))
    }

    /// The assigned expression behind a possibly imported alias name.
    fn find_alias(
        &self,
        module_path: &[String],
        path: &[String],
        depth: usize,
    ) -> Option<(Vec<String>, Expr)> {
        if depth > MAX_IMPORT_DEPTH {
            return None;
        }
        let module = Rc::clone(self.modules.borrow().get(module_path)?);
        let (import, imported) = match path {
            [single] => {
                if let Some(alias) = module.scope.aliases.get(single) {
                    return alias.clone().map(|alias| (module_path.to_vec(), alias));
                }
                if module.scope.bindings.contains_key(single) {
                    return None;
                }
                let import = module.imports.get(single)?;
                (import, import.imported_name.clone()?)
            }
            [module_name, member] => match module.imports.get(module_name) {
                Some(import) if import.imported_name.is_none() => (import, member.clone()),
                _ => return None,
            },
            _ => return None,
        };
        self.ensure_loaded(&import.module_path, depth + 1);
        self.find_alias(&import.module_path, &[imported], depth + 1)
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
            PendingValue::Sequence(items) => {
                let mut values = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        PendingValue::Unpack(inner) => match resolve(inner)? {
                            LiteralValue::Sequence(inner) => values.extend(inner),
                            _ => return None,
                        },
                        item => values.push(resolve(item)?),
                    }
                }
                Some(LiteralValue::Sequence(values))
            }
            PendingValue::Unpack(_) => None,
            PendingValue::Construct {
                callee,
                args,
                keywords,
                otherwise,
            } => self
                .construct_record(module_path, enclosing_class, callee, args, keywords, depth)
                .or_else(|| {
                    if !args.is_empty() || !keywords.is_empty() {
                        return None;
                    }
                    self.call_return_value(module_path, enclosing_class, callee, depth)
                })
                .or_else(|| resolve(otherwise.as_deref()?)),
            PendingValue::Comprehension {
                elt,
                generators,
                locals,
            } => {
                let evaluate = |expr: &Expr, scope: &LocalValues| {
                    let pending = {
                        let modules = self.modules.borrow();
                        pending_value(expr, &modules.get(module_path)?.names, scope)?
                    };
                    self.resolve_pending(module_path, enclosing_class, &pending, depth)
                };
                let mut values = Vec::new();
                super::cases::eval_generators(elt, generators, &evaluate, locals, &mut values)?;
                Some(LiteralValue::Sequence(values))
            }
            PendingValue::Generic {
                base,
                builtin,
                args,
                index,
            } => {
                if let Some(builtin) = builtin {
                    return Some(LiteralValue::Named((*builtin).into()));
                }
                if let Some(Callee::External(name)) =
                    self.find_callee(module_path, enclosing_class, base, depth)
                {
                    let typing_name = name
                        .strip_prefix("typing.")
                        .or_else(|| name.strip_prefix("typing_extensions."));
                    if let Some(typing_name) = typing_name {
                        return self
                            .typing_alias_name(
                                module_path,
                                enclosing_class,
                                typing_name,
                                args,
                                depth,
                            )
                            .map(LiteralValue::Named);
                    }
                }
                resolve(index.as_deref()?)
            }
            PendingValue::Enumerate { source, start } => {
                let items = match resolve(source)? {
                    LiteralValue::Sequence(items) => items,
                    LiteralValue::Mapping(entries) => {
                        entries.into_iter().map(|(key, _)| key).collect()
                    }
                    _ => return None,
                };
                let mut values = Vec::with_capacity(items.len());
                for (offset, item) in items.into_iter().enumerate() {
                    let index = start.checked_add(i128::try_from(offset).ok()?)?;
                    values.push(LiteralValue::Sequence(vec![LiteralValue::Int(index), item]));
                }
                Some(LiteralValue::Sequence(values))
            }
            PendingValue::Zip { sources, strict } => {
                let mut columns = Vec::with_capacity(sources.len());
                for source in sources {
                    columns.push(match resolve(source)? {
                        LiteralValue::Sequence(items) => items,
                        LiteralValue::Mapping(entries) => {
                            entries.into_iter().map(|(key, _)| key).collect()
                        }
                        _ => return None,
                    });
                }
                let len = columns.iter().map(Vec::len).min().unwrap_or(0);
                if *strict && columns.iter().any(|column| column.len() != len) {
                    return None;
                }
                let mut columns: Vec<_> = columns.into_iter().map(Vec::into_iter).collect();
                Some(LiteralValue::Sequence(
                    (0..len)
                        .map(|_| {
                            LiteralValue::Sequence(
                                columns.iter_mut().filter_map(Iterator::next).collect(),
                            )
                        })
                        .collect(),
                ))
            }
            PendingValue::SetOf(source) => {
                let items = match resolve(source) {
                    Some(LiteralValue::Sequence(items) | LiteralValue::Set(items)) => items,
                    Some(LiteralValue::Mapping(entries)) => {
                        entries.into_iter().map(|(key, _)| key).collect()
                    }
                    _ => return Some(LiteralValue::Opaque),
                };
                Some(set_value(items).unwrap_or(LiteralValue::Opaque))
            }
            PendingValue::Mapping(items) => {
                let mut entries: Vec<(LiteralValue, Option<LiteralValue>)> = Vec::new();
                for (key, value) in items {
                    if let PendingValue::Unpack(mapping) = key {
                        let LiteralValue::Mapping(unpacked) = resolve(mapping)? else {
                            return None;
                        };
                        for (key, value) in unpacked {
                            match entries.iter_mut().find(|(existing, _)| *existing == key) {
                                Some(entry) => entry.1 = value,
                                None => entries.push((key, value)),
                            }
                        }
                        continue;
                    }
                    let key = resolve(key)?;
                    if matches!(
                        key,
                        LiteralValue::Opaque | LiteralValue::Mapping(_) | LiteralValue::Record(_)
                    ) {
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
                    (LiteralValue::Set(items), ElementView::Keys) if *sorted => items,
                    (LiteralValue::Mapping(entries), ElementView::Keys) => {
                        entries.into_iter().map(|(key, _)| key).collect()
                    }
                    (LiteralValue::Mapping(entries), ElementView::Values) => entries
                        .into_iter()
                        .map(|(_, value)| value)
                        .collect::<Option<_>>()?,
                    (LiteralValue::Mapping(entries), ElementView::Items) => entries
                        .into_iter()
                        .map(|(key, value)| Some(LiteralValue::Sequence(vec![key, value?])))
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
            PendingValue::TypeOr { union, value } => resolve(value).or_else(|| {
                let mut members = Vec::new();
                let typing = self.collect_type_members(module_path, union, depth, &mut members)?;
                union_value(members, typing)
            }),
            PendingValue::Param { values, id } => Some(LiteralValue::Param {
                values: values.iter().map(resolve).collect::<Option<_>>()?,
                id: id.clone(),
            }),
            PendingValue::CallResult(callee) => self
                .call_returns_object(module_path, enclosing_class, callee, depth)
                .then_some(LiteralValue::Opaque),
            PendingValue::LiteralArgs { callee, argument } => {
                match self.find_callee(module_path, enclosing_class, callee, depth)? {
                    Callee::External(name) if GET_ARGS_CALLABLES.contains(&name.as_str()) => {}
                    _ => return None,
                }
                self.literal_type_values(module_path, argument, depth)
                    .map(LiteralValue::Sequence)
            }
            PendingValue::ConfiguredCall {
                callee,
                argument,
                argument_value,
                single_argument,
                otherwise,
            } => {
                let qualified = (!self.literal_values_functions.is_empty()
                    || !self.passthrough_functions.is_empty())
                .then(|| self.qualified_callee(module_path, enclosing_class, callee, depth))
                .flatten();
                if let Some(qualified) = qualified {
                    if *single_argument && self.literal_values_functions.contains(&qualified) {
                        return self
                            .union_literal_values(module_path, argument, depth)
                            .map(LiteralValue::Sequence);
                    }
                    if self.passthrough_functions.contains(&qualified) {
                        return resolve(argument_value.as_deref()?);
                    }
                }
                resolve(otherwise.as_deref()?)
            }
            PendingValue::Index { source, key } => match (resolve(source)?, resolve(key)?) {
                (LiteralValue::Sequence(items), LiteralValue::Int(index)) => {
                    let index = if index < 0 {
                        items.len().checked_sub(index.unsigned_abs() as usize)?
                    } else {
                        index as usize
                    };
                    items.into_iter().nth(index)
                }
                (LiteralValue::Mapping(entries), key)
                    if matches!(
                        key,
                        LiteralValue::String(_) | LiteralValue::EnumMember(_) | LiteralValue::None
                    ) =>
                {
                    entries
                        .into_iter()
                        .find(|(existing, _)| *existing == key)
                        .and_then(|(_, value)| value)
                }
                _ => None,
            },
            PendingValue::FString(parts) => {
                let mut text = String::new();
                for part in parts {
                    match resolve(part)? {
                        LiteralValue::String(value) => text.push_str(&value),
                        LiteralValue::Int(value) => text.push_str(&value.to_string()),
                        LiteralValue::Bool(value) => {
                            text.push_str(if value { "True" } else { "False" })
                        }
                        LiteralValue::None => text.push_str("None"),
                        _ => return None,
                    }
                }
                Some(LiteralValue::String(text))
            }
            PendingValue::Slice {
                source,
                start,
                stop,
            } => {
                let LiteralValue::Sequence(items) = resolve(source)? else {
                    return None;
                };
                let len = items.len() as i128;
                let clamp = |bound: Option<i128>, default: i128| {
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
        let (member, nested) = rest.split_last()?;
        let class_path: Vec<&str> = std::iter::once(class_name)
            .chain(nested.iter().copied())
            .collect();
        self.lookup_member_at(module_path, &class_path, member, depth)
    }

    fn lookup_class_member(
        &self,
        module_path: &[String],
        class_name: &str,
        member: &str,
        depth: usize,
    ) -> Option<ResolvedValue> {
        self.lookup_member_at(module_path, &[class_name], member, depth)
    }

    /// Look up `member` on the class at `class_path` (outermost first) in a module.
    fn lookup_member_at(
        &self,
        module_path: &[String],
        class_path: &[&str],
        member: &str,
        depth: usize,
    ) -> Option<ResolvedValue> {
        let module = Rc::clone(self.modules.borrow().get(module_path)?);
        let (class_name, outer) = class_path.split_last()?;
        let parent = outer
            .iter()
            .try_fold(&module.scope, |scope, name| scope.children.get(*name))?;
        let class = parent.children.get(*class_name)?;
        if let Some(binding) = class.bindings.get(member) {
            // Modules loaded on demand keep pending class constants until something reads them.
            let pending = match &*binding.value.borrow() {
                BindingValue::Pending(pending) if !binding.reassigned => Some(Rc::clone(pending)),
                _ => None,
            };
            if let Some(pending) = pending {
                let enclosing = outer.is_empty().then_some(*class_name);
                let value = self.resolve_pending(module_path, enclosing, &pending, depth + 1)?;
                *binding.value.borrow_mut() = BindingValue::Known(value);
            }
            let value = binding.known_value()?;
            let is_member = !matches!(value, LiteralValue::Named(_));
            return Some(if is_member && parent.class_is_enum(class_name) {
                ResolvedValue::EnumMember
            } else {
                ResolvedValue::Literal(value)
            });
        }
        if class.children.contains_key(member) || class.functions.contains(member) {
            return Some(ResolvedValue::Literal(LiteralValue::Named(member.into())));
        }
        if class.defined_functions.contains(member) || depth > MAX_IMPORT_DEPTH {
            return None;
        }
        // Without computing the MRO, accept a member only when every base defining it agrees.
        let mut found = None;
        for base in &class.bases {
            let segments: Vec<String> = base.split('.').map(String::from).collect();
            // Base expressions see the enclosing class body, then module globals.
            let enclosing = match outer {
                [] => None,
                [single] => Some(*single),
                _ if parent.children.contains_key(&segments[0])
                    || parent.bindings.contains_key(&segments[0]) =>
                {
                    return None
                }
                _ => None,
            };
            let value = match self.find_callee(module_path, enclosing, &segments, depth + 1)? {
                Callee::Class { module, path } => {
                    let path: Vec<&str> = path.iter().map(String::as_str).collect();
                    self.lookup_member_at(&module, &path, member, depth + 1)
                }
                Callee::External(_) => None,
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

fn collect_star_imports(module: &ModModule, context: &[String]) -> Vec<Vec<String>> {
    module
        .body
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::ImportFrom(import_from)
                if import_from
                    .names
                    .iter()
                    .any(|alias| alias.name.as_str() == "*") =>
            {
                let module_path: Vec<String> = import_from
                    .module
                    .iter()
                    .flat_map(|name| name.split('.').map(String::from))
                    .collect();
                resolve_relative_import(context, import_from.level as usize, &module_path)
            }
            _ => None,
        })
        .collect()
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
                            submodules: Vec::new(),
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
                                submodules: Vec::new(),
                            },
                        );
                    } else if let Some(first) = parts.first() {
                        let binding = imports.entry(first.clone()).or_insert(ImportBinding {
                            module_path: vec![first.clone()],
                            imported_name: None,
                            submodules: Vec::new(),
                        });
                        if binding.imported_name.is_some() || binding.module_path != [first.clone()]
                        {
                            *binding = ImportBinding {
                                module_path: vec![first.clone()],
                                imported_name: None,
                                submodules: Vec::new(),
                            };
                        }
                        if parts.len() > 1 {
                            binding.submodules.push(parts);
                        }
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

/// The value of an integer literal, or `None` past the 128-bit range rtest folds.
fn int_literal_value(int: &ruff_python_ast::Int) -> Option<i128> {
    if let Some(value) = int.as_u64() {
        return Some(i128::from(value));
    }
    // Literals past u64 keep their source token, such as `0x1_0000_0000_0000_0000`.
    let token = int.to_string().replace('_', "");
    let lower = token.to_ascii_lowercase();
    let (digits, radix) = match lower.get(..2) {
        Some("0x") => (&lower[2..], 16),
        Some("0o") => (&lower[2..], 8),
        Some("0b") => (&lower[2..], 2),
        _ => (lower.as_str(), 10),
    };
    i128::from_str_radix(digits, radix).ok()
}

/// Try to extract a literal value from an expression.
///
/// This is intentionally limited to simple literals to avoid complexity.
pub(crate) fn try_extract_literal(expr: &Expr) -> Option<LiteralValue> {
    match expr {
        Expr::NumberLiteral(num) => {
            use ruff_python_ast::Number;
            match &num.value {
                Number::Int(i) => int_literal_value(i).map(LiteralValue::Int),
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

#[cfg(test)]
mod tests {
    use super::*;
    use ruff_python_parser::parse_module;

    /// Resolve the final expression statement of `source` as a parametrize value.
    fn resolve_last(source: &str) -> Option<LiteralValue> {
        let parsed = parse_module(source).expect("valid source");
        let module = parsed.into_syntax();
        let resolver = ConstantResolver::from_module(&module);
        let Some(Stmt::Expr(last)) = module.body.last() else {
            panic!("source must end with an expression");
        };
        resolver.resolve_expr(None, &last.value, &LocalValues::new())
    }

    fn strings(values: &[&str]) -> LiteralValue {
        LiteralValue::Sequence(
            values
                .iter()
                .map(|v| LiteralValue::String((*v).into()))
                .collect(),
        )
    }

    #[test]
    fn test_get_args_flattens_and_deduplicates_literals() {
        let source = indoc::indoc! {r#"
            import typing
            from typing import Literal, TypeAlias, get_args
            Inner = Literal["a", "b"]
            Outer: TypeAlias = Literal[Inner, "c", typing.Literal["a", "d"]]
            get_args(Outer)
        "#};
        assert_eq!(resolve_last(source), Some(strings(&["a", "b", "c", "d"])));
    }

    #[test]
    fn test_get_args_keeps_values_of_different_types() {
        let source = indoc::indoc! {r#"
            from typing_extensions import Literal, get_args
            get_args(Literal[1, True, None, "1"])
        "#};
        assert_eq!(
            resolve_last(source),
            Some(LiteralValue::Sequence(vec![
                LiteralValue::Int(1),
                LiteralValue::Bool(true),
                LiteralValue::None,
                LiteralValue::String("1".into()),
            ]))
        );
    }

    #[test]
    fn test_get_args_needs_typing_and_a_literal() {
        for source in [
            "from typing import Literal, Union, get_args\nget_args(Union[Literal['a'], Literal['b']])",
            "from typing import Literal\ndef get_args(x): return x\nget_args(Literal['a'])",
            "from typing import get_args\nfrom other import Literal\nget_args(Literal['a'])",
            "from typing import Literal, get_args\nget_args(Literal[b'a'])",
        ] {
            assert_eq!(resolve_last(source), None, "{source}");
        }
    }

    #[test]
    fn test_class_calls_are_objects() {
        let source = indoc::indoc! {r#"
            from dataclasses import dataclass
            class _Base:
                @classmethod
                def create(cls) -> "_Base": return cls()
            @dataclass(frozen=True)
            class _Row(_Base):
                class Inner(Exception):
                    pass
            [_Row(), _Row.Inner(), _Row.create()]
        "#};
        assert_eq!(
            resolve_last(source),
            Some(LiteralValue::Sequence(vec![LiteralValue::Opaque; 3]))
        );
    }

    #[test]
    fn test_value_like_classes_are_not_objects() {
        for source in [
            "import enum\nclass _E(enum.Enum):\n    A = 1\n_E(1)",
            "class _S(str):\n    pass\n_S('x')",
            "class _N:\n    def __new__(cls): return 1\n_N()",
            "class _M(metaclass=type):\n    pass\n_M()",
            "class _G:\n    def __getattr__(self, name): return name\n_G()",
            "def wrap(cls): return cls\n@wrap\nclass _W:\n    pass\n_W()",
            "from vendor import StrEnum\nclass _V(StrEnum):\n    pass\n_V()",
            "class _B:\n    @classmethod\n    def make(cls) -> int: return 1\nclass _C(_B):\n    pass\n_C.make()",
        ] {
            assert_eq!(resolve_last(source), None, "{source}");
        }
    }

    #[test]
    fn test_allowlisted_callables_and_external_bases_are_objects() {
        for source in [
            "from dateutil.relativedelta import relativedelta\nrelativedelta(days=1)",
            "import pyarrow as pa\npa.array([])",
            "import uuid\nuuid.UUID(int=1)",
            "from contextlib import nullcontext as does_not_raise\ndoes_not_raise()",
            "import dataclasses\ndataclasses.replace(x, y=1)",
            "from datetime import timezone, timedelta\ntimezone(timedelta(hours=1))",
            "object()",
            "from vendor import Model\nclass _M(Model):\n    pass\n_M()",
        ] {
            assert_eq!(resolve_last(source), Some(LiteralValue::Opaque), "{source}");
        }
        assert_eq!(
            resolve_last("from vendor import relativedelta\nrelativedelta(days=1)"),
            None
        );
    }
}
