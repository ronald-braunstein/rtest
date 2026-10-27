//! AST-based expansion of `@rtest.mark.cases` and `@pytest.mark.parametrize` decorators.
//!
//! This module extracts test case information from decorator AST nodes and expands
//! parametrized tests into individual test cases during collection.

use ruff_python_ast::name::UnqualifiedName;
use ruff_python_ast::{Decorator, Expr, ExprCall, ExprList, ExprTuple, Keyword};

use super::constant_resolver::{
    try_extract_literal, ConstantResolver, LocalValues, ResolvedConstant,
};
use super::discovery::TestDiscoveryConfig;

/// A literal value that can be statically extracted from AST.
#[derive(Debug, Clone, PartialEq)]
pub enum LiteralValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    None,
    /// A tuple/list of literal values (for multi-param cases like `(1, "a")`).
    Sequence(Vec<LiteralValue>),
    /// A value we can count but cannot statically evaluate (e.g., dataclass instances, dicts).
    /// Used for generating positional fallback IDs.
    Opaque,
    /// An enum member. Pytest IDs use `str(member)` → `Class.MEMBER`.
    EnumMember(String),
    /// A class or function. Pytest IDs use its `__name__`.
    Named(String),
    /// A dict literal: keys in insertion order, values when they are statically known.
    Mapping(Vec<(LiteralValue, Option<LiteralValue>)>),
    /// `pytest.param(*values, id=...)` stored in a constant.
    Param {
        values: Vec<LiteralValue>,
        id: Option<String>,
    },
    /// An instance of a plain record class (see the resolver's record shapes) with the
    /// attributes its constructor sets, when known. Pytest IDs it positionally.
    Record(Vec<(String, Option<LiteralValue>)>),
    /// A `set` or `frozenset` of hashable scalars, deduplicated in source order. Its
    /// iteration order depends on hashing, so it is only expanded through `sorted()`.
    Set(Vec<LiteralValue>),
}

/// Specification for a single `@cases` or `@parametrize` decorator.
#[derive(Debug, Clone)]
pub struct CasesSpec {
    /// Final case IDs, including any explicit overrides.
    pub ids: Vec<String>,
    /// Pytest IDs, already made unique per decorator; any duplicates left need runtime collection.
    pub pytest_unique_ids: bool,
}

/// A project decorator that forwards to `pytest.mark.parametrize(argname, records, ids=...)`.
///
/// Configured under `[[tool.rtest.parametrize-wrappers]]`. The wrapper takes
/// `(argnames, argvalues, ids=None)` and passes every case to the test as one record argument:
/// argvalues that are already `record` calls are used as-is, other rows are zipped with
/// `argnames` into records. When `id_field` appears in `argnames`, each record's `id_field`
/// value is the case ID (pytest's `ids=attrgetter(id_field)`); otherwise IDs are `argnameN`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParametrizeWrapper {
    /// Fully qualified decorator, e.g. `package.testing.parametrize`.
    pub decorator: String,
    /// The single argument name the wrapper passes to pytest.
    pub argname: String,
    /// Fully qualified record class whose keyword arguments become attributes.
    pub record: Option<String>,
    /// Record attribute used as the case ID when it is listed in argnames.
    pub id_field: Option<String>,
}

/// How `@pytest.mark.parametrize` duplicate IDs are made unique; pytest 8 changed the rule.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DuplicateIds {
    /// Pytest version unknown, or duplicates are a collection error: collect them with pytest.
    #[default]
    RuntimeCollection,
    /// Pytest 7 appends a counter to every copy (`a0`, `a1`), even when that collides.
    Pytest7,
    /// Pytest 8+ adds `_` after a trailing digit (`1_0`) and skips counters that collide.
    Pytest8,
}

impl DuplicateIds {
    /// Choose the rule for an installed pytest version.
    ///
    /// Pytest 9 can reject duplicates (`strict_parametrization_ids`), so callers pass whether
    /// the configuration may enable that.
    pub fn for_pytest(major: u32, strict_ids_possible: bool) -> Self {
        match major {
            7 => Self::Pytest7,
            8 => Self::Pytest8,
            9.. if !strict_ids_possible => Self::Pytest8,
            _ => Self::RuntimeCollection,
        }
    }

    /// Make one parametrize call's IDs unique the way pytest does.
    fn apply(self, ids: Vec<String>) -> Vec<String> {
        let mut counts = std::collections::HashMap::<String, usize>::new();
        for id in &ids {
            *counts.entry(id.clone()).or_default() += 1;
        }
        if self == Self::RuntimeCollection || counts.values().all(|&count| count == 1) {
            return ids;
        }
        let mut next_suffix = std::collections::HashMap::<String, usize>::new();
        let mut resolved = ids.clone();
        for (index, id) in ids.iter().enumerate() {
            if counts[id] == 1 {
                continue;
            }
            let separator = match self {
                Self::Pytest8 if id.ends_with(|c: char| c.is_ascii_digit()) => "_",
                _ => "",
            };
            let suffix = next_suffix.entry(id.clone()).or_default();
            let mut unique = format!("{id}{separator}{suffix}");
            while self == Self::Pytest8 && resolved.contains(&unique) {
                *suffix += 1;
                unique = format!("{id}{separator}{suffix}");
            }
            resolved[index] = unique;
            *suffix += 1;
        }
        resolved
    }
}

/// Parsed decorators cached for inheritance; an empty list means no cases decorators.
pub type MethodCasesInfo = Result<Vec<CasesSpec>, CannotExpandReason>;

/// Reason why cases could not be statically expanded.
#[derive(Debug, Clone)]
pub enum CannotExpandReason {
    /// Argvalues references a variable, e.g., `DATA`.
    VariableReference(String),
    /// Argvalues contains a function call, e.g., `get_data()`.
    FunctionCall(String),
    /// Argvalues contains a list/dict/set comprehension.
    Comprehension,
    /// Catch-all for other unsupported expressions.
    UnsupportedExpression(String),
}

impl std::fmt::Display for CannotExpandReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VariableReference(name) => {
                write!(f, "argvalues references variable '{}'", name)
            }
            Self::FunctionCall(name) => {
                write!(f, "argvalues contains function call '{}'", name)
            }
            Self::Comprehension => {
                write!(f, "argvalues contains a comprehension")
            }
            Self::UnsupportedExpression(desc) => {
                write!(f, "argvalues contains unsupported expression: {}", desc)
            }
        }
    }
}

/// Result of attempting to expand test cases from decorators.
#[derive(Debug, Clone)]
pub enum CasesExpansion {
    /// No `@cases` or `@parametrize` decorators found.
    NotDecorated,
    /// Successfully expanded to multiple test cases.
    Expanded(Vec<String>),
    /// Cannot statically expand; fall back to base test name.
    CannotExpand(CannotExpandReason),
}

/// Format a warning message for tests that cannot be statically expanded.
pub fn format_cannot_expand_warning(nodeid: &str, reason: &CannotExpandReason) -> String {
    format!(
        "warning: Cannot statically expand test cases for '{}': {}",
        nodeid, reason
    )
}

/// How parametrize case IDs are generated from resolved constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamIdStyle {
    /// `@rtest.mark.cases`: use the source path (`Color.RED`, `TEST_DATA[1]`).
    SourcePath,
    /// `@pytest.mark.parametrize`: match pytest IDs (runtime values; `str(enum)`).
    RuntimeValue,
}

/// Parse decorators and return the cases expansion result.
///
/// The resolver supplies literal values for constant references such as `Color.RED`.
pub fn parse_decorators_for_cases(
    decorators: &[Decorator],
    resolver: &ConstantResolver,
    enclosing_class: Option<&str>,
    config: &TestDiscoveryConfig,
) -> CasesExpansion {
    match parse_decorators_to_specs(decorators, resolver, enclosing_class, config) {
        Ok(specs) => expand_cases(specs.iter()),
        Err(reason) => CasesExpansion::CannotExpand(reason),
    }
}

/// Parse decorators into specs without expanding.
///
/// This is useful for caching method-level specs separately from class-level specs,
/// allowing proper combination during inheritance.
pub fn parse_decorators_to_specs(
    decorators: &[Decorator],
    resolver: &ConstantResolver,
    enclosing_class: Option<&str>,
    config: &TestDiscoveryConfig,
) -> MethodCasesInfo {
    if resolver.has_runtime_parameters() {
        return Err(CannotExpandReason::UnsupportedExpression(
            "runtime fixtures, markers, or pytest_generate_tests".into(),
        ));
    }
    decorators
        .iter()
        .filter_map(|decorator| {
            parse_single_decorator(decorator, resolver, enclosing_class, config).transpose()
        })
        .collect()
}

/// Combine class-level and method-level specs, then expand.
///
/// Class specs become outer (slower-varying) parameters, method specs become
/// inner (faster-varying) parameters. This matches pytest's behavior for
/// class-level `@parametrize` decorators.
pub fn combine_and_expand_specs(
    class_info: &MethodCasesInfo,
    method_info: &MethodCasesInfo,
) -> CasesExpansion {
    match (class_info, method_info) {
        (Err(reason), _) | (_, Err(reason)) => CasesExpansion::CannotExpand(reason.clone()),
        (Ok(class), Ok(method)) => expand_cases(class.iter().chain(method)),
    }
}

/// Parse a single decorator to extract cases information.
fn parse_single_decorator(
    decorator: &Decorator,
    resolver: &ConstantResolver,
    enclosing_class: Option<&str>,
    config: &TestDiscoveryConfig,
) -> Result<Option<CasesSpec>, CannotExpandReason> {
    let Expr::Call(call) = &decorator.expression else {
        return Ok(None);
    };
    let duplicate_ids = config.duplicate_ids;

    let decorator_name = resolver.imported_expression_name(&call.func);
    if let Some(wrapper) = config
        .parametrize_wrappers
        .iter()
        .find(|wrapper| decorator_name.as_deref() == Some(wrapper.decorator.as_str()))
    {
        let ids = extract_wrapper_ids(call, wrapper, resolver, enclosing_class)?;
        return Ok(Some(CasesSpec {
            ids: duplicate_ids.apply(ids),
            pytest_unique_ids: true,
        }));
    }

    let Some(id_style) = decorator_id_style(&call.func, resolver) else {
        return Ok(None);
    };

    let [argnames, values, ..] = call.arguments.args.as_ref() else {
        return Err(CannotExpandReason::UnsupportedExpression(
            "missing required arguments".to_string(),
        ));
    };

    let parameters = extract_argnames(argnames)?;

    let ids_lambda = call
        .arguments
        .keywords
        .iter()
        .find(|kw| kw.arg.as_deref() == Some("ids"))
        .and_then(|kw| kw.value.as_lambda_expr());
    if let (ParamIdStyle::RuntimeValue, Some(lambda)) = (id_style, ids_lambda) {
        let ids = lambda_ids(lambda, values, &parameters, resolver, enclosing_class)?;
        return Ok(Some(CasesSpec {
            ids: duplicate_ids.apply(ids),
            pytest_unique_ids: true,
        }));
    }

    let ids = match (
        id_style,
        extract_ids_kwarg(&call.arguments.keywords, resolver, enclosing_class),
    ) {
        (ParamIdStyle::RuntimeValue, Ok(Some(ids))) => {
            let count = sequence_elts(values)
                .filter(|values| {
                    values.iter().all(|value| {
                        !value.is_starred_expr()
                            && parameters.accepts_row_size(
                                sequence_elts(value)
                                    .filter(|row| !row.iter().any(Expr::is_starred_expr))
                                    .map(<[Expr]>::len),
                            )
                    })
                })
                .map(<[Expr]>::len)
                .or_else(|| {
                    let values = resolve_argvalues(values, resolver, enclosing_class)?;
                    values
                        .iter()
                        .all(|value| {
                            // A param's own id wins over `ids=`; leave that to pytest.
                            !matches!(value, LiteralValue::Param { id: Some(_), .. })
                                && parameters.accepts_row_size(match value {
                                    LiteralValue::Sequence(row) => Some(row.len()),
                                    LiteralValue::Param { values, .. } => Some(values.len()),
                                    _ => None,
                                })
                        })
                        .then_some(values.len())
                });
            if count != Some(ids.len()) {
                return Err(CannotExpandReason::UnsupportedExpression(
                    "ids length must match a static case sequence".into(),
                ));
            }
            ids
        }
        (ParamIdStyle::RuntimeValue, Err(reason)) => return Err(reason),
        (_, supplied) => {
            let defaults =
                extract_argvalues(values, resolver, enclosing_class, &parameters, id_style)?;
            let ids = supplied.ok().flatten().unwrap_or_default();
            let (count, supplied_count) = (defaults.len(), ids.len());
            ids.into_iter()
                .chain(defaults.into_iter().skip(supplied_count))
                .take(count)
                .collect()
        }
    };
    let pytest_unique_ids = id_style == ParamIdStyle::RuntimeValue;
    Ok(Some(CasesSpec {
        ids: if pytest_unique_ids {
            duplicate_ids.apply(ids)
        } else {
            ids
        },
        pytest_unique_ids,
    }))
}

/// Case IDs for a configured parametrize wrapper (see [`ParametrizeWrapper`]).
///
/// Only the case count and record IDs are needed: other record fields never reach the ID.
fn extract_wrapper_ids(
    call: &ExprCall,
    wrapper: &ParametrizeWrapper,
    resolver: &ConstantResolver,
    enclosing_class: Option<&str>,
) -> Result<Vec<String>, CannotExpandReason> {
    let unsupported = |why: &str| CannotExpandReason::UnsupportedExpression(why.into());
    let [argnames, values] = call.arguments.args.as_ref() else {
        return Err(unsupported(
            "wrapper takes argnames, argvalues, and keyword ids",
        ));
    };
    let cases = sequence_elts(values)
        .filter(|cases| !cases.iter().any(Expr::is_starred_expr))
        .ok_or_else(|| unsupported("wrapper argvalues must be a list or tuple"))?;

    // The wrapper replaces falsy `ids` (None or empty) with the id field when it can.
    let explicit_ids = extract_ids_kwarg(&call.arguments.keywords, resolver, enclosing_class)?;
    match &explicit_ids {
        Some(ids) if !ids.is_empty() && ids.len() == cases.len() => return Ok(ids.clone()),
        Some(ids) if !ids.is_empty() => {
            return Err(unsupported("ids length must match a static case sequence"))
        }
        _ => {}
    }
    let positional_ids = || {
        if explicit_ids.is_some() && !cases.is_empty() {
            // Pytest rejects `ids=[]` for a non-empty case list.
            return Err(unsupported("ids length must match a static case sequence"));
        }
        Ok((0..cases.len())
            .map(|index| format!("{}{index}", wrapper.argname))
            .collect())
    };

    let Some(id_field) = wrapper.id_field.as_deref() else {
        return positional_ids();
    };
    // `id_field in argnames`: substring test for a string, membership for a sequence.
    let row_names = match argnames {
        Expr::StringLiteral(names) if names.value.to_str().contains(id_field) => None,
        Expr::StringLiteral(_) => return positional_ids(),
        _ => Some(extract_argnames(argnames)?.names),
    };
    let id_index = match &row_names {
        Some(names) => match names.iter().position(|name| name == id_field) {
            Some(index) => Some(index),
            None => return positional_ids(),
        },
        None => None,
    };
    let positional = |index: usize| format!("{}{index}", wrapper.argname);

    fn as_record<'a>(
        case: &'a Expr,
        wrapper: &ParametrizeWrapper,
        resolver: &ConstantResolver,
    ) -> Option<&'a ExprCall> {
        match case {
            Expr::Call(record)
                if wrapper.record.is_some()
                    && resolver.imported_expression_name(&record.func) == wrapper.record =>
            {
                Some(record)
            }
            _ => None,
        }
    }
    let record_call = |case| as_record(case, wrapper, resolver);
    let records = cases
        .first()
        .is_some_and(|case| record_call(case).is_some());
    cases
        .iter()
        .enumerate()
        .map(|(index, case)| {
            let id_expr = if records {
                let record = record_call(case)
                    .ok_or_else(|| unsupported("wrapper cases mix records and rows"))?;
                if record.arguments.keywords.iter().any(|kw| kw.arg.is_none()) {
                    return Err(unsupported("record uses **kwargs"));
                }
                record
                    .arguments
                    .find_keyword(id_field)
                    .map(|kw| &kw.value)
                    .ok_or_else(|| unsupported("record has no static id field"))?
            } else {
                // Rows are zipped with argnames, so the ID comes from the matching column.
                let row = sequence_elts(case)
                    .filter(|row| !row.iter().any(Expr::is_starred_expr))
                    .ok_or_else(|| unsupported("wrapper row must be a list or tuple"))?;
                id_index
                    .and_then(|index| row.get(index))
                    .ok_or_else(|| unsupported("wrapper row has no id column"))?
            };
            let (id, _) = extract_literal(
                id_expr,
                resolver,
                enclosing_class,
                ParamIdStyle::RuntimeValue,
            )?;
            match id {
                LiteralValue::None => Ok(positional(index)),
                LiteralValue::Int(_)
                | LiteralValue::Float(_)
                | LiteralValue::String(_)
                | LiteralValue::Bool(_)
                | LiteralValue::EnumMember(_)
                | LiteralValue::Named(_) => Ok(literal_to_id_string(&id)),
                _ => Err(unsupported("record id is not a static scalar")),
            }
        })
        .collect()
}

/// Identify `@rtest.mark.cases` and imported aliases of `@pytest.mark.parametrize`.
fn decorator_id_style(func: &Expr, resolver: &ConstantResolver) -> Option<ParamIdStyle> {
    match UnqualifiedName::from_expr(func)?.segments() {
        ["rtest", "mark", "cases"] => Some(ParamIdStyle::SourcePath),
        _ => (resolver.imported_expression_name(func).as_deref()
            == Some("pytest.mark.parametrize"))
        .then_some(ParamIdStyle::RuntimeValue),
    }
}

/// Nonempty argument names and whether each case unpacks a row.
struct CaseParameters {
    names: Vec<String>,
    unpack_rows: bool,
}

impl CaseParameters {
    fn name(&self, index: usize) -> &str {
        self.names.get(index).map(String::as_str).unwrap_or("arg")
    }

    fn accepts_row_size(&self, size: Option<usize>) -> bool {
        !self.unpack_rows || size == Some(self.names.len())
    }
}

/// Normalize scalar versus unpacked rows at the AST boundary.
fn extract_argnames(expr: &Expr) -> Result<CaseParameters, CannotExpandReason> {
    let (names, unpack_rows) = match expr {
        Expr::StringLiteral(s) => {
            let names: Vec<String> = s
                .value
                .to_str()
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            (names, false)
        }
        // Support list/tuple of strings: ["a", "b", "c"] or ("a", "b", "c")
        Expr::List(ExprList { elts, .. }) | Expr::Tuple(ExprTuple { elts, .. }) => {
            let names = elts
                .iter()
                .map(|elt| {
                    elt.as_string_literal_expr()
                        .map(|s| s.value.to_str().to_string())
                        .ok_or_else(|| {
                            CannotExpandReason::UnsupportedExpression(
                                "argnames list must contain only strings".into(),
                            )
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            (names, true)
        }
        Expr::Name(name) => return Err(CannotExpandReason::VariableReference(name.id.to_string())),
        _ => {
            return Err(CannotExpandReason::UnsupportedExpression(
                "argnames must be a string or list/tuple of strings".to_string(),
            ))
        }
    };
    if names.is_empty() {
        return Err(CannotExpandReason::UnsupportedExpression(
            "empty argnames".into(),
        ));
    }
    Ok(CaseParameters {
        unpack_rows: unpack_rows || names.len() > 1,
        names,
    })
}

/// Keep positional placeholders separate from real IDs (including "" and "-").
#[derive(Debug)]
enum CaseId {
    Value(String),
    Positional,
    Parts(Vec<CaseId>),
}

impl CaseId {
    fn from_literal(value: &LiteralValue) -> Self {
        match value {
            LiteralValue::Opaque
            | LiteralValue::Mapping(_)
            | LiteralValue::Param { .. }
            | LiteralValue::Record(_)
            | LiteralValue::Set(_) => Self::Positional,
            LiteralValue::Sequence(values) => {
                Self::Parts(values.iter().map(Self::from_literal).collect())
            }
            _ => Self::Value(literal_to_id_string(value)),
        }
    }

    /// The ID of one parameter's value: pytest IDs a list or tuple value positionally.
    fn component(self) -> Self {
        match self {
            Self::Parts(_) => Self::Positional,
            id => id,
        }
    }

    fn render(&self, argnames: &CaseParameters, index: usize, offset: usize) -> String {
        match self {
            Self::Value(id) => id.clone(),
            Self::Positional => format!("{}{index}", argnames.name(offset)),
            Self::Parts(parts) => parts
                .iter()
                .enumerate()
                .map(|(i, part)| part.render(argnames, index, offset + i))
                .collect::<Vec<_>>()
                .join("-"),
        }
    }
}

/// Extract argument values from the second decorator argument.
///
/// Returns the case ID for each value; values are only needed while deriving IDs.
/// For opaque values (complex objects we can't evaluate), generates positional IDs
/// using the corresponding argname (e.g., `data0`, `num1`).
fn extract_argvalues(
    expr: &Expr,
    resolver: &ConstantResolver,
    enclosing_class: Option<&str>,
    argnames: &CaseParameters,
    id_style: ParamIdStyle,
) -> Result<Vec<String>, CannotExpandReason> {
    match expr {
        Expr::List(ExprList { elts, .. }) | Expr::Tuple(ExprTuple { elts, .. })
            if id_style != ParamIdStyle::RuntimeValue
                || !elts.iter().any(Expr::is_starred_expr) =>
        {
            elts.iter()
                .enumerate()
                .map(|(idx, elt)| {
                    if id_style == ParamIdStyle::RuntimeValue {
                        return extract_pytest_case(elt, argnames, idx, resolver, enclosing_class);
                    }
                    let (_, id) = extract_literal(elt, resolver, enclosing_class, id_style)?;
                    Ok(id.render(argnames, idx, 0))
                })
                .collect()
        }
        // A named sequence keeps source-path IDs for cases and value IDs for parametrize.
        Expr::Name(name) => {
            let (values, source_path) = match resolver.resolve_in_class(enclosing_class, expr) {
                Some(ResolvedConstant {
                    value: LiteralValue::Sequence(values),
                    source_path,
                }) => (values, source_path),
                Some(ResolvedConstant {
                    value: LiteralValue::Mapping(entries),
                    source_path,
                }) if id_style == ParamIdStyle::RuntimeValue => (
                    entries.into_iter().map(|(key, _)| key).collect(),
                    source_path,
                ),
                _ => return Err(CannotExpandReason::VariableReference(name.id.to_string())),
            };
            Ok(match id_style {
                ParamIdStyle::SourcePath => {
                    let source_name = source_path.join(".");
                    values
                        .iter()
                        .map(|v| format!("{}[{}]", source_name, literal_to_id_string(v)))
                        .collect()
                }
                ParamIdStyle::RuntimeValue => values
                    .iter()
                    .enumerate()
                    .map(|(idx, v)| id_for_resolved_case_value(v, argnames, idx))
                    .collect(),
            })
        }
        _ if id_style == ParamIdStyle::RuntimeValue => {
            let values =
                resolve_argvalues(expr, resolver, enclosing_class).ok_or_else(|| match expr {
                    Expr::Call(call) => CannotExpandReason::FunctionCall(get_call_name(&call.func)),
                    Expr::ListComp(_)
                    | Expr::SetComp(_)
                    | Expr::DictComp(_)
                    | Expr::Generator(_) => CannotExpandReason::Comprehension,
                    _ => CannotExpandReason::UnsupportedExpression(
                        "argvalues must be a list or tuple".to_string(),
                    ),
                })?;
            Ok(values
                .iter()
                .enumerate()
                .map(|(idx, v)| id_for_resolved_case_value(v, argnames, idx))
                .collect())
        }
        Expr::Call(call) => {
            let func_name = get_call_name(&call.func);
            Err(CannotExpandReason::FunctionCall(func_name))
        }
        Expr::ListComp(_) | Expr::SetComp(_) | Expr::DictComp(_) | Expr::Generator(_) => {
            Err(CannotExpandReason::Comprehension)
        }
        _ => Err(CannotExpandReason::UnsupportedExpression(
            "argvalues must be a list or tuple".to_string(),
        )),
    }
}

/// Most comprehension results a decorator may expand to.
const MAX_COMPREHENSION_LEN: usize = 10_000;

/// Statically evaluate a non-literal argvalues expression to its sequence of cases.
fn resolve_argvalues(
    expr: &Expr,
    resolver: &ConstantResolver,
    enclosing_class: Option<&str>,
) -> Option<Vec<LiteralValue>> {
    let value = match expr {
        Expr::ListComp(comp) => {
            eval_comprehension(&comp.elt, &comp.generators, resolver, enclosing_class)?
        }
        Expr::Generator(comp) => {
            eval_comprehension(&comp.elt, &comp.generators, resolver, enclosing_class)?
        }
        _ => resolver.resolve_expr(enclosing_class, expr, &LocalValues::new())?,
    };
    match value {
        LiteralValue::Sequence(values) => Some(values),
        LiteralValue::Mapping(entries) => Some(entries.into_iter().map(|(key, _)| key).collect()),
        _ => None,
    }
}

fn eval_comprehension(
    elt: &Expr,
    generators: &[ruff_python_ast::Comprehension],
    resolver: &ConstantResolver,
    enclosing_class: Option<&str>,
) -> Option<LiteralValue> {
    let evaluate =
        |expr: &Expr, locals: &LocalValues| resolver.resolve_expr(enclosing_class, expr, locals);
    let mut values = Vec::new();
    eval_generators(elt, generators, &evaluate, &LocalValues::new(), &mut values)?;
    Some(LiteralValue::Sequence(values))
}

/// Evaluates expressions with comprehension variables bound in `locals`.
pub(crate) type Evaluate<'a> = dyn Fn(&Expr, &LocalValues) -> Option<LiteralValue> + 'a;

pub(crate) fn eval_generators(
    elt: &Expr,
    generators: &[ruff_python_ast::Comprehension],
    evaluate: &Evaluate,
    locals: &LocalValues,
    values: &mut Vec<LiteralValue>,
) -> Option<()> {
    let Some((generator, rest)) = generators.split_first() else {
        values.push(evaluate(elt, locals)?);
        return (values.len() <= MAX_COMPREHENSION_LEN).then_some(());
    };
    if generator.is_async {
        return None;
    }
    let items = match evaluate(&generator.iter, locals)? {
        LiteralValue::Sequence(items) => items,
        LiteralValue::Mapping(entries) => entries.into_iter().map(|(key, _)| key).collect(),
        _ => return None,
    };
    for item in items {
        let mut scope = locals.clone();
        bind_target(&generator.target, item, &mut scope)?;
        let mut keep = true;
        for condition in &generator.ifs {
            keep &= eval_condition(condition, evaluate, &scope)?;
        }
        if keep {
            eval_generators(elt, rest, evaluate, &scope, values)?;
        }
    }
    Some(())
}

fn bind_target(target: &Expr, value: LiteralValue, locals: &mut LocalValues) -> Option<()> {
    match (target, value) {
        (Expr::Name(name), value) => {
            locals.insert(name.id.to_string(), value);
            Some(())
        }
        (
            Expr::Tuple(ExprTuple { elts, .. }) | Expr::List(ExprList { elts, .. }),
            LiteralValue::Sequence(values),
        ) if elts.len() == values.len() => elts
            .iter()
            .zip(values)
            .try_for_each(|(target, value)| bind_target(target, value, locals)),
        _ => None,
    }
}

/// Evaluate a comprehension filter; `None` when Python's result cannot be known statically.
fn eval_condition(expr: &Expr, evaluate: &Evaluate, locals: &LocalValues) -> Option<bool> {
    use ruff_python_ast::{BoolOp, CmpOp, UnaryOp};
    let resolve = |expr: &Expr| evaluate(expr, locals);
    match expr {
        Expr::BoolOp(bool_op) => {
            let results = bool_op
                .values
                .iter()
                .map(|value| eval_condition(value, evaluate, locals))
                .collect::<Option<Vec<_>>>()?;
            Some(match bool_op.op {
                BoolOp::And => results.iter().all(|r| *r),
                BoolOp::Or => results.iter().any(|r| *r),
            })
        }
        Expr::UnaryOp(unary) if unary.op == UnaryOp::Not => {
            eval_condition(&unary.operand, evaluate, locals).map(|r| !r)
        }
        Expr::Compare(compare) => {
            let ([op], [left, right]) = (compare.ops.as_ref(), compare.operands.as_ref()) else {
                return None;
            };
            let (left, right) = (resolve(left)?, resolve(right)?);
            match op {
                CmpOp::Eq => literals_equal(&left, &right),
                CmpOp::NotEq => literals_equal(&left, &right).map(|r| !r),
                CmpOp::In | CmpOp::NotIn => {
                    let items = match right {
                        LiteralValue::Sequence(items) | LiteralValue::Set(items) => items,
                        LiteralValue::Mapping(entries) => {
                            entries.into_iter().map(|(key, _)| key).collect()
                        }
                        _ => return None,
                    };
                    let mut found = false;
                    for item in &items {
                        found |= literals_equal(&left, item)?;
                    }
                    Some(found == (*op == CmpOp::In))
                }
                _ => None,
            }
        }
        expr => match resolve(expr)? {
            LiteralValue::Bool(value) => Some(value),
            LiteralValue::Int(value) => Some(value != 0),
            LiteralValue::String(value) => Some(!value.is_empty()),
            LiteralValue::None => Some(false),
            LiteralValue::Sequence(items) | LiteralValue::Set(items) => Some(!items.is_empty()),
            LiteralValue::Mapping(entries) => Some(!entries.is_empty()),
            // Objects may define `__bool__` or `__len__`.
            _ => None,
        },
    }
}

/// Python `==` for statically known values; `None` where numeric or enum coercion could apply.
fn literals_equal(left: &LiteralValue, right: &LiteralValue) -> Option<bool> {
    use LiteralValue as V;
    match (left, right) {
        (V::String(a), V::String(b)) => Some(a == b),
        (V::Int(a), V::Int(b)) => Some(a == b),
        (V::Bool(a), V::Bool(b)) => Some(a == b),
        (V::None, V::None) => Some(true),
        (V::Named(a), V::Named(b)) => Some(a == b),
        (V::EnumMember(a), V::EnumMember(b)) => Some(a == b),
        (V::Sequence(a), V::Sequence(b)) => {
            if a.len() != b.len() {
                return Some(false);
            }
            let mut equal = true;
            for (a, b) in a.iter().zip(b) {
                equal &= literals_equal(a, b)?;
            }
            Some(equal)
        }
        (
            V::Opaque
            | V::Mapping(_)
            | V::Set(_)
            | V::Param { .. }
            | V::EnumMember(_)
            | V::Float(_),
            _,
        )
        | (
            _,
            V::Opaque
            | V::Mapping(_)
            | V::Set(_)
            | V::Param { .. }
            | V::EnumMember(_)
            | V::Float(_),
        )
        | (V::Int(_) | V::Bool(_), V::Int(_) | V::Bool(_)) => None,
        _ => Some(false),
    }
}

/// One pytest parametrize case: nested list/tuple/dict values use `argnameN` IDs.
fn extract_pytest_case(
    elt: &Expr,
    argnames: &CaseParameters,
    idx: usize,
    resolver: &ConstantResolver,
    enclosing_class: Option<&str>,
) -> Result<String, CannotExpandReason> {
    // Sequence-form argnames unpack rows even for one argument. String "x"
    // keeps a list/tuple row as the value; both forms share the component logic.
    let id = match sequence_elts(elt).filter(|_| argnames.unpack_rows) {
        Some(components) => {
            extract_literal_sequence(
                components,
                resolver,
                enclosing_class,
                ParamIdStyle::RuntimeValue,
            )?
            .1
        }
        None => {
            let (_, id) =
                extract_literal(elt, resolver, enclosing_class, ParamIdStyle::RuntimeValue)?;
            match id {
                CaseId::Parts(parts) if argnames.unpack_rows => {
                    CaseId::Parts(parts.into_iter().map(CaseId::component).collect())
                }
                id => id.component(),
            }
        }
    };
    Ok(id.render(argnames, idx, 0))
}

/// Largest string or sequence built by `*` repetition; larger results need runtime collection.
const MAX_REPEATED_LEN: usize = 100_000;

pub(crate) fn eval_literal_binop(
    op: ruff_python_ast::Operator,
    left: &LiteralValue,
    right: &LiteralValue,
) -> Option<LiteralValue> {
    use ruff_python_ast::Operator;
    match (left, right, op) {
        (LiteralValue::String(a), LiteralValue::String(b), Operator::Add) => {
            return Some(LiteralValue::String(format!("{a}{b}")));
        }
        (LiteralValue::Sequence(a), LiteralValue::Sequence(b), Operator::Add) => {
            return Some(LiteralValue::Sequence([a.as_slice(), b].concat()));
        }
        (LiteralValue::String(s), LiteralValue::Int(n), Operator::Mult)
        | (LiteralValue::Int(n), LiteralValue::String(s), Operator::Mult) => {
            let count = usize::try_from(*n).unwrap_or(0);
            return (s.len().saturating_mul(count) <= MAX_REPEATED_LEN)
                .then(|| LiteralValue::String(s.repeat(count)));
        }
        (LiteralValue::Sequence(items), LiteralValue::Int(n), Operator::Mult)
        | (LiteralValue::Int(n), LiteralValue::Sequence(items), Operator::Mult) => {
            let count = usize::try_from(*n).unwrap_or(0);
            return (items.len().saturating_mul(count) <= MAX_REPEATED_LEN).then(|| {
                LiteralValue::Sequence(
                    std::iter::repeat_n(items, count)
                        .flatten()
                        .cloned()
                        .collect(),
                )
            });
        }
        _ => {}
    }
    let (LiteralValue::Int(l), LiteralValue::Int(r)) = (left, right) else {
        return None;
    };
    let value = match op {
        Operator::Add => l.checked_add(*r)?,
        Operator::Sub => l.checked_sub(*r)?,
        Operator::Mult => l.checked_mul(*r)?,
        Operator::LShift if (0..64).contains(r) => l.checked_shl(*r as u32)?,
        Operator::RShift if (0..64).contains(r) => l.checked_shr(*r as u32)?,
        Operator::BitOr => *l | *r,
        Operator::BitXor => *l ^ *r,
        Operator::BitAnd => *l & *r,
        Operator::Pow if *r >= 0 && *r <= 63 => l.checked_pow(*r as u32)?,
        _ => return None,
    };
    Some(LiteralValue::Int(value))
}

fn sequence_elts(expr: &Expr) -> Option<&[Expr]> {
    match expr {
        Expr::Tuple(ExprTuple { elts, .. }) | Expr::List(ExprList { elts, .. }) => Some(elts),
        _ => None,
    }
}

fn id_for_resolved_case_value(
    value: &LiteralValue,
    argnames: &CaseParameters,
    idx: usize,
) -> String {
    let row;
    let value = match value {
        LiteralValue::Param { id: Some(id), .. } => return ascii_escape_string(id),
        LiteralValue::Param { values, id: None } => match values.as_slice() {
            [single] => single,
            values => {
                row = LiteralValue::Sequence(values.to_vec());
                &row
            }
        },
        value => value,
    };
    let parts = match value {
        LiteralValue::Sequence(parts) if argnames.unpack_rows => parts.as_slice(),
        _ => std::slice::from_ref(value),
    };
    parts
        .iter()
        .enumerate()
        .map(|(i, part)| match part {
            LiteralValue::Sequence(_)
            | LiteralValue::Opaque
            | LiteralValue::Mapping(_)
            | LiteralValue::Param { .. }
            | LiteralValue::Record(_)
            | LiteralValue::Set(_) => format!("{}{}", argnames.name(i), idx),
            _ => literal_to_id_string(part),
        })
        .collect::<Vec<_>>()
        .join("-")
}

/// Extract a literal value from an expression.
///
/// Returns the value and its ID expression, preserving positional placeholders until rendering.
fn extract_literal(
    expr: &Expr,
    resolver: &ConstantResolver,
    enclosing_class: Option<&str>,
    id_style: ParamIdStyle,
) -> Result<(LiteralValue, CaseId), CannotExpandReason> {
    let literal = match expr {
        Expr::NumberLiteral(_)
        | Expr::StringLiteral(_)
        | Expr::BytesLiteral(_)
        | Expr::BooleanLiteral(_)
        | Expr::UnaryOp(_)
        | Expr::NoneLiteral(_) => try_extract_literal(expr).ok_or_else(|| {
            CannotExpandReason::UnsupportedExpression("non-literal scalar".into())
        })?,
        Expr::BinOp(_) | Expr::Call(_) | Expr::Subscript(_) | Expr::FString(_)
            if id_style == ParamIdStyle::RuntimeValue
                && resolver
                    .imported_expression_name(call_func(expr))
                    .as_deref()
                    != Some("pytest.param") =>
        {
            if let Some(value) = resolver.resolve_expr(enclosing_class, expr, &LocalValues::new()) {
                return Ok(element_value(value));
            }
            return extract_unresolved_literal(expr, resolver, enclosing_class, id_style);
        }
        _ => return extract_unresolved_literal(expr, resolver, enclosing_class, id_style),
    };
    let id = CaseId::from_literal(&literal);
    Ok((literal, id))
}

/// The callee of a call expression, or the expression itself.
fn call_func(expr: &Expr) -> &Expr {
    match expr {
        Expr::Call(call) => &call.func,
        _ => expr,
    }
}

/// A single parametrize value: pytest IDs list, tuple, and dict objects positionally.
fn element_value(value: LiteralValue) -> (LiteralValue, CaseId) {
    match value {
        LiteralValue::Sequence(_)
        | LiteralValue::Mapping(_)
        | LiteralValue::Set(_)
        | LiteralValue::Param { .. } => (LiteralValue::Opaque, CaseId::Positional),
        value => {
            let id = CaseId::from_literal(&value);
            (value, id)
        }
    }
}

fn extract_unresolved_literal(
    expr: &Expr,
    resolver: &ConstantResolver,
    enclosing_class: Option<&str>,
    id_style: ParamIdStyle,
) -> Result<(LiteralValue, CaseId), CannotExpandReason> {
    let literal = match expr {
        Expr::BinOp(binop) => {
            let (left, _) = extract_literal(&binop.left, resolver, enclosing_class, id_style)?;
            let (right, _) = extract_literal(&binop.right, resolver, enclosing_class, id_style)?;
            if let Some(lit) = eval_literal_binop(binop.op, &left, &right) {
                let id = CaseId::from_literal(&lit);
                return Ok((lit, id));
            }
            if id_style == ParamIdStyle::RuntimeValue {
                return Err(CannotExpandReason::UnsupportedExpression(format!(
                    "binary op '{}'",
                    binop.op.as_str()
                )));
            }
            LiteralValue::Opaque
        }
        Expr::Lambda(_) if id_style == ParamIdStyle::RuntimeValue => {
            // pytest `_idval_from_value` uses `__name__` → `<lambda>`.
            return Ok((LiteralValue::Opaque, CaseId::Value("<lambda>".to_string())));
        }
        Expr::Starred(_) => {
            return Err(CannotExpandReason::UnsupportedExpression(
                "starred unpack".to_string(),
            ))
        }
        Expr::Tuple(ExprTuple { elts, .. }) | Expr::List(ExprList { elts, .. }) => {
            if id_style == ParamIdStyle::RuntimeValue {
                // Nested list/tuple as a parameter value is opaque to pytest (`argnameN`).
                return Ok((LiteralValue::Opaque, CaseId::Positional));
            }
            return extract_literal_sequence(elts, resolver, enclosing_class, id_style);
        }
        Expr::Name(_) | Expr::Attribute(_) => {
            if let Some(resolved) = resolver.resolve_in_class(enclosing_class, expr) {
                let id = id_for_resolved(&resolved, id_style);
                return Ok((resolved.value, id));
            }
            if id_style == ParamIdStyle::RuntimeValue {
                // Pytest would use str(runtime value). Guessing `argnameN` is a silent mismatch.
                return Err(CannotExpandReason::VariableReference(get_call_name(expr)));
            }
            // Unresolved name/attribute - treat as opaque (can count but not evaluate)
            LiteralValue::Opaque
        }
        Expr::Call(call)
            if resolver.imported_expression_name(&call.func).as_deref() == Some("pytest.param") =>
        {
            return extract_pytest_param(call, resolver, enclosing_class, id_style);
        }
        // Constructors (`MyData(1)`, `date(2024, 1, 1)`) produce objects that pytest IDs as
        // `argnameN`. Helper calls may return str/int/enum values, so they need runtime collection.
        Expr::Call(call) if id_style == ParamIdStyle::RuntimeValue => {
            if !is_constructor_call(&call.func) {
                return Err(CannotExpandReason::FunctionCall(get_call_name(&call.func)));
            }
            LiteralValue::Opaque
        }
        // Comprehensions cannot be counted without evaluation
        Expr::ListComp(_) | Expr::SetComp(_) | Expr::DictComp(_) | Expr::Generator(_) => {
            return Err(CannotExpandReason::Comprehension);
        }
        // Dicts and sets are IDed positionally; other expressions may evaluate to values
        // pytest IDs by value.
        Expr::Dict(_) | Expr::Set(_) => LiteralValue::Opaque,
        Expr::FString(_) | Expr::TString(_) if id_style == ParamIdStyle::RuntimeValue => {
            return Err(CannotExpandReason::UnsupportedExpression("f-string".into()))
        }
        _ if id_style == ParamIdStyle::RuntimeValue => {
            return Err(CannotExpandReason::UnsupportedExpression(
                "expression evaluated at runtime".into(),
            ))
        }
        // Source-path IDs only need a count.
        _ => LiteralValue::Opaque,
    };
    let id = CaseId::from_literal(&literal);
    Ok((literal, id))
}

fn extract_pytest_param(
    call: &ruff_python_ast::ExprCall,
    resolver: &ConstantResolver,
    enclosing_class: Option<&str>,
    id_style: ParamIdStyle,
) -> Result<(LiteralValue, CaseId), CannotExpandReason> {
    // Explicit id= is pytest-compatible even when the value is a helper call.
    if let Some(keyword) = call
        .arguments
        .keywords
        .iter()
        .find(|kw| kw.arg.as_deref() == Some("id") && !matches!(kw.value, Expr::NoneLiteral(_)))
    {
        let (LiteralValue::String(id), _) =
            extract_literal(&keyword.value, resolver, enclosing_class, id_style)?
        else {
            return Err(CannotExpandReason::UnsupportedExpression(
                "pytest.param id= is not a static string".into(),
            ));
        };
        return Ok((
            LiteralValue::String(id.clone()),
            CaseId::Value(ascii_escape_string(&id)),
        ));
    }
    match call.arguments.args.as_ref() {
        [] => Ok((LiteralValue::Opaque, CaseId::Positional)),
        [arg] => extract_literal(arg, resolver, enclosing_class, id_style),
        args => extract_literal_sequence(args, resolver, enclosing_class, id_style),
    }
}

fn extract_literal_sequence(
    expressions: &[Expr],
    resolver: &ConstantResolver,
    enclosing_class: Option<&str>,
    id_style: ParamIdStyle,
) -> Result<(LiteralValue, CaseId), CannotExpandReason> {
    let (values, ids) = expressions
        .iter()
        .map(|expr| {
            let (value, id) = extract_literal(expr, resolver, enclosing_class, id_style)?;
            Ok(match id_style {
                ParamIdStyle::RuntimeValue => (value, id.component()),
                ParamIdStyle::SourcePath => (value, id),
            })
        })
        .collect::<Result<(Vec<_>, Vec<_>), _>>()?;
    Ok((LiteralValue::Sequence(values), CaseId::Parts(ids)))
}

fn id_for_resolved(resolved: &ResolvedConstant, id_style: ParamIdStyle) -> CaseId {
    match id_style {
        ParamIdStyle::SourcePath => CaseId::Value(resolved.source_path.join(".")),
        ParamIdStyle::RuntimeValue => CaseId::from_literal(&resolved.value),
    }
}

/// Stdlib datetime types are constructors but named in lowercase (`from datetime import date`).
/// Pytest IDs those instances as `argnameN`, same as PascalCase classes.
const DATETIME_CONSTRUCTOR_NAMES: &[&str] = &["date", "datetime", "timedelta", "time"];

pub(crate) fn is_constructor_call(func: &Expr) -> bool {
    match func {
        Expr::Name(name) => {
            let id = name.id.as_str();
            DATETIME_CONSTRUCTOR_NAMES.contains(&id)
                || id.chars().next().is_some_and(char::is_uppercase)
        }
        // `datetime.date(...)` / `dt.timedelta(...)` after `import datetime` or `import datetime as dt`.
        // `time.time()` is the stdlib clock: pytest IDs the returned float.
        Expr::Attribute(attr) => {
            let attr_name = attr.attr.as_str();
            DATETIME_CONSTRUCTOR_NAMES.contains(&attr_name)
                && matches!(
                    attr.value.as_ref(),
                    Expr::Name(n) if !(n.id.as_str() == "time" && attr_name == "time")
                )
        }
        _ => false,
    }
}

/// Get the name of a called function for error messages.
fn get_call_name(func: &Expr) -> String {
    UnqualifiedName::from_expr(func)
        .and_then(|name| name.segments().last().map(|segment| (*segment).to_string()))
        .unwrap_or_else(|| "unknown".into())
}

/// Extract the `ids` keyword argument if present.
/// IDs from `ids=lambda value: <expr>` for a single argname. Pytest calls the function on
/// each value (a param's own `id` wins) and IDs the result like a value; `None` keeps the
/// default ID, which is only reproduced here when it is positional.
fn lambda_ids(
    lambda: &ruff_python_ast::ExprLambda,
    values: &Expr,
    parameters: &CaseParameters,
    resolver: &ConstantResolver,
    enclosing_class: Option<&str>,
) -> Result<Vec<String>, CannotExpandReason> {
    let invalid = || {
        CannotExpandReason::UnsupportedExpression(
            "ids= callable must map each static value to a scalar ID".into(),
        )
    };
    let parameter = lambda
        .parameters
        .as_deref()
        .and_then(|params| match (params.args.as_slice(), params.len()) {
            ([single], 1) if single.default.is_none() => Some(single.parameter.name.to_string()),
            _ => None,
        })
        .ok_or_else(invalid)?;
    if parameters.unpack_rows || parameters.names.len() != 1 {
        return Err(invalid());
    }
    let values = resolve_argvalues(values, resolver, enclosing_class).ok_or_else(invalid)?;
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            let value = match value {
                LiteralValue::Param { id: Some(id), .. } => return Ok(ascii_escape_string(id)),
                LiteralValue::Param { values, id: None } => match values.as_slice() {
                    [single] => single,
                    _ => return Err(invalid()),
                },
                value => value,
            };
            let locals = LocalValues::from([(parameter.clone(), value.clone())]);
            match resolver.resolve_expr(enclosing_class, &lambda.body, &locals) {
                Some(LiteralValue::None)
                    if matches!(CaseId::from_literal(value), CaseId::Positional) =>
                {
                    Ok(format!("{}{idx}", parameters.name(0)))
                }
                Some(
                    id @ (LiteralValue::String(_)
                    | LiteralValue::Int(_)
                    | LiteralValue::Float(_)
                    | LiteralValue::Bool(_)
                    | LiteralValue::EnumMember(_)
                    | LiteralValue::Named(_)),
                ) => Ok(literal_to_id_string(&id)),
                _ => Err(invalid()),
            }
        })
        .collect()
}

fn extract_ids_kwarg(
    keywords: &[Keyword],
    resolver: &ConstantResolver,
    enclosing_class: Option<&str>,
) -> Result<Option<Vec<String>>, CannotExpandReason> {
    let Some(kw) = keywords.iter().find(|kw| kw.arg.as_deref() == Some("ids")) else {
        return Ok(None);
    };
    if matches!(kw.value, Expr::NoneLiteral(_)) {
        return Ok(None);
    }
    let invalid = || {
        CannotExpandReason::UnsupportedExpression(
            "ids= requires a static sequence of non-null scalar IDs".into(),
        )
    };
    let value = match &kw.value {
        Expr::ListComp(comp) => {
            eval_comprehension(&comp.elt, &comp.generators, resolver, enclosing_class)
        }
        value => try_extract_literal(value).or_else(|| {
            resolver
                .resolve_in_class(enclosing_class, value)
                .map(|resolved| resolved.value)
        }),
    };
    let Some(LiteralValue::Sequence(ids)) = value else {
        return Err(invalid());
    };
    ids.iter()
        .map(|id| match id {
            LiteralValue::None
            | LiteralValue::Sequence(_)
            | LiteralValue::Opaque
            | LiteralValue::Mapping(_)
            | LiteralValue::Param { .. }
            | LiteralValue::Record(_)
            | LiteralValue::Set(_)
            | LiteralValue::Named(_) => Err(invalid()),
            _ => Ok(literal_to_id_string(id)),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// Convert a literal value to its string representation for use as a case ID.
pub fn literal_to_id_string(value: &LiteralValue) -> String {
    match value {
        LiteralValue::Int(i) => i.to_string(),
        LiteralValue::Float(f) => python_str_float(*f),
        LiteralValue::String(s) => ascii_escape_string(s),
        LiteralValue::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        LiteralValue::None => "None".to_string(),
        LiteralValue::Sequence(seq) => {
            let parts: Vec<String> = seq.iter().map(literal_to_id_string).collect();
            parts.join("-")
        }
        // Opaque values get positional IDs assigned in extract_argvalues
        LiteralValue::Opaque
        | LiteralValue::Mapping(_)
        | LiteralValue::Param { .. }
        | LiteralValue::Record(_)
        | LiteralValue::Set(_) => String::new(),
        LiteralValue::EnumMember(id) | LiteralValue::Named(id) => id.clone(),
    }
}

/// JSON supplies correctly rounded shortest digits; Python uses a different
/// exponent spelling and switches to scientific notation at 1e-4, not 1e-5.
fn python_str_float(f: f64) -> String {
    let s = serde_json::Number::from_f64(f).map_or_else(
        || f.to_string().to_ascii_lowercase(),
        |number| number.to_string(),
    );
    if let Some((mantissa, exponent)) = s.split_once('e') {
        if let Ok(exponent) = exponent.parse::<i32>() {
            return format!("{mantissa}e{exponent:+03}");
        }
    }
    if (1e-5..1e-4).contains(&f.abs()) {
        let mut mantissa = s.replace("0.0000", "");
        mantissa.insert(usize::from(f.is_sign_negative()) + 1, '.');
        return format!("{}e-05", mantissa.trim_end_matches('.'));
    }
    s
}

/// Escape a string for use in test IDs.
///
/// Escapes backslashes, control characters, and non-ASCII characters.
fn ascii_escape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_ascii_control() => {
                result.push_str(&format!("\\x{:02x}", c as u32));
            }
            c if c.is_ascii() => result.push(c),
            c => {
                // Match Python `str.encode("unicode_escape")` / pytest `ascii_escaped`:
                // latin-1 is `\xHH`, BMP `\uXXXX`, supplementary `\UXXXXXXXX`.
                let code = c as u32;
                if code <= 0xFF {
                    result.push_str(&format!("\\x{:02x}", code));
                } else if code <= 0xFFFF {
                    result.push_str(&format!("\\u{:04x}", code));
                } else {
                    result.push_str(&format!("\\U{:08x}", code));
                }
            }
        }
    }
    result
}

/// Expand cases specs into individual test cases using cartesian product.
fn expand_cases<'a>(specs: impl DoubleEndedIterator<Item = &'a CasesSpec>) -> CasesExpansion {
    // Process the innermost decorator first, preserving empty IDs as real components.
    let mut specs = specs.rev();
    let Some(first) = specs.next() else {
        return CasesExpansion::NotDecorated;
    };
    let mut ids = first.ids.clone();
    let mut pytest_unique_ids = first.pytest_unique_ids;
    for spec in specs {
        pytest_unique_ids |= spec.pytest_unique_ids;
        ids = ids
            .iter()
            .flat_map(|existing| spec.ids.iter().map(move |id| format!("{existing}-{id}")))
            .collect();
    }
    if pytest_unique_ids
        && (ids.is_empty()
            || ids.iter().collect::<std::collections::HashSet<_>>().len() != ids.len())
    {
        return CasesExpansion::CannotExpand(CannotExpandReason::UnsupportedExpression(
            "empty or duplicate pytest cases require runtime collection".into(),
        ));
    }
    CasesExpansion::Expanded(deduplicate_ids(ids))
}

/// Deduplicate IDs by adding `_1`, `_2` suffixes for duplicates.
fn deduplicate_ids(ids: Vec<String>) -> Vec<String> {
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut result = Vec::with_capacity(ids.len());

    for id in ids {
        let count = seen.entry(id.clone()).or_insert(0);
        if *count == 0 {
            result.push(id);
        } else {
            result.push(format!("{}_{}", id, count));
        }
        *count += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::DuplicateIds;

    fn apply(rule: DuplicateIds, ids: &[&str]) -> Vec<String> {
        rule.apply(ids.iter().map(|id| (*id).to_string()).collect())
    }

    #[test]
    fn test_duplicate_ids_match_pytest_versions() {
        let cases: [(&[&str], &[&str], &[&str]); 4] = [
            (
                &["True", "False", "False"],
                &["True", "False0", "False1"],
                &["True", "False0", "False1"],
            ),
            (&["1", "1"], &["10", "11"], &["1_0", "1_1"]),
            (&["a", "a", "a0"], &["a0", "a1", "a0"], &["a1", "a2", "a0"]),
            (&["", ""], &["0", "1"], &["0", "1"]),
        ];
        for (ids, pytest7, pytest8) in cases {
            assert_eq!(apply(DuplicateIds::Pytest7, ids), pytest7, "{ids:?}");
            assert_eq!(apply(DuplicateIds::Pytest8, ids), pytest8, "{ids:?}");
            assert_eq!(apply(DuplicateIds::RuntimeCollection, ids), ids, "{ids:?}");
        }
    }

    #[test]
    fn test_duplicate_ids_rule_for_pytest_version() {
        assert_eq!(
            DuplicateIds::for_pytest(6, false),
            DuplicateIds::RuntimeCollection
        );
        assert_eq!(DuplicateIds::for_pytest(7, true), DuplicateIds::Pytest7);
        assert_eq!(DuplicateIds::for_pytest(8, true), DuplicateIds::Pytest8);
        assert_eq!(DuplicateIds::for_pytest(9, false), DuplicateIds::Pytest8);
        assert_eq!(
            DuplicateIds::for_pytest(9, true),
            DuplicateIds::RuntimeCollection
        );
    }
}
