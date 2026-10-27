//! AST-based expansion of `@rtest.mark.cases` and `@pytest.mark.parametrize` decorators.
//!
//! This module extracts test case information from decorator AST nodes and expands
//! parametrized tests into individual test cases during collection.

use ruff_python_ast::name::UnqualifiedName;
use ruff_python_ast::{Decorator, Expr, ExprList, ExprTuple, Keyword};

use super::constant_resolver::{try_extract_literal, ConstantResolver, ResolvedConstant};

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
}

/// Specification for a single `@cases` or `@parametrize` decorator.
#[derive(Debug, Clone)]
pub struct CasesSpec {
    /// Final case IDs, including any explicit overrides.
    pub ids: Vec<String>,
    /// Pytest IDs, already made unique per decorator; any duplicates left need runtime collection.
    pub pytest_unique_ids: bool,
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
    duplicate_ids: DuplicateIds,
) -> CasesExpansion {
    match parse_decorators_to_specs(decorators, resolver, enclosing_class, duplicate_ids) {
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
    duplicate_ids: DuplicateIds,
) -> MethodCasesInfo {
    if resolver.has_runtime_parameters() {
        return Err(CannotExpandReason::UnsupportedExpression(
            "runtime fixtures, markers, or pytest_generate_tests".into(),
        ));
    }
    decorators
        .iter()
        .filter_map(|decorator| {
            parse_single_decorator(decorator, resolver, enclosing_class, duplicate_ids).transpose()
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
    duplicate_ids: DuplicateIds,
) -> Result<Option<CasesSpec>, CannotExpandReason> {
    let Expr::Call(call) = &decorator.expression else {
        return Ok(None);
    };

    let Some(id_style) = decorator_id_style(&call.func, resolver) else {
        return Ok(None);
    };

    let [argnames, values, ..] = call.arguments.args.as_ref() else {
        return Err(CannotExpandReason::UnsupportedExpression(
            "missing required arguments".to_string(),
        ));
    };

    let parameters = extract_argnames(argnames)?;

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
                .or_else(
                    || match resolver.resolve_in_class(enclosing_class, values)?.value {
                        LiteralValue::Sequence(values) => values
                            .iter()
                            .all(|value| {
                                parameters.accepts_row_size(match value {
                                    LiteralValue::Sequence(row) => Some(row.len()),
                                    _ => None,
                                })
                            })
                            .then_some(values.len()),
                        _ => None,
                    },
                );
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
            LiteralValue::Opaque => Self::Positional,
            LiteralValue::Sequence(values) => {
                Self::Parts(values.iter().map(Self::from_literal).collect())
            }
            _ => Self::Value(literal_to_id_string(value)),
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
        Expr::List(ExprList { elts, .. }) | Expr::Tuple(ExprTuple { elts, .. }) => elts
            .iter()
            .enumerate()
            .map(|(idx, elt)| {
                if id_style == ParamIdStyle::RuntimeValue {
                    return extract_pytest_case(elt, argnames, idx, resolver, enclosing_class);
                }
                let (_, id) = extract_literal(elt, resolver, enclosing_class, id_style)?;
                Ok(id.render(argnames, idx, 0))
            })
            .collect(),
        // A named sequence keeps source-path IDs for cases and value IDs for parametrize.
        Expr::Name(name) => {
            let Some(ResolvedConstant {
                value: LiteralValue::Sequence(values),
                source_path,
            }) = resolver.resolve_in_class(enclosing_class, expr)
            else {
                return Err(CannotExpandReason::VariableReference(name.id.to_string()));
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
    let (_, id) = match sequence_elts(elt).filter(|_| argnames.unpack_rows) {
        Some(components) => extract_literal_sequence(
            components,
            resolver,
            enclosing_class,
            ParamIdStyle::RuntimeValue,
        )?,
        None => extract_literal(elt, resolver, enclosing_class, ParamIdStyle::RuntimeValue)?,
    };
    Ok(id.render(argnames, idx, 0))
}

fn eval_literal_binop(
    op: ruff_python_ast::Operator,
    left: &LiteralValue,
    right: &LiteralValue,
) -> Option<LiteralValue> {
    use ruff_python_ast::Operator;
    if let (LiteralValue::String(a), LiteralValue::String(b), Operator::Add) = (left, right, op) {
        return Some(LiteralValue::String(format!("{a}{b}")));
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
    let parts = match value {
        LiteralValue::Sequence(parts) if argnames.unpack_rows => parts.as_slice(),
        _ => std::slice::from_ref(value),
    };
    parts
        .iter()
        .enumerate()
        .map(|(i, part)| match part {
            LiteralValue::Sequence(_) | LiteralValue::Opaque => {
                format!("{}{}", argnames.name(i), idx)
            }
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
        Expr::FString(fstring) if id_style == ParamIdStyle::RuntimeValue => {
            let value = eval_fstring(fstring, resolver, enclosing_class)
                .ok_or_else(|| CannotExpandReason::UnsupportedExpression("f-string".into()))?;
            LiteralValue::String(value)
        }
        // Dicts and sets are IDed positionally; other expressions may evaluate to values
        // pytest IDs by value.
        Expr::Dict(_) | Expr::Set(_) => LiteralValue::Opaque,
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

/// Evaluate an f-string whose interpolations are constants formatted with `str()`.
fn eval_fstring(
    fstring: &ruff_python_ast::ExprFString,
    resolver: &ConstantResolver,
    enclosing_class: Option<&str>,
) -> Option<String> {
    use ruff_python_ast::{ConversionFlag, FStringPartRef, InterpolatedStringElement};
    let mut text = String::new();
    for part in fstring.value.iter() {
        let fstring = match part {
            FStringPartRef::Literal(literal) => {
                text.push_str(&literal.value);
                continue;
            }
            FStringPartRef::FString(fstring) => fstring,
        };
        for element in fstring.elements.iter() {
            let interpolation = match element {
                InterpolatedStringElement::Literal(literal) => {
                    text.push_str(&literal.value);
                    continue;
                }
                InterpolatedStringElement::Interpolation(interpolation) => interpolation,
            };
            if interpolation.debug_text.is_some()
                || interpolation.format_spec.is_some()
                || !matches!(
                    interpolation.conversion,
                    ConversionFlag::None | ConversionFlag::Str
                )
            {
                return None;
            }
            let value = match interpolation.expression.as_ref() {
                expr @ (Expr::Name(_) | Expr::Attribute(_)) => {
                    resolver.resolve_in_class(enclosing_class, expr)?.value
                }
                Expr::BytesLiteral(_) => return None,
                expr => try_extract_literal(expr)?,
            };
            match value {
                LiteralValue::String(value) => text.push_str(&value),
                LiteralValue::Int(value) => text.push_str(&value.to_string()),
                LiteralValue::Bool(value) => text.push_str(if value { "True" } else { "False" }),
                LiteralValue::None => text.push_str("None"),
                _ => return None,
            }
        }
    }
    Some(text)
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
        .map(|expr| extract_literal(expr, resolver, enclosing_class, id_style))
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

fn is_constructor_call(func: &Expr) -> bool {
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
    let value = try_extract_literal(&kw.value).or_else(|| {
        resolver
            .resolve_in_class(enclosing_class, &kw.value)
            .map(|resolved| resolved.value)
    });
    let Some(LiteralValue::Sequence(ids)) = value else {
        return Err(invalid());
    };
    ids.iter()
        .map(|id| match id {
            LiteralValue::None | LiteralValue::Sequence(_) | LiteralValue::Opaque => Err(invalid()),
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
        LiteralValue::Opaque => String::new(),
        LiteralValue::EnumMember(id) => id.clone(),
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
