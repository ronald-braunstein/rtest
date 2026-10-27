//! Configuration parsing for pytest settings from pyproject.toml

use crate::python_discovery::ParametrizeWrapper;
use log::debug;
use std::path::{Path, PathBuf};
use toml::Value;

/// Pytest configuration from pyproject.toml
#[derive(Debug, Clone, Default)]
pub struct PytestConfig {
    /// Test paths to search for tests
    pub testpaths: Vec<PathBuf>,
    /// Glob patterns for test file names (e.g., "test_*.py", "*_test.py")
    pub python_files: Vec<String>,
    /// Patterns for test class names (e.g., "Test*")
    pub python_classes: Vec<String>,
    /// Patterns for test function/method names (e.g., "test*")
    pub python_functions: Vec<String>,
}

/// Read pytest configuration from pyproject.toml
pub fn read_pytest_config(root_path: &Path) -> PytestConfig {
    let pyproject_path = root_path.join("pyproject.toml");

    if !pyproject_path.exists() {
        debug!("No pyproject.toml found at {pyproject_path:?}");
        return PytestConfig::default();
    }

    let content = match std::fs::read_to_string(&pyproject_path) {
        Ok(content) => content,
        Err(e) => {
            debug!("Failed to read pyproject.toml: {e}");
            return PytestConfig::default();
        }
    };

    let toml_value: Value = match toml::from_str(&content) {
        Ok(value) => value,
        Err(e) => {
            debug!("Failed to parse pyproject.toml: {e}");
            return PytestConfig::default();
        }
    };

    let mut config = PytestConfig::default();

    let ini_options = toml_value
        .get("tool")
        .and_then(|t| t.get("pytest"))
        .and_then(|p| p.get("ini_options"));

    if let Some(values) = read_string_array(ini_options, "testpaths") {
        config.testpaths = values.map(PathBuf::from).collect();
        debug!("Found testpaths in pyproject.toml: {:?}", config.testpaths);
    }

    for (name, patterns) in [
        ("python_files", &mut config.python_files),
        ("python_classes", &mut config.python_classes),
        ("python_functions", &mut config.python_functions),
    ] {
        if let Some(values) = read_string_array(ini_options, name) {
            *patterns = values.map(String::from).collect();
            debug!("Found {name} in pyproject.toml: {patterns:?}");
        }
    }

    config
}

/// Whether pytest 9 configuration under `root_path` may turn duplicate parametrize IDs into
/// collection errors (`strict_parametrization_ids`, falling back to `strict`).
///
/// `pyproject.toml` pytest tables are evaluated; other config files and `PYTEST_ADDOPTS` count
/// when they mention `strict` at all, since collectors must not guess IDs pytest would reject.
pub fn strict_parametrization_ids_possible(root_path: &Path) -> bool {
    if std::env::var("PYTEST_ADDOPTS").is_ok_and(|opts| opts.contains("strict")) {
        return true;
    }
    let mentions_strict = |name: &str| {
        std::fs::read_to_string(root_path.join(name)).is_ok_and(|text| text.contains("strict"))
    };
    if [
        "pytest.toml",
        ".pytest.toml",
        "pytest.ini",
        "tox.ini",
        "setup.cfg",
    ]
    .into_iter()
    .any(mentions_strict)
    {
        return true;
    }
    let Ok(text) = std::fs::read_to_string(root_path.join("pyproject.toml")) else {
        return false;
    };
    let Ok(toml_value) = toml::from_str::<Value>(&text) else {
        return mentions_strict("pyproject.toml");
    };
    let pytest = toml_value.get("tool").and_then(|t| t.get("pytest"));
    let ini_options = pytest.and_then(|p| p.get("ini_options"));
    pytest.is_some_and(strict_ids_in_table) || ini_options.is_some_and(strict_ids_in_table)
}

fn strict_ids_in_table(table: &Value) -> bool {
    let enabled = |key: &str| {
        table.get(key).map(|value| match value {
            Value::Boolean(enabled) => *enabled,
            Value::String(text) => !matches!(
                text.to_ascii_lowercase().as_str(),
                "false" | "no" | "off" | "0"
            ),
            _ => true,
        })
    };
    let addopts_strict = table.get("addopts").is_some_and(|addopts| match addopts {
        Value::String(text) => text.contains("strict"),
        Value::Array(items) => items
            .iter()
            .any(|item| item.as_str().is_some_and(|text| text.contains("strict"))),
        _ => false,
    });
    addopts_strict
        || enabled("strict_parametrization_ids")
            .or_else(|| enabled("strict"))
            .unwrap_or(false)
}

/// Read `[tool.rtest] literal-values-functions` from `pyproject.toml`: qualified names of
/// functions that return the values of a `Literal` type (or a union of them) like `get_args`.
pub fn read_literal_values_functions(root_path: &Path) -> Vec<String> {
    read_function_list(root_path, "literal-values-functions")
}

/// Read `[tool.rtest] passthrough-functions` from `pyproject.toml`: qualified names of
/// functions that return their first argument unchanged, such as validators.
pub fn read_passthrough_functions(root_path: &Path) -> Vec<String> {
    read_function_list(root_path, "passthrough-functions")
}

fn read_function_list(root_path: &Path, key: &str) -> Vec<String> {
    std::fs::read_to_string(root_path.join("pyproject.toml"))
        .ok()
        .and_then(|text| toml::from_str::<Value>(&text).ok())
        .and_then(|toml_value| {
            toml_value
                .get("tool")?
                .get("rtest")?
                .get(key)?
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str().map(String::from))
                        .collect()
                })
        })
        .unwrap_or_default()
}

/// Read `[[tool.rtest.parametrize-wrappers]]` entries from `pyproject.toml`.
///
/// Entries without a string `decorator` and `argname` are ignored, so tests using them keep
/// collecting as undecorated functions exactly as before.
pub fn read_parametrize_wrappers(root_path: &Path) -> Vec<ParametrizeWrapper> {
    let Some(toml_value) = std::fs::read_to_string(root_path.join("pyproject.toml"))
        .ok()
        .and_then(|text| toml::from_str::<Value>(&text).ok())
    else {
        return Vec::new();
    };
    let entries = toml_value
        .get("tool")
        .and_then(|t| t.get("rtest"))
        .and_then(|r| r.get("parametrize-wrappers"))
        .and_then(Value::as_array);
    let string =
        |entry: &Value, key: &str| entry.get(key).and_then(Value::as_str).map(String::from);
    entries
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            Some(ParametrizeWrapper {
                decorator: string(entry, "decorator")?,
                argname: string(entry, "argname")?,
                record: string(entry, "record"),
                id_field: string(entry, "id-field"),
            })
        })
        .collect()
}

/// Read a TOML array of strings under the given `key`, yielding an iterator of
/// its string elements. Returns `None` when the key is absent or not an array.
fn read_string_array<'a>(
    ini_options: Option<&'a Value>,
    key: &str,
) -> Option<impl Iterator<Item = &'a str>> {
    ini_options
        .and_then(|i| i.get(key))
        .and_then(|t| t.as_array())
        .map(|array| array.iter().filter_map(|v| v.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_strict_parametrization_ids_possible() {
        let cases = [
            ("pyproject.toml", "[tool.mypy]\nstrict = true\n", false),
            (
                "pyproject.toml",
                "[tool.pytest.ini_options]\nstrict = \"true\"\n",
                true,
            ),
            (
                "pyproject.toml",
                "[tool.pytest]\nstrict = true\nstrict_parametrization_ids = false\n",
                false,
            ),
            (
                "pyproject.toml",
                "[tool.pytest.ini_options]\naddopts = \"-o strict=true\"\n",
                true,
            ),
            (
                "pytest.ini",
                "[pytest]\nstrict_parametrization_ids = true\n",
                true,
            ),
            ("pytest.ini", "[pytest]\naddopts = -ra\n", false),
        ];
        for (name, content, expected) in cases {
            let temp_dir = TempDir::new().expect("temp dir");
            fs::write(temp_dir.path().join(name), content).expect("write config");
            assert_eq!(
                strict_parametrization_ids_possible(temp_dir.path()),
                expected,
                "{content}"
            );
        }
    }

    #[test]
    fn test_read_pytest_config_no_file() {
        let temp_dir = TempDir::new().unwrap();
        let config = read_pytest_config(temp_dir.path());
        assert!(config.testpaths.is_empty());
        assert!(config.python_files.is_empty());
        assert!(config.python_classes.is_empty());
        assert!(config.python_functions.is_empty());
    }

    #[test]
    fn test_read_pytest_config_no_testpaths() {
        let temp_dir = TempDir::new().unwrap();
        let pyproject_path = temp_dir.path().join("pyproject.toml");

        let content = indoc! {r#"
            [tool.pytest.ini_options]
            filterwarnings = ["error"]
        "#};

        fs::write(&pyproject_path, content).unwrap();

        let config = read_pytest_config(temp_dir.path());
        assert!(config.testpaths.is_empty());
        assert!(config.python_files.is_empty());
        assert!(config.python_classes.is_empty());
        assert!(config.python_functions.is_empty());
    }

    #[test]
    fn reads_all_configured_paths_and_patterns() {
        let temp_dir = TempDir::new().expect("temporary project");
        fs::write(
            temp_dir.path().join("pyproject.toml"),
            indoc! {r#"
            [tool.pytest.ini_options]
            testpaths = ["tests", "test"]
            python_files = ["test_*.py", "*_test.py", "check_*.py"]
            python_classes = ["Test*", "Check*", "*Suite"]
            python_functions = ["test*", "check_*", "*_test"]
        "#},
        )
        .expect("write configuration");

        let config = read_pytest_config(temp_dir.path());
        assert_eq!(
            config.testpaths,
            [PathBuf::from("tests"), PathBuf::from("test")]
        );
        assert_eq!(
            config.python_files,
            ["test_*.py", "*_test.py", "check_*.py"]
        );
        assert_eq!(config.python_classes, ["Test*", "Check*", "*Suite"]);
        assert_eq!(config.python_functions, ["test*", "check_*", "*_test"]);
    }

    #[test]
    fn reads_parametrize_wrappers_and_skips_incomplete_entries() {
        let temp_dir = TempDir::new().expect("temporary project");
        fs::write(
            temp_dir.path().join("pyproject.toml"),
            indoc! {r#"
            [[tool.rtest.parametrize-wrappers]]
            decorator = "pkg.testing.parametrize"
            argname = "p"
            record = "pkg.testing.Params"
            id-field = "id"

            [[tool.rtest.parametrize-wrappers]]
            decorator = "pkg.other.parametrize"
            argname = "case"

            [[tool.rtest.parametrize-wrappers]]
            decorator = "pkg.incomplete.parametrize"
        "#},
        )
        .expect("write configuration");

        assert_eq!(
            read_parametrize_wrappers(temp_dir.path()),
            [
                ParametrizeWrapper {
                    decorator: "pkg.testing.parametrize".into(),
                    argname: "p".into(),
                    record: Some("pkg.testing.Params".into()),
                    id_field: Some("id".into()),
                },
                ParametrizeWrapper {
                    decorator: "pkg.other.parametrize".into(),
                    argname: "case".into(),
                    record: None,
                    id_field: None,
                },
            ]
        );
        assert!(read_parametrize_wrappers(&temp_dir.path().join("missing")).is_empty());
    }
}
