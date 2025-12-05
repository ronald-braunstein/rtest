//! Integration between Rust collection and pytest execution.

use crate::collection::error::{CollectionError, CollectionOutcome, CollectionWarning};
use crate::collection::nodes::{collect_one_node, Session};
use crate::collection::types::Collector;
use std::collections::HashSet;
use std::path::PathBuf;
use std::rc::Rc;

/// Holds errors and warnings encountered during collection
#[derive(Debug)]
pub struct CollectionErrors {
    pub errors: Vec<(String, CollectionError)>,
    pub warnings: Vec<CollectionWarning>,
}

/// Results from collection with parametrize tracking
#[derive(Debug)]
pub struct CollectionResultsWithSplit {
    pub test_nodes: Vec<String>,
    pub errors: CollectionErrors,
    /// Files without parametrized tests (relative paths)
    pub simple_files: Vec<String>,
    /// Files with parametrized tests (relative paths)
    pub param_files: Vec<String>,
    /// Test nodes from parametrized files only
    pub param_file_tests: Vec<String>,
}

/// Check if source code contains parametrize decorators
fn source_has_parametrize(source: &str) -> bool {
    // Check for common parametrize patterns
    source.contains("@pytest.mark.parametrize")
        || source.contains("@mark.parametrize")
        || source.contains("@parametrize(")
}

/// Run the Rust-based collection and return test node IDs
pub fn collect_tests_rust(
    rootpath: PathBuf,
    args: &[String],
) -> Result<(Vec<String>, CollectionErrors), CollectionError> {
    let session = Rc::new(Session::new(rootpath));
    let mut collection_errors = CollectionErrors {
        errors: Vec::new(),
        warnings: Vec::new(),
    };

    match session.perform_collect(args) {
        Ok(collectors) => {
            let mut test_nodes = Vec::new();

            for collector in collectors {
                collect_items_recursive(
                    collector.as_ref(),
                    &mut test_nodes,
                    &mut collection_errors,
                );
            }

            Ok((test_nodes, collection_errors))
        }
        Err(e) => Err(e),
    }
}

/// Run collection and split results by parametrized vs simple files
pub fn collect_tests_rust_with_split(
    rootpath: PathBuf,
    args: &[String],
) -> Result<CollectionResultsWithSplit, CollectionError> {
    let session = Rc::new(Session::new(rootpath.clone()));
    let mut collection_errors = CollectionErrors {
        errors: Vec::new(),
        warnings: Vec::new(),
    };

    match session.perform_collect(args) {
        Ok(collectors) => {
            let mut test_nodes = Vec::new();
            let mut param_files: HashSet<String> = HashSet::new();
            let mut simple_files: HashSet<String> = HashSet::new();
            let mut param_file_tests = Vec::new();

            for collector in collectors {
                collect_items_recursive_with_split(
                    collector.as_ref(),
                    &mut test_nodes,
                    &mut collection_errors,
                    &rootpath,
                    &mut param_files,
                    &mut simple_files,
                    &mut param_file_tests,
                );
            }

            // Convert sets to sorted vecs
            let mut simple_files_vec: Vec<String> = simple_files.into_iter().collect();
            let mut param_files_vec: Vec<String> = param_files.into_iter().collect();
            simple_files_vec.sort();
            param_files_vec.sort();

            Ok(CollectionResultsWithSplit {
                test_nodes,
                errors: collection_errors,
                simple_files: simple_files_vec,
                param_files: param_files_vec,
                param_file_tests,
            })
        }
        Err(e) => Err(e),
    }
}

/// Recursively collect all test items
fn collect_items_recursive(
    collector: &dyn Collector,
    test_nodes: &mut Vec<String>,
    collection_errors: &mut CollectionErrors,
) {
    if collector.is_item() {
        test_nodes.push(collector.nodeid().into());
    } else {
        let report = collect_one_node(collector);
        match report.outcome {
            CollectionOutcome::Passed => {
                for child in report.result {
                    collect_items_recursive(child.as_ref(), test_nodes, collection_errors);
                }
            }
            CollectionOutcome::Failed => {
                if let Some(error) = report.error_type {
                    collection_errors
                        .errors
                        .push((report.nodeid.clone(), error));
                }
            }
            _ => {}
        }
    }
}

/// Recursively collect test items while tracking parametrized vs simple files
fn collect_items_recursive_with_split(
    collector: &dyn Collector,
    test_nodes: &mut Vec<String>,
    collection_errors: &mut CollectionErrors,
    rootpath: &PathBuf,
    param_files: &mut HashSet<String>,
    simple_files: &mut HashSet<String>,
    param_file_tests: &mut Vec<String>,
) {
    if collector.is_item() {
        let nodeid = collector.nodeid();
        test_nodes.push(nodeid.into());

        // Check if this test's file is in param_files
        if let Some(file_path) = nodeid.split("::").next() {
            if param_files.contains(file_path) {
                param_file_tests.push(nodeid.into());
            }
        }
    } else {
        // Check if this is a Python file (Module) and classify it
        let path = collector.path();
        if path.extension().map_or(false, |ext| ext == "py") && path.is_file() {
            let relative_path = path
                .strip_prefix(rootpath)
                .unwrap_or(path)
                .to_string_lossy()
                .into_owned();

            // Read file and check for parametrize
            if let Ok(source) = std::fs::read_to_string(path) {
                if source_has_parametrize(&source) {
                    param_files.insert(relative_path);
                } else {
                    simple_files.insert(relative_path);
                }
            }
        }

        let report = collect_one_node(collector);
        match report.outcome {
            CollectionOutcome::Passed => {
                for child in report.result {
                    collect_items_recursive_with_split(
                        child.as_ref(),
                        test_nodes,
                        collection_errors,
                        rootpath,
                        param_files,
                        simple_files,
                        param_file_tests,
                    );
                }
            }
            CollectionOutcome::Failed => {
                if let Some(error) = report.error_type {
                    collection_errors
                        .errors
                        .push((report.nodeid.clone(), error));
                }
            }
            _ => {}
        }
    }
}

/// Display collection results in a format similar to pytest
pub fn display_collection_results(test_nodes: &[String], errors: &CollectionErrors) {
    // ANSI color codes
    const RED: &str = "\x1b[31m";
    const BOLD_RED: &str = "\x1b[1;31m";
    const YELLOW: &str = "\x1b[33m";
    const RESET: &str = "\x1b[0m";

    if !errors.errors.is_empty() {
        println!(
            "===================================== ERRORS ======================================"
        );
        for (nodeid, error) in &errors.errors {
            println!("{BOLD_RED}_ ERROR collecting {nodeid} _{RESET}");
            match error {
                CollectionError::ParseError(msg) => {
                    println!("{RED}E   {msg}{RESET}");
                }
                CollectionError::ImportError(msg) => {
                    println!("{RED}E   ImportError: {msg}{RESET}");
                }
                CollectionError::IoError(e) => {
                    println!("{RED}E   IO Error: {e}{RESET}");
                }
                CollectionError::SkipError(msg) => {
                    println!("{RED}E   Skipped: {msg}{RESET}");
                }
            }
        }
        println!(
            "!!!!!!!!!!!!!!!!!!!!! Warning: {} errors during collection !!!!!!!!!!!!!!!!!!!!!",
            errors.errors.len()
        );
    }

    let item_count = test_nodes.len();
    let error_count = errors.errors.len();
    let warning_count = errors.warnings.len();

    if item_count == 0 && error_count == 0 {
        println!("No tests collected.");
    } else {
        let mut summary_parts = Vec::new();

        if item_count > 0 {
            summary_parts.push(format!(
                "collected {} item{}",
                item_count,
                if item_count == 1 { "" } else { "s" }
            ));
        }

        if error_count > 0 {
            summary_parts.push(format!(
                "{} error{}",
                error_count,
                if error_count == 1 { "" } else { "s" }
            ));
        }

        if warning_count > 0 {
            summary_parts.push(format!(
                "{} warning{}",
                warning_count,
                if warning_count == 1 { "" } else { "s" }
            ));
        }

        if !summary_parts.is_empty() {
            println!("{}", summary_parts.join(" / "));
        }

        if !test_nodes.is_empty() {
            println!();
            for node in test_nodes {
                println!("  {node}");
            }
        }
    }

    // Display warnings after the test list
    if !errors.warnings.is_empty() {
        println!();
        println!(
            "=============================== warnings summary ==============================="
        );
        for warning in &errors.warnings {
            println!("{YELLOW}{warning}{RESET}");
        }
        println!("-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html");
    }
}
