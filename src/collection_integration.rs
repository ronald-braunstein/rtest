//! Integration between Rust collection and pytest execution.

use crate::collection::error::{CollectionError, CollectionResult, CollectionWarning};
use crate::collection::nodes::Session;
use crate::collection::types::{CollectionBranch, CollectionNode};
use crate::python_discovery::{DiscoveryCache, DuplicateIds};
#[cfg(feature = "extension-module")]
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Holds errors and warnings encountered during collection
#[derive(Debug)]
pub struct CollectionErrors {
    pub errors: Vec<(String, CollectionError)>,
    pub warnings: Vec<CollectionWarning>,
}

#[cfg(feature = "extension-module")]
impl CollectionErrors {
    pub(crate) fn write_cannot_expand_report(&self, writer: &mut dyn Write) -> io::Result<()> {
        self.warnings
            .iter()
            .filter_map(|warning| warning.cannot_expand_nodeid.as_deref())
            .try_for_each(|nodeid| writeln!(writer, "rtest-cannot-expand: {nodeid}"))?;
        writer.flush()
    }
}

/// Run the Rust-based collection and return test node IDs
pub fn collect_tests_rust(
    rootpath: PathBuf,
    args: &[String],
    duplicate_ids: DuplicateIds,
) -> Result<(Vec<String>, CollectionErrors), CollectionError> {
    let mut session = Session::new(rootpath);
    session.config.duplicate_ids = duplicate_ids;
    session.config.parametrize_wrappers =
        crate::config::read_parametrize_wrappers(&session.rootpath);
    session.config.literal_values_functions =
        crate::config::read_literal_values_functions(&session.rootpath);
    session.config.passthrough_functions =
        crate::config::read_passthrough_functions(&session.rootpath);
    let mut collection_errors = CollectionErrors {
        errors: Vec::new(),
        warnings: Vec::new(),
    };

    let collectors = session.perform_collect(args)?;
    let mut items = Vec::new();
    for collector in collectors {
        expand_directories(&session, collector, &mut items);
    }

    let modules: Vec<&CollectionBranch> = items
        .iter()
        .filter_map(|item| match item {
            PlannedItem::Module(branch) => Some(branch),
            _ => None,
        })
        .collect();
    let mut module_results = collect_modules_in_parallel(&session, &modules).into_iter();

    let mut test_nodes = Vec::new();
    let mut discovery = DiscoveryCache::default();
    for item in items {
        match item {
            PlannedItem::Test(nodeid) => test_nodes.push(nodeid),
            PlannedItem::Warnings(warnings) => collection_errors.warnings.extend(warnings),
            PlannedItem::Error(nodeid, error) => collection_errors.errors.push((nodeid, error)),
            PlannedItem::Module(branch) => {
                let result = module_results
                    .next()
                    .expect("one result per planned module");
                record_branch_result(
                    &session,
                    &branch,
                    result,
                    &mut test_nodes,
                    &mut collection_errors,
                    &mut discovery,
                );
            }
        }
    }
    Ok((test_nodes, collection_errors))
}

type BranchResult = CollectionResult<(Vec<CollectionNode>, Vec<CollectionWarning>)>;

/// Collection work in output order; module branches are collected later, in parallel.
enum PlannedItem {
    Test(String),
    Module(CollectionBranch),
    Warnings(Vec<CollectionWarning>),
    Error(String, CollectionError),
}

/// Walk directories in order so module files can be parsed independently afterwards.
fn expand_directories(session: &Session, node: CollectionNode, items: &mut Vec<PlannedItem>) {
    match node {
        CollectionNode::Test(nodeid) => items.push(PlannedItem::Test(nodeid)),
        CollectionNode::Branch(branch @ CollectionBranch::Module(_)) => {
            items.push(PlannedItem::Module(branch));
        }
        CollectionNode::Branch(branch) => {
            match branch.collect(session, &mut DiscoveryCache::default()) {
                Ok((children, warnings)) => {
                    items.push(PlannedItem::Warnings(warnings));
                    for child in children {
                        expand_directories(session, child, items);
                    }
                }
                Err(error) => items.push(PlannedItem::Error(branch.nodeid(session), error)),
            }
        }
    }
}

/// Collect modules on worker threads, returning results in input order. Each worker keeps its
/// own discovery cache, so modules imported by many test files are parsed once per worker.
fn collect_modules_in_parallel(
    session: &Session,
    modules: &[&CollectionBranch],
) -> Vec<BranchResult> {
    let workers = collection_threads().min(modules.len()).max(1);
    let next = AtomicUsize::new(0);
    let mut results: Vec<(usize, BranchResult)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut discovery = DiscoveryCache::default();
                    let mut done = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(module) = modules.get(index) else {
                            break;
                        };
                        done.push((index, module.collect(session, &mut discovery)));
                    }
                    done
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("collection worker panicked"))
            .collect()
    });
    results.sort_unstable_by_key(|(index, _)| *index);
    results.into_iter().map(|(_, result)| result).collect()
}

fn collection_threads() -> usize {
    std::env::var("RTEST_COLLECT_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
        })
}

fn record_branch_result(
    session: &Session,
    branch: &CollectionBranch,
    result: BranchResult,
    test_nodes: &mut Vec<String>,
    collection_errors: &mut CollectionErrors,
    discovery: &mut DiscoveryCache,
) {
    match result {
        Ok((children, warnings)) => {
            collection_errors.warnings.extend(warnings);
            for child in children {
                match child {
                    CollectionNode::Test(nodeid) => test_nodes.push(nodeid),
                    CollectionNode::Branch(child) => {
                        let result = child.collect(session, discovery);
                        record_branch_result(
                            session,
                            &child,
                            result,
                            test_nodes,
                            collection_errors,
                            discovery,
                        );
                    }
                }
            }
        }
        Err(error) => collection_errors
            .errors
            .push((branch.nodeid(session), error)),
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
                CollectionError::FileNotFound(path) => {
                    println!(
                        "{RED}E   file or directory not found: {}{RESET}",
                        path.display()
                    );
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
        let summary_parts: Vec<String> = [
            (item_count, "collected ", "item"),
            (error_count, "", "error"),
            (warning_count, "", "warning"),
        ]
        .into_iter()
        .filter(|(count, _, _)| *count > 0)
        .map(|(count, prefix, label)| {
            format!(
                "{prefix}{count} {label}{}",
                if count == 1 { "" } else { "s" }
            )
        })
        .collect();

        println!("{}", summary_parts.join(" / "));

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
