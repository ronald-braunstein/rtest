//! Collection configuration.

use std::path::PathBuf;

use crate::python_discovery::{DuplicateIds, ParametrizeWrapper};

/// Configuration for collection
#[derive(Debug, Clone)]
pub struct CollectionConfig {
    pub ignore_patterns: Vec<String>,
    pub norecursedirs: Vec<String>,
    pub testpaths: Vec<PathBuf>,
    pub python_files: Vec<String>,
    pub python_classes: Vec<String>,
    pub python_functions: Vec<String>,
    pub duplicate_ids: DuplicateIds,
    pub parametrize_wrappers: Vec<ParametrizeWrapper>,
    pub literal_values_functions: Vec<String>,
    pub passthrough_functions: Vec<String>,
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self {
            ignore_patterns: vec![],
            norecursedirs: vec![
                "*.egg".into(),
                ".*".into(),
                "_darcs".into(),
                "build".into(),
                "CVS".into(),
                "dist".into(),
                "node_modules".into(),
                "venv".into(),
                "{arch}".into(),
            ],
            testpaths: vec![],
            python_files: vec!["test_*.py".into(), "*_test.py".into()],
            python_classes: vec!["Test*".into()],
            python_functions: vec!["test*".into()],
            duplicate_ids: DuplicateIds::default(),
            parametrize_wrappers: vec![],
            literal_values_functions: vec![],
            passthrough_functions: vec![],
        }
    }
}
