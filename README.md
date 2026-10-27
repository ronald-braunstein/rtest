# rtest

[![PyPI version](https://badge.fury.io/py/rtest.svg)](https://badge.fury.io/py/rtest)
[![Python](https://img.shields.io/pypi/pyversions/rtest.svg)](https://pypi.org/project/rtest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python test runner built with Rust, featuring high-performance test collection and parallel test execution.

> **⚠️ Development Status**: This project is in early development (v0.0.x). Expect bugs, breaking changes, and evolving
> features as we work toward stability.

## Performance

`rtest` discovers tests through static AST analysis without importing test modules. CI measures collection on fixed
Click, Flask, and Pydantic source revisions using a release build, 3 warmups, and 20 measured runs.

The [benchmark job](https://github.com/hughhan1/rtest/actions/workflows/ci.yml) publishes timings, collected test IDs,
and raw Hyperfine reports. See [the benchmark guide](scripts/benchmark/README.md) to reproduce a run.

Static discovery and pytest's import-based collection perform different work and can discover different tests.
The benchmark reports collection time only; it does not claim equivalent test coverage, execution speedups,
or performance regressions across different CI machines.

## Quick Start

### Installation

```bash
pip install rtest
```

_Requires Python 3.10+_

Building from source requires Rust 1.97+ and the initialized Ruff submodule (`git submodule update --init`).

### Basic Usage

```bash
# Collect tests (fast AST-based collection)
rtest --collect-only

# Run tests with native runner
rtest --runner native

# Run tests in parallel (4 workers)
rtest --runner native -n 4
```

### Native Runner

The native runner (`--runner native`) executes tests using rtest's own decorators:

```python
import rtest

@rtest.mark.cases("value", [1, 2, 3])
def test_example(value):
    assert value > 0

@rtest.mark.skip(reason="Not implemented yet")
def test_skipped():
    pass
```

The native runner respects `python_files`, `python_classes`, and `python_functions` patterns from your
`pyproject.toml` under `[tool.pytest.ini_options]`.

For compatibility, `@pytest.mark.parametrize` and `@pytest.mark.skip` decorators are also supported.

## Roadmap

- **Fixtures** - `@rtest.fixture` with function/class/module scopes and dependency resolution
- **conftest.py support** - Fixture discovery across directory hierarchy
- **Distribution modes** - Group tests by module/class, optimized scheduling algorithms
- **Built-in assertions** - `rtest.raises()` and other assertion helpers
- **Additional markers** - `@rtest.mark.xfail`, `@rtest.mark.skipif`
- **Test selection** - `-k` expression filtering, `--last-failed`, `--failed-first`
- **Better error formatting** - Rich diffs, colorized output, traceback filtering
- **Async test support** - Native `async def test_*()` handling
- **Watch mode** - Re-run tests automatically on file changes

## Known Limitations

### Parametrized Test Discovery

`rtest` supports limited **static expansion** of parametrized tests, including simple uses of:

- **Literal values**: numbers, strings, booleans, None, lists/tuples of literals
- **Module-level constants**: `DATA = [1, 2, 3]` then `@cases("x", DATA)`
- **Class constants**: `Config.MAX_SIZE`
- **Enum members**: `Color.RED`
- **Nested class constants**: `Outer.Inner.VALUE`

```python
from enum import Enum
import rtest

class Color(Enum):
    RED = 1
    GREEN = 2

DATA = [1, 2, 3]

@rtest.mark.cases("value", DATA)  # Resolves to [1, 2, 3]
def test_module_constant(value):
    assert value > 0

@rtest.mark.cases("color", [Color.RED, Color.GREEN])  # Resolves enum members
def test_enum_values(color):
    assert color.value in [1, 2]
```

When static expansion fails, `rtest` emits a warning and falls back to the base test name:

```python
@pytest.mark.parametrize("value", get_data())  # Function call
def test_dynamic(value):
    assert value > 0

@pytest.mark.parametrize("value", [x for x in range(3)])  # Comprehension
def test_comprehension(value):
    assert value > 0
```

```plaintext
warning: Cannot statically expand test cases for 'test.py::test_dynamic': argvalues contains function call 'get_data'
warning: Cannot statically expand test cases for 'test.py::test_comprehension': argvalues contains a comprehension
```

In these cases, test execution is still functionally equivalent - pytest automatically runs all parametrized variants
when given the base function name.

#### Project-specific parametrize decorators

Projects that wrap `pytest.mark.parametrize` in their own decorator can describe the wrapper in `pyproject.toml` so
`rtest` expands it statically:

```toml
[[tool.rtest.parametrize-wrappers]]
decorator = "myproject.testing.parametrize"    # fully qualified decorator name
argname = "p"                                  # the single argname the wrapper passes to pytest
record = "myproject.testing.Params"            # optional: row type accepted in place of tuples
id-field = "id"                                # optional: field whose value becomes the test ID
```

The wrapper is called as `@parametrize(argnames, argvalues, ids=None)` and passes every case to the test as one
`argname` argument: argvalues that are already `record` calls are used as-is, and other rows are zipped with
`argnames`. IDs come from a static `ids=` list, then from `id-field` when the argnames include it, and otherwise
default to `{argname}{index}`. Anything that cannot be resolved statically falls back to pytest, as above.

#### Literal-values helpers

Functions that return the values of a `Literal` type, like `typing.get_args`, can be listed by qualified name.
Calls with one argument then expand from the type, with unions (`A | B` or `Union[A, B]`) flattened left to right:

```toml
[tool.rtest]
literal-values-functions = ["myproject.literals.LiteralTools.values"]
```

Both `LiteralTools.values(Color)` and `LiteralTools[Color].values(Color)` resolve. Each `Literal` keeps its own
de-duplication and union members that are equal as types are dropped, matching `typing.get_args`; a union member
that is not a `Literal` of strings falls back to pytest.

#### Pass-through helpers

Functions that validate their first argument and return it unchanged can be listed by qualified name, so a call
resolves to that argument:

```toml
[tool.rtest]
passthrough-functions = ["myproject.validation.check_values", "graphql_lib.gql"]
```

#### Functions, comprehensions and types

- A function or `@staticmethod`/`@classmethod` without parameters whose body is a single `return` (after an
  optional docstring) resolves to the returned expression, evaluated in its defining module.
- Comprehensions, `set`/`frozenset` literals and calls, `sorted(...)`, `enumerate(...)`, `zip(...)` and `{**a, ...}`
  evaluate over static values. A set must be sorted before it is used as argvalues, since its iteration order is
  only known at runtime.
- Classes reached through a module (`import pkg.mod`, `import pkg.mod as m`, `from pkg import mod`) ID by
  `__name__`, as do subscripted types: `list[int]` is `list`, `Optional[int]` is `Optional`, a `Literal[...]` is
  `Literal`, and a union of distinct members is `Union` (Python drops duplicate members first, so
  `Union[int, int]` is `int`). `int | None` has no `__name__` and is IDed positionally.
- `float("nan")` and other plain numeric strings, `http.HTTPStatus` members and `pytest.ExitCode` members (both
  `IntEnum`s, IDed by their integer value) resolve to their values.

#### Records and IDs

Instances of plain record classes are counted statically and IDed positionally, and their attributes can feed IDs
(`ids=lambda case: case.name` or `ids=[case.name for case in CASES]`). A record class is a `@dataclass` without
`__init__` or `__post_init__` whose only base is another such dataclass (`Generic[T]` aside), a `NamedTuple`,
or a class whose `__init__` only assigns parameters to
`self` attributes. Names reached through `from module import *` follow a static `__all__` (or the module's public
names when it has none), and nested classes resolve members through dotted bases such as
`class Status(Enums.Status): pass`.

### Path Separator Handling

`rtest` uses platform-specific path separators in test nodeids, while `pytest` normalizes all paths to use forward
slashes (`/`) regardless of platform. For example:

**On Windows:**

- pytest shows: `tests/unit/test_example.py::test_function`
- rtest shows: `tests\unit\test_example.py::test_function`

**On Unix/macOS:**

- Both show: `tests/unit/test_example.py::test_function`

This difference is intentional as `rtest` preserves the native path format of the operating system.

### Native Runner Scope

The native runner (`--runner native`) does not currently support pytest fixtures or `conftest.py` discovery. Projects
that rely on fixtures should use `--runner pytest` instead. Fixture support is tracked in
[#105](https://github.com/hughhan1/rtest/issues/105).

### pytest-xdist Serialization with Decimal Objects

When using `--runner pytest -n N` (parallel pytest execution), projects that use `Decimal` objects in parametrized test
data may fail with a serialization error:

```plaintext
execnet.gateway_base.DumpError: can't serialize <class 'decimal.Decimal'>
```

This is a limitation of pytest-xdist's inter-process communication, which cannot serialize certain Python types. Tests
pass when run without parallelization (`--runner pytest` without `-n`). See
[#116](https://github.com/hughhan1/rtest/issues/116) for details.

### Collection Failures on Complex Codebases

Some projects with complex runtime behavior (e.g., extensive use of plugins, dynamic imports, or metaclass-heavy
patterns) may fail during `rtest` collection despite working correctly with `pytest`. For example, the
[pydantic](https://github.com/pydantic/pydantic) repository encounters errors when collected by `rtest`. See
[#55](https://github.com/hughhan1/rtest/issues/55) for details.

## Contributing

We welcome contributions! See [Contributing Guide](CONTRIBUTING.rst).

## License

MIT - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This project takes inspiration from [Astral](https://astral.sh) and leverages crates from [`ruff`].
