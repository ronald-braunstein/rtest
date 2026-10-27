"""Check hybrid collection parity and report lifecycle against pytest."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from test_helpers import create_test_project, extract_test_lines

from test_utils import RtestResult, run_rtest

ORACLE_PYTHON = os.environ.get("RTEST_PARITY_PYTHON", sys.executable)


def oracle_pytest_version() -> str:
    return subprocess.run(
        [ORACLE_PYTHON, "-c", "import pytest; print(pytest.__version__)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def pytest_collect(root: Path, files: list[str]) -> list[str]:
    """Use the installed pytest as the oracle, optionally testing another version."""
    result = subprocess.run(
        [
            ORACLE_PYTHON,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *files,
        ],
        cwd=root,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [line for line in result.stdout.splitlines() if line.startswith(tuple(files)) and "::" in line]


def collect(root: Path, *args: str) -> RtestResult:
    return run_rtest(["--collect-only", *args], cwd=str(root))


def normalize_nodeid(nodeid: str) -> str:
    path, separator, test = nodeid.partition("::")
    return path.replace("\\", "/") + separator + test


def test_hybrid_collection_matches_pytest() -> None:
    files = {
        "test_static.py": """
            import pytest
            @pytest.mark.parametrize("x", [1, 2])
            class TestClass:
                @pytest.mark.parametrize("y", ["a", "b"])
                def test_combined(self, x, y): pass
            def helper(): return 1
            @pytest.mark.parametrize("value", [helper(), helper()], ids=["one", "two"])
            def test_explicit_ids(value): pass
            @pytest.mark.parametrize("value", [pytest.param(helper(), id="caf\u00e9")])
            def test_escaped_id(value): pass
            from dataclasses import dataclass
            from datetime import date
            from decimal import Decimal
            @dataclass
            class Row:
                value: int
            @pytest.mark.parametrize("value", [Row(1), Row(2)])
            def test_objects(value): pass
            @pytest.mark.parametrize("amount,when", [(Decimal("1.5"), date(2024, 1, 1)), (2, date(2024, 2, 1))])
            def test_constructor_rows(amount, when): pass
            @pytest.mark.parametrize("value", [1, 1])
            def test_numeric_duplicates(value): pass
            @pytest.mark.parametrize("value", [True, False, False])
            def test_bool_duplicates(value): pass
            @pytest.mark.parametrize("x", [1, 2])
            @pytest.mark.parametrize("y", ["a", "a"])
            def test_stacked_duplicates(x, y): pass
        """,
        "test_values.py": """
            import enum
            import pytest
            class Color(enum.Enum):
                RED = "r"
                GREEN = "g"
                BLUE = "b"
            class Plain:
                pass
            class Holder:
                class Nested:
                    pass
                NAMES = ("n1", "n2")
            def helper(): return 1
            def make_plain() -> Plain: return Plain()
            MAPPING = {"b": 2, "a": 1}
            BASE = ["x", "y"]
            CASES = [pytest.param(1, id="one"), pytest.param(2, marks=pytest.mark.skip)]
            ROWS = [pytest.param(3, 4, id="row"), pytest.param(5, 6)]
            @pytest.mark.parametrize("value", [Plain, int, helper, Holder.Nested])
            def test_named(value): pass
            @pytest.mark.parametrize("key", MAPPING)
            def test_mapping(key): pass
            @pytest.mark.parametrize("key", sorted(MAPPING.keys()))
            def test_sorted_keys(key): pass
            @pytest.mark.parametrize("value", list(MAPPING.values()))
            def test_values(value): pass
            @pytest.mark.parametrize("value", BASE + ["z"] + list(Holder.NAMES[1:]))
            def test_concat_and_slice(value): pass
            @pytest.mark.parametrize("value", ["ab" * 2, 3 * "c"])
            def test_repeat(value): pass
            @pytest.mark.parametrize("value", range(1, 4))
            def test_range(value): pass
            COLORS = (Color.RED, Color.GREEN, Color.BLUE)
            PAIRS = [("a", 1), ("b", 2), ("c", 3)]
            @pytest.mark.parametrize("value", [name for name, _ in PAIRS if not name == "b"])
            def test_comprehension(value): pass
            @pytest.mark.parametrize("value", [c for c in COLORS if c in (Color.RED, Color.BLUE) and c != Color.RED])
            def test_enum_filter(value): pass
            @pytest.mark.parametrize("value", CASES)
            def test_params(value): pass
            @pytest.mark.parametrize("x,y", ROWS)
            def test_param_rows(x, y): pass
            @pytest.mark.parametrize("value", [{"a": 1}, (1, 2), make_plain()])
            def test_positional(value): pass
        """,
        "shapes.py": """
            from typing import Literal, Self
            Size = Literal["s", "m"]
            class Base:
                @classmethod
                def create(cls, code: str) -> "Base": return cls()
                @classmethod
                def build(cls) -> Self: return cls()
            class Mixin:
                pass
            class Error(Mixin, Base):
                pass
            class Shape:
                class Corner:
                    pass
                @staticmethod
                def make() -> "Shape": return Shape()
        """,
        "test_calls.py": """
            import enum
            import typing
            import uuid
            from contextlib import nullcontext as does_not_raise
            from dataclasses import dataclass, replace
            from datetime import timedelta, timezone
            from typing import Literal, TypeAlias, get_args
            import pytest
            import shapes
            from shapes import Error, Shape, Size
            class Color(enum.Enum):
                RED = "r"
            ONE = 1
            Mode = Literal["a", "b", "a"]
            Nested: TypeAlias = Literal[Mode, "c", Literal["d", "a"]]
            Mixed = Literal[Color.RED, ONE, True, None, -2]
            @dataclass
            class _Row:
                n: int
            class _Outer:
                class Inner:
                    pass
            @pytest.mark.parametrize("value", get_args(Mode))
            def test_literal(value): pass
            @pytest.mark.parametrize("value", typing.get_args(Nested))
            def test_nested_literal(value): pass
            @pytest.mark.parametrize("value", get_args(Mixed))
            def test_mixed_literal(value): pass
            @pytest.mark.parametrize("value", get_args(Size) + get_args(shapes.Size))
            def test_imported_literal(value): pass
            @pytest.mark.parametrize("value", [_Row(1), _Outer.Inner(), Shape.Corner(), Shape.make()])
            def test_classes(value): pass
            @pytest.mark.parametrize("value", [Error.create("x"), Error.build(), shapes.Base.build()])
            def test_inherited_methods(value): pass
            @pytest.mark.parametrize(
                "value",
                [
                    replace(_Row(1), n=2),
                    does_not_raise(),
                    pytest.raises(ValueError),
                    uuid.UUID(int=1),
                    timezone(timedelta(hours=1)),
                    object(),
                ],
            )
            def test_allowlisted_calls(value): pass
            NAME = "n"
            class Ids:
                PREFIX = "id"
                FLAG = False
            PAIRS = [("a", 1), ("b", 2)]
            LOOKUP = {"k": "v", Color.RED: "red"}
            @pytest.mark.parametrize(
                "value",
                [
                    f"x-{ONE}-{NAME}",
                    f"{Ids.PREFIX}:{Ids.FLAG}:{None}",
                    f"{NAME!s}" "-tail",
                    {1, 2},
                    {"a": 1},
                    PAIRS[-1][0],
                    PAIRS[1],
                    LOOKUP["k"],
                    LOOKUP[Color.RED],
                ],
            )
            def test_evaluated_expressions(value): pass
            class TestNested:
                class _Case:
                    pass
                @pytest.mark.parametrize("value", [_Case()])
                def test_class_scope(self, value): pass
        """,
        "test_duplicates.py": """
            import pytest
            @pytest.mark.parametrize("value", [1, 2, 3], ids=["a", "a", "a0"])
            def test_collision(value): pass
        """,
        "test_objects.py": """
            import pytest
            @pytest.mark.parametrize("x,y", [(str(1), str(2))])
            def test_helper_calls(x, y): pass
            def make_row(): return (1, 2)
            @pytest.mark.parametrize("x,y", [make_row()], ids=["row"])
            def test_opaque_row(x, y): pass
            @pytest.mark.parametrize("value", [1, 2], ids=[None, "named"])
            def test_partial_ids(value): pass
            @pytest.mark.parametrize("value", [1, 2], ids=str)
            def test_callable_ids(value): pass
            VALUES = [1, 2]
            @pytest.mark.parametrize("value", [*VALUES], ids=["one", "two"])
            def test_unpacked(value): pass
            @pytest.mark.parametrize("value", [*VALUES])
            def test_unpacked_without_ids(value): pass
            @pytest.mark.parametrize("value", [])
            def test_empty(value): pass
            import enum
            class Mode(enum.Enum):
                A = 1
                ALIAS = 1
            @pytest.mark.parametrize("value", [m for m in Mode])
            def test_enum_iteration(value): pass
        """,
        # Calls whose results pytest IDs by value or `__name__`, one file each.
        **{
            f"test_runtime_{name}.py": f"import pytest\n{source}\n"
            f'@pytest.mark.parametrize("value", {argvalues})\ndef test_item(value): pass\n'
            for name, source, argvalues in [
                (
                    "union_args",
                    'from typing import Literal, Union, get_args\nEither = Union[Literal["a"], Literal["b"]]',
                    "get_args(Either)",
                ),
                ("enum", "import enum\nclass _Mode(enum.Enum):\n    A = 'a'", "[_Mode('a')]"),
                ("own_name", "class _Named:\n    __name__ = 'named'", "[_Named()]"),
                ("str_subclass", "class _Text(str):\n    pass", "[_Text('t')]"),
                ("custom_new", "class _Built:\n    def __new__(cls): return 'built'", "[_Built()]"),
                (
                    "metaclass",
                    "class _Meta(type):\n    pass\nclass _WithMeta(metaclass=_Meta):\n    pass",
                    "[_WithMeta()]",
                ),
                ("shadowed_allowlist", "def replace(value): return 'replaced'", "[replace(1)]"),
                ("fstring_format", "ONE = 1", '[f"{ONE:03}"]'),
                ("fstring_repr", "NAME = 'n'", '[f"{NAME!r}"]'),
                ("conditional", "ONE = 1", '["a" if ONE else "b"]'),
                ("fstring_runtime", "import os", '[f"{os.sep}x"]'),
                ("fstring_call", "def name(): return 'n'", '[f"{name()}"]'),
                ("comparison", "ONE = 1", "[ONE == 1]"),
                (
                    "value_return",
                    "class _Base:\n    @classmethod\n    def code(cls) -> str: return 'c'\n"
                    "class _Child(_Base):\n    pass",
                    "[_Child.code()]",
                ),
            ]
        },
        "test_fixture.py": """
            from pytest import fixture as fx, mark
            @fx(params=[1, 2])
            async def value(request): return request.param
            @mark.parametrize("other", [3, 4])
            def test_combined(value, other): pass
        """,
        "base.py": """
            import pytest
            class Base:
                @pytest.fixture(params=[1, 2], autouse=True)
                def mode(self, request): pass
        """,
        "test_inherited.py": """
            from base import Base
            class TestChild(Base):
                def test_item(self): pass
        """,
        "fixtures/conftest.py": """
            import pytest as pt
            @pt.fixture(params=[1, 2], autouse=True)
            def value(request): return request.param
        """,
        "fixtures/nested/test_autouse.py": "def test_item(): pass\n",
        "generated/conftest.py": """
            def pytest_generate_tests(metafunc):
                metafunc.parametrize("value", [1, 2])
        """,
        "generated/test_hook.py": "def test_item(value): pass\n",
        "test_module_mark.py": """
            import pytest
            pytestmark = pytest.mark.parametrize("value", [1, 2])
            def test_item(value): pass
        """,
        "test_class_mark.py": """
            import pytest
            class TestMarked:
                pytestmark = pytest.mark.parametrize("value", [1, 2])
                def test_item(self, value): pass
        """,
    }
    with create_test_project({name: textwrap.dedent(source) for name, source in files.items()}) as root:
        test_files = [name for name in files if Path(name).name.startswith("test_")]
        report = root / "fallback.txt"
        version = oracle_pytest_version()
        args = ("--cannot-expand-report", str(report), "--pytest-version", version)
        result = collect(root, *args, *test_files)
        assert result.returncode == 0, result.output
        assert "rtest-cannot-expand:" not in result.stdout
        reported = {
            normalize_nodeid(line.removeprefix("rtest-cannot-expand: ")) for line in report.read_text().splitlines()
        }
        fallback = {nodeid.split("::", 1)[0] for nodeid in reported}
        # Pytest 7 emits colliding duplicate IDs, which only runtime collection reproduces.
        static_files = {"test_static.py", "test_values.py", "test_calls.py"} | (
            {"test_duplicates.py"} if int(version.split(".")[0]) >= 8 else set()
        )
        assert fallback == set(test_files) - static_files
        collected = [normalize_nodeid(line) for line in extract_test_lines(result.output_lines)]
        assert reported == {nodeid for nodeid in collected if nodeid.split("::", 1)[0] in fallback}
        static = [nodeid for nodeid in collected if nodeid not in reported]
        hybrid = static + pytest_collect(root, sorted(fallback))
        assert sorted(hybrid) == sorted(pytest_collect(root, test_files))
        # Compare multisets: pytest 7 itself can emit duplicates for colliding explicit IDs.


def test_parametrize_wrapper_matches_pytest() -> None:
    files = {
        "pyproject.toml": """
            [[tool.rtest.parametrize-wrappers]]
            decorator = "helpers.params.parametrize"
            argname = "p"
            record = "helpers.params.Params"
            id-field = "id"
        """,
        "helpers/__init__.py": "",
        "helpers/params.py": """
            from operator import attrgetter
            import pytest
            class Params:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            def parametrize(argdefs, argvalues, ids=None):
                def decorator(f):
                    if argvalues and isinstance(argvalues[0], Params):
                        records = argvalues
                    else:
                        records = [Params(**dict(zip(argdefs, row))) for row in argvalues]
                    idfunc = ids
                    if not idfunc and "id" in argdefs:
                        idfunc = attrgetter("id")
                    return pytest.mark.parametrize("p", records, ids=idfunc)(f)
                return decorator
        """,
        "test_static_wrapper.py": """
            import pytest
            from datetime import date
            from helpers.params import Params, parametrize
            from helpers import params
            def helper(): return object()
            @parametrize(["amount", "when"], [Params(amount=helper(), when=date(2024, 1, 1)), Params(amount=2)])
            def test_records_without_id(p): pass
            @parametrize(["id", "amount"], [Params(id="first", amount=helper()), Params(id="caf\u00e9", amount=2)])
            def test_record_ids(p): pass
            @parametrize(["id", "amount"], [Params(id=None, amount=1), Params(id=7, amount=2)])
            def test_scalar_and_missing_ids(p): pass
            @parametrize(["id", "amount"], [("row-a", helper()), ("row-b", 2)])
            def test_row_ids(p): pass
            @parametrize(["amount"], [(helper(),), (2,)])
            def test_rows_without_id(p): pass
            @parametrize("amount", [Params(id="ignored", amount=1)])
            def test_string_argdefs_without_id(p): pass
            @parametrize("valid_amount", [Params(id="sub", amount=1)])
            def test_string_argdefs_substring(p): pass
            @parametrize(["id"], [Params(id="dup"), Params(id="dup")])
            def test_duplicate_ids(p): pass
            @parametrize(["amount"], [Params(amount=1), Params(amount=2)], ids=["one", "two"])
            def test_explicit_ids(p): pass
            @parametrize(["id", "amount"], [Params(id="from-field", amount=1)], ids=[])
            def test_empty_ids_use_field(p): pass
            @pytest.mark.parametrize("x", [1, 2])
            @params.parametrize(["id"], [Params(id="a"), Params(id="b")])
            def test_stacked(x, p): pass
            class TestWrapped:
                @parametrize(["id", "amount"], [Params(id="m1", amount=1), Params(id="m2", amount=2)])
                def test_method(self, p): pass
        """,
        "test_runtime_wrapper.py": """
            from helpers.params import Params, parametrize
            def make_id(): return "made"
            CASES = [Params(id="a")]
            @parametrize(["id"], [Params(id=make_id())])
            def test_call_id(p): pass
            @parametrize(["id"], CASES)
            def test_named_cases(p): pass
            @parametrize(["amount"], [Params(amount=1)], ids=lambda p: "x")
            def test_callable_ids(p): pass
        """,
    }
    with create_test_project({name: textwrap.dedent(source) for name, source in files.items()}) as root:
        test_files = ["test_static_wrapper.py", "test_runtime_wrapper.py"]
        report = root / "fallback.txt"
        version = oracle_pytest_version()
        result = collect(root, "--cannot-expand-report", str(report), "--pytest-version", version, *test_files)
        assert result.returncode == 0, result.output
        reported = {
            normalize_nodeid(line.removeprefix("rtest-cannot-expand: ")) for line in report.read_text().splitlines()
        }
        assert {nodeid.split("::", 1)[0] for nodeid in reported} == {"test_runtime_wrapper.py"}
        assert len(reported) == 3, reported
        collected = [normalize_nodeid(line) for line in extract_test_lines(result.output_lines)]
        static = [nodeid for nodeid in collected if nodeid not in reported]
        fallback = [nodeid for nodeid in pytest_collect(root, test_files) if nodeid.split("[", 1)[0] in reported]
        assert sorted(static + fallback) == sorted(pytest_collect(root, test_files))


def test_cannot_expand_report_destinations_and_lifecycle() -> None:
    files = {
        "test_dynamic.py": (
            "import pytest\ndef values(): return [0, 1]\n"
            "@pytest.mark.parametrize('x', values())\ndef test_collected(x): pass\n"
        ),
        "test_static.py": "def test_static(): pass\n",
        "test_empty.py": "",
        "test_bad.py": "def test_broken(\n",
    }
    with create_test_project(files) as root:
        record = "rtest-cannot-expand: test_dynamic.py::test_collected"
        for destination in ("stdout", "stderr", "none"):
            result = collect(root, "--cannot-expand-report", destination, "test_dynamic.py")
            assert result.returncode == 0, result.output
            assert (record in result.stdout) == (destination == "stdout")
            assert (record in result.stderr) == (destination == "stderr")
        assert record in collect(root, "test_dynamic.py").stdout

        report = root / "fallback.txt"
        args = ("--cannot-expand-report", str(report))
        result = collect(root, *args, "test_dynamic.py")
        assert result.returncode == 0, result.output
        assert report.read_text().splitlines() == [record]
        for name, code in (("test_static.py", 0), ("test_empty.py", 0), ("missing.py", 4), ("test_bad.py", 1)):
            report.write_text("stale report\n")
            result = collect(root, *args, name)
            assert result.returncode == code, result.output
            assert report.read_text() == ""

        result = collect(root, *args, "test_dynamic.py", "test_bad.py")
        assert result.returncode == 1, "Partial collection must not be accepted as success"
        assert "test_dynamic.py::test_collected" in result.stdout
        assert "ERRORS" in result.output and "test_bad.py" in result.output
        assert report.read_text().splitlines() == [record]
        result = collect(root, "--cannot-expand-report", str(root / "missing" / "report.txt"), "test_dynamic.py")
        assert result.returncode != 0
        assert "cannot open cannot-expand report" in result.stderr


def test_duplicate_ids_follow_pytest_version() -> None:
    source = """
        import pytest
        @pytest.mark.parametrize("value", [1, 1])
        def test_numeric(value): pass
        @pytest.mark.parametrize("value", [1, 2, 3], ids=["a", "a", "a0"])
        def test_collision(value): pass
    """
    pytest8_ids = [
        "test_numeric[1_0]",
        "test_numeric[1_1]",
        "test_collision[a1]",
        "test_collision[a2]",
        "test_collision[a0]",
    ]
    cases = [
        (("--pytest-version", "7.3.1"), {}, ["test_numeric[10]", "test_numeric[11]"], {"test_collision"}),
        (("--pytest-version", "8"), {}, pytest8_ids, set()),
        (("--pytest-version", "9.1.1"), {}, pytest8_ids, set()),
        ((), {}, pytest8_ids, set()),
        (
            ("--pytest-version", "9"),
            {"pytest.ini": "[pytest]\nstrict = true\n"},
            [],
            {"test_numeric", "test_collision"},
        ),
    ]
    for args, config, expected_ids, fallback_tests in cases:
        with create_test_project({"test_dups.py": textwrap.dedent(source), **config}) as root:
            report = root / "fallback.txt"
            result = collect(root, "--cannot-expand-report", str(report), *args, "test_dups.py")
            assert result.returncode == 0, result.output
            reported = {line.split("::", 1)[1] for line in report.read_text().splitlines()}
            assert reported == fallback_tests, args
            collected = [line.split("::", 1)[1] for line in extract_test_lines(result.output_lines)]
            assert [test for test in collected if test not in reported] == expected_ids, args
