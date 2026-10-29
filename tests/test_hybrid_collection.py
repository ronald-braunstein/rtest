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
            UNPACKED = [1, 2]
            @pytest.mark.parametrize("value", [*UNPACKED], ids=["one", "two"])
            def test_unpacked(value): pass
            @pytest.mark.parametrize("value", [*UNPACKED])
            def test_unpacked_without_ids(value): pass
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
            @pytest.mark.parametrize("value", [0.5 * 1024 * 1024, 1 / 2, 1 / 3, 0.1 + 0.2, 2 * 1.5])
            def test_float_arithmetic(value): pass
            @pytest.mark.parametrize(
                "value",
                [2**63 - 1, 2**64 - 1, -1 * 2**100, 1 << 70, 0x1_0000_0000_0000_0000, 18_446_744_073_709_551_616 * 2],
            )
            def test_int_arithmetic_past_64_bits(value): pass
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
            def make_row(x=1): return (x, 2)
            @pytest.mark.parametrize("x,y", [make_row()], ids=["row"])
            def test_opaque_row(x, y): pass
            @pytest.mark.parametrize("value", [1, 2], ids=[None, "named"])
            def test_partial_ids(value): pass
            @pytest.mark.parametrize("value", [1, 2], ids=str)
            def test_callable_ids(value): pass
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
                ("fstring_call", "def name(n='n'): return n", '[f"{name()}"]'),
                ("comparison", "ONE = 1", "[ONE == 1]"),
                ("int_past_128_bits", "", "[2**127]"),
                ("int_literal_past_128_bits", "", "[340_282_366_920_938_463_463_374_607_431_768_211_456]"),
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
            def make_id(made="made"): return made
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


def test_dict_items_match_pytest() -> None:
    files = {
        "test_static_items.py": """
            import enum
            import pytest
            class Color(enum.Enum):
                RED = "r"
                GREEN = "g"
            class Holder:
                LABELS = {"x": "ex", "y": "why"}
            STRINGS = {"b": "bee", "a": "ay"}
            LISTS = {"one": [1], "two": [2, 2]}
            COLORS = {"red": Color.RED, "green": Color.GREEN}
            NUMBERS = {3: "three", 1: "one", 2: "two"}
            @pytest.mark.parametrize("pair", STRINGS.items())
            def test_pairs(pair): pass
            @pytest.mark.parametrize("key,value", STRINGS.items())
            def test_string_values(key, value): pass
            @pytest.mark.parametrize("key,value", LISTS.items())
            def test_list_values(key, value): pass
            @pytest.mark.parametrize("key,value", COLORS.items())
            def test_enum_values(key, value): pass
            @pytest.mark.parametrize("key,value", sorted(STRINGS.items()))
            def test_sorted_string_keys(key, value): pass
            @pytest.mark.parametrize("key,value", sorted(NUMBERS.items()))
            def test_sorted_int_keys(key, value): pass
            @pytest.mark.parametrize("key,value", list(Holder.LABELS.items()))
            def test_class_constant(key, value): pass
            class TestItems:
                @pytest.mark.parametrize("key,value", Holder.LABELS.items())
                def test_method(self, key, value): pass
        """,
        "test_runtime_items.py": """
            import pytest
            def helper(n=1): return n
            def make_mapping(a=1): return {"a": a}
            RUNTIME = {"a": helper(), "b": 2}
            @pytest.mark.parametrize("key,value", RUNTIME.items())
            def test_runtime_value(key, value): pass
            @pytest.mark.parametrize("key,value", make_mapping().items())
            def test_runtime_mapping(key, value): pass
        """,
    }
    with create_test_project({name: textwrap.dedent(source) for name, source in files.items()}) as root:
        test_files = ["test_static_items.py", "test_runtime_items.py"]
        report = root / "fallback.txt"
        version = oracle_pytest_version()
        result = collect(root, "--cannot-expand-report", str(report), "--pytest-version", version, *test_files)
        assert result.returncode == 0, result.output
        reported = {
            normalize_nodeid(line.removeprefix("rtest-cannot-expand: ")) for line in report.read_text().splitlines()
        }
        assert reported == {
            "test_runtime_items.py::test_runtime_value",
            "test_runtime_items.py::test_runtime_mapping",
        }
        collected = [normalize_nodeid(line) for line in extract_test_lines(result.output_lines)]
        static = [nodeid for nodeid in collected if nodeid not in reported]
        fallback = [nodeid for nodeid in pytest_collect(root, test_files) if nodeid.split("[", 1)[0] in reported]
        assert sorted(static + fallback) == sorted(pytest_collect(root, test_files))


def assert_hybrid_matches_pytest(root: Path, test_files: list[str], expected_fallbacks: set[str]) -> None:
    report = root / "fallback.txt"
    version = oracle_pytest_version()
    result = collect(root, "--cannot-expand-report", str(report), "--pytest-version", version, *test_files)
    assert result.returncode == 0, result.output
    reported = {normalize_nodeid(line.removeprefix("rtest-cannot-expand: ")) for line in report.read_text().splitlines()}
    assert reported == expected_fallbacks
    collected = [normalize_nodeid(line) for line in extract_test_lines(result.output_lines)]
    static = [nodeid for nodeid in collected if nodeid not in reported]
    expected = pytest_collect(root, test_files)
    fallback = [nodeid for nodeid in expected if nodeid.split("[", 1)[0] in reported]
    assert sorted(static + fallback) == sorted(expected)


def test_literal_values_functions_match_pytest() -> None:
    files = {
        "pyproject.toml": """
            [tool.rtest]
            literal-values-functions = ["literals.LiteralTools.values", "literals.literal_values"]
        """,
        "literals.py": """
            import types
            import typing
            from typing import Generic, Literal, TypeVar, Union, get_args, get_origin
            T = TypeVar("T")
            class LiteralTools(Generic[T]):
                @classmethod
                def values(cls, literal_type):
                    if get_origin(literal_type) in (Union, types.UnionType):
                        return tuple(v for member in get_args(literal_type) for v in cls.values(member))
                    values = get_args(literal_type)
                    if not all(isinstance(v, str) for v in values):
                        raise TypeError(literal_type)
                    return values
            def literal_values(literal_type):
                return LiteralTools.values(literal_type)
            Env = Literal["prod", "dev", "prod"]
            Extra = Literal["qa", "dev"]
        """,
        "test_static_literals.py": """
            import typing
            from typing import Literal, Optional, Union
            import pytest
            from literals import Env, Extra, LiteralTools, literal_values
            Same = Literal["dev", "prod"]
            Combined = Env | Extra
            NAMES = LiteralTools.values(Combined)
            @pytest.mark.parametrize("env", LiteralTools[Env].values(Env))
            def test_subscripted_owner(env): pass
            @pytest.mark.parametrize("env", LiteralTools.values(Env | Extra))
            def test_pep604_union(env): pass
            @pytest.mark.parametrize("env", LiteralTools.values(Union[Extra, Env, Same]))
            def test_union_drops_equal_members(env): pass
            @pytest.mark.parametrize("env", literal_values(typing.Union[Env, Literal["x"]]))
            def test_function(env): pass
            @pytest.mark.parametrize("env", NAMES)
            def test_constant(env): pass
            @pytest.mark.parametrize("env", [e for e in LiteralTools.values(Combined) if e != "qa"])
            def test_comprehension(env): pass
        """,
        "test_runtime_literals.py": """
            from typing import Literal
            import pytest
            from literals import LiteralTools
            Mixed = Literal["a"] | int
            def values(literal_type): return ("other",)
            @pytest.mark.parametrize("env", LiteralTools.values(Mixed))
            def test_non_literal_member(env): pass
            @pytest.mark.parametrize("env", values(Literal["a"]))
            def test_unconfigured_function(env): pass
        """,
    }
    with create_test_project({name: textwrap.dedent(source) for name, source in files.items()}) as root:
        assert_hybrid_matches_pytest(
            root,
            ["test_static_literals.py", "test_runtime_literals.py"],
            {
                "test_runtime_literals.py::test_non_literal_member",
                "test_runtime_literals.py::test_unconfigured_function",
            },
        )


def test_returned_values_and_sets_match_pytest() -> None:
    files = {
        "pyproject.toml": """
            [tool.rtest]
            passthrough-functions = ["values.Checks.validate"]
        """,
        "values.py": """
            from dataclasses import dataclass
            from typing import Literal, get_args
            Env = Literal["local", "dev", "stage", "prod"]
            class EnvValues:
                LOCAL = "local"
                DEV = "dev"
            class Status:
                OPEN = "open"
                DONE = "done"
                @staticmethod
                def all_statuses():
                    "Every status."
                    return {Status.OPEN, Status.DONE, Status.OPEN}
                @staticmethod
                def open_statuses():
                    return [Status.OPEN]
            class Checks:
                @classmethod
                def validate(cls, values, literal_type):
                    for value in values:
                        if value not in get_args(literal_type):
                            raise ValueError(value)
                    return list(values)
            @dataclass(frozen=True)
            class Config:
                enabled: bool
            CONFIG = {"b": Config(True), "a": Config(False), "c": Config(True)}
            def remote_envs():
                return (env for env in get_args(Env) if env not in {EnvValues.LOCAL, EnvValues.DEV})
            def enabled_names():
                return [name for name, config in CONFIG.items() if config.enabled]
            def validated():
                return Checks.validate(["dev", "prod"], Env)
        """,
        "test_static_returns.py": """
            from typing import get_args
            import pytest
            from values import Env, Status, enabled_names, remote_envs, validated
            LOCAL_ONLY = [env for env in get_args(Env) if env == "local"]
            class TestClassBody:
                def _cases():
                    return [1, 2]
                @pytest.mark.parametrize("value", _cases())
                def test_class_function(self, value): pass
            @pytest.mark.parametrize("env", remote_envs())
            def test_generator_with_set_filter(env): pass
            @pytest.mark.parametrize("status", sorted(Status.all_statuses()))
            def test_sorted_set(status): pass
            @pytest.mark.parametrize("status", Status.open_statuses())
            def test_static_method(status): pass
            @pytest.mark.parametrize("name", enabled_names())
            def test_record_filter(name): pass
            @pytest.mark.parametrize("name", sorted({n for n in ("b", "a", "b")}))
            def test_set_comprehension(name): pass
            @pytest.mark.parametrize("env", validated())
            def test_passthrough(env): pass
            @pytest.mark.parametrize("env", LOCAL_ONLY)
            def test_module_comprehension(env): pass
            @pytest.mark.parametrize("value", [{1, 2}, frozenset()])
            def test_set_values(value): pass
        """,
        "test_runtime_returns.py": """
            import pytest
            from values import Status
            def twice(): return [1]
            def twice(): return [2]
            def with_parameter(n=2): return list(range(n))
            def two_statements():
                values = [1]
                return values
            @pytest.mark.parametrize("value", twice())
            def test_redefined(value): pass
            @pytest.mark.parametrize("value", with_parameter())
            def test_parameter(value): pass
            @pytest.mark.parametrize("value", two_statements())
            def test_statements(value): pass
            @pytest.mark.parametrize("status", Status.all_statuses())
            def test_unsorted_set(status): pass
            @pytest.mark.parametrize("value", sorted({1, True}))
            def test_mixed_set(value): pass
        """,
    }
    with create_test_project({name: textwrap.dedent(source) for name, source in files.items()}) as root:
        assert_hybrid_matches_pytest(
            root,
            ["test_static_returns.py", "test_runtime_returns.py"],
            {
                "test_runtime_returns.py::test_redefined",
                "test_runtime_returns.py::test_parameter",
                "test_runtime_returns.py::test_statements",
                "test_runtime_returns.py::test_unsorted_set",
                "test_runtime_returns.py::test_mixed_set",
            },
        )


def test_types_records_and_module_attributes_match_pytest() -> None:
    files = {
        "pkg/__init__.py": "",
        "pkg/types.py": """
            from dataclasses import dataclass
            from typing import Generic, Literal, TypeVar
            T = TypeVar("T")
            Color = Literal["red", "blue"]
            Reordered = Literal["blue", "red"]
            Spooky = Literal["orange", "black"]
            Palette = Color | Spooky | Literal["white"]
            class Summary: pass
            class Detail: pass
            @dataclass(frozen=True)
            class Definition(Generic[T]):
                name: str
            @dataclass(frozen=True)
            class Integration(Definition[T], Generic[T]):
                shapes: tuple
                label: str = "default"
            INTEGRATION = Integration(name="first", shapes=("a", "b"))
        """,
        "pkg/other.py": """
            class Summary: pass
        """,
        "test_static_types.py": """
            import http
            from decimal import Decimal
            from http import HTTPStatus
            from typing import List, Optional, Union
            import pytest
            import pkg.types
            import pkg.types as aliased
            from pkg import types as reporting_types
            from pkg.types import INTEGRATION, Color, Palette, Reordered, Spooky
            PAIR = (1, 2)
            class Grouped:
                FIRST = "first"
                BOTH = (FIRST, "second")
                @pytest.mark.parametrize("name", BOTH)
                def test_class_body(self, name): pass
            @pytest.mark.parametrize("cls", [reporting_types.Summary, aliased.Detail, pkg.types.Summary])
            def test_module_attributes(cls): pass
            @pytest.mark.parametrize(
                "kind",
                [list[str], dict, tuple[int, ...], List[int], Optional[int], Union[str, None],
                 Union[int, str], Union[int], Optional[list[int]], str | None, int | int],
            )
            def test_generic_types(kind): pass
            @pytest.mark.parametrize(
                "kind",
                [Color, Palette, Union[Color, Spooky], Color | Spooky, Union[Color, Reordered],
                 Color | None, int | Color, list[int] | Color],
            )
            def test_literal_unions(kind): pass
            @pytest.mark.parametrize(
                "value",
                [float("nan"), float("-inf"), float(" 1.5 "), float(3), Decimal("1.5"), HTTPStatus.OK,
                 http.HTTPStatus.NOT_FOUND, pytest.ExitCode.TESTS_FAILED],
            )
            def test_scalars(value): pass
            @pytest.mark.parametrize("shapes, label, name", [(INTEGRATION.shapes, INTEGRATION.label, INTEGRATION.name)])
            def test_generic_record(shapes, label, name): pass
            @pytest.mark.parametrize("pair, name", [(PAIR, "a"), pytest.param(PAIR, "b")])
            def test_sequence_component(pair, name): pass
            @pytest.mark.parametrize("pair", [PAIR, pytest.param(PAIR)])
            def test_sequence_value(pair): pass
            @pytest.mark.parametrize("index, name", enumerate(["a", "b"], start=1))
            def test_enumerate(index, name): pass
            @pytest.mark.parametrize("left, right", zip(["a", "b"], [1, 2], strict=True))
            def test_zip(left, right): pass
            @pytest.mark.parametrize("key", {**{"b": 1}, "a": 2})
            def test_dict_unpack(key): pass
        """,
        "test_runtime_types.py": """
            import typing
            from typing import List, Union
            import pytest
            from pkg import other, types
            @pytest.mark.parametrize("kind", [Union[types.Summary, other.Summary]])
            def test_same_named_classes(kind): pass
            @pytest.mark.parametrize("kind", [Union[List[int], typing.List[int]]])
            def test_generic_spellings(kind): pass
            @pytest.mark.parametrize("value", [float("1_000")])
            def test_grouped_float(value): pass
        """,
    }
    with create_test_project({name: textwrap.dedent(source) for name, source in files.items()}) as root:
        assert_hybrid_matches_pytest(
            root,
            ["test_static_types.py", "test_runtime_types.py"],
            {
                "test_runtime_types.py::test_same_named_classes",
                "test_runtime_types.py::test_generic_spellings",
                "test_runtime_types.py::test_grouped_float",
            },
        )


def test_imported_class_constants_match_pytest() -> None:
    files = {
        "constants.py": """
            from typing import ClassVar
            class Lookup:
                NAMES: ClassVar[dict[str, str]] = {"b": "Bee", "a": "Ay"}
        """,
        "test_imported_class_constants.py": """
            import pytest
            from constants import Lookup
            @pytest.mark.parametrize("key", Lookup.NAMES.keys())
            def test_keys(key): pass
            @pytest.mark.parametrize("value", sorted(Lookup.NAMES.values()))
            def test_values(value): pass
        """,
    }
    with create_test_project({name: textwrap.dedent(source) for name, source in files.items()}) as root:
        assert_hybrid_matches_pytest(root, ["test_imported_class_constants.py"], set())


def test_star_imports_and_nested_bases_match_pytest() -> None:
    files = {
        "pkg/__init__.py": "",
        "pkg/_gen/__init__.py": "",
        "pkg/_gen/enums.py": """
            from typing import Final, Literal, TypeAlias
            class ThingEnums:
                class Status:
                    Enum: TypeAlias = Literal["ACTIVE", "CLOSED"]
                    ACTIVE: Final = "ACTIVE"
                    CLOSED: Final = "CLOSED"
        """,
        "pkg/_gen/entity.py": """
            import enum
            from pkg._gen.enums import ThingEnums
            class Thing:
                class Status(ThingEnums.Status):
                    pass
                class Color(enum.Enum):
                    RED = "r"
            def make_thing():
                return Thing()
        """,
        "pkg/_gen/all_types.py": """
            from pkg._gen.entity import Thing, make_thing
            __all__ = ["Thing"]
        """,
        "pkg/types.py": """
            from pkg._gen.all_types import *  # noqa: F403
            from pkg._gen.all_types import __all__ as __all__
        """,
        "pkg/public.py": """
            from .enums_reexport import *  # noqa: F403
        """,
        "pkg/enums_reexport.py": """
            from pkg._gen.entity import *  # noqa: F403
        """,
        "pkg/dynamic.py": """
            from pkg._gen.entity import *  # noqa: F403
            __all__ = [name for name in list(globals()) if not name.startswith("_")]
        """,
        "pkg/dynamic_reexport.py": """
            from pkg.dynamic import *  # noqa: F403
        """,
        "test_static_star.py": """
            import pytest
            from pkg.types import Thing
            from pkg.public import ThingEnums, make_thing
            @pytest.mark.parametrize("status", [Thing.Status.ACTIVE, Thing.Status.CLOSED])
            def test_nested_base(status): pass
            @pytest.mark.parametrize("status", [ThingEnums.Status.ACTIVE])
            def test_public_names(status): pass
            @pytest.mark.parametrize("value", [Thing, Thing.Status, make_thing])
            def test_named(value): pass
            @pytest.mark.parametrize("color", [Thing.Color.RED])
            def test_nested_enum(color): pass
        """,
        "test_runtime_star.py": """
            import pytest
            from pkg.dynamic import Thing
            from pkg.dynamic_reexport import Thing as Reexported
            @pytest.mark.parametrize("status", [Thing.Status.ACTIVE])
            def test_star_source(status): pass
            @pytest.mark.parametrize("status", [Reexported.Status.ACTIVE])
            def test_dynamic_all(status): pass
        """,
    }
    with create_test_project({name: textwrap.dedent(source) for name, source in files.items()}) as root:
        assert_hybrid_matches_pytest(
            root,
            ["test_static_star.py", "test_runtime_star.py"],
            {"test_runtime_star.py::test_dynamic_all"},
        )


def test_unpacking_and_method_names_match_pytest() -> None:
    files = {
        "helpers.py": """
            import functools
            class Writer:
                @staticmethod
                def advance(ctx): pass
                @classmethod
                def build(cls): pass
                def plain(self): pass
                @functools.cache
                def cached(self): pass
            FIRST = ["a", "b"]
        """,
        "test_static_unpack.py": """
            import pytest
            from helpers import FIRST, Writer
            BOTH = [*FIRST, "c", *("d",)]
            @pytest.mark.parametrize("value", BOTH)
            def test_named_unpack(value): pass
            @pytest.mark.parametrize("value", [*FIRST, "z"])
            def test_inline_unpack(value): pass
            @pytest.mark.parametrize("action", [Writer.advance, Writer.build, Writer.plain])
            def test_methods(action): pass
            @pytest.mark.parametrize("status,action", [("s", Writer.advance), ("t", Writer.build)])
            def test_method_rows(status, action): pass
        """,
        "test_runtime_unpack.py": """
            import pytest
            from helpers import Writer
            MAPPING = {"k": 1}
            @pytest.mark.parametrize("action", [Writer.cached])
            def test_wrapping_decorator(action): pass
            @pytest.mark.parametrize("value", [*"ab"])
            def test_string_unpack(value): pass
        """,
    }
    with create_test_project({name: textwrap.dedent(source) for name, source in files.items()}) as root:
        assert_hybrid_matches_pytest(
            root,
            ["test_static_unpack.py", "test_runtime_unpack.py"],
            {
                "test_runtime_unpack.py::test_wrapping_decorator",
                "test_runtime_unpack.py::test_string_unpack",
            },
        )


def test_record_attribute_ids_match_pytest() -> None:
    files = {
        "records.py": """
            import dataclasses
            from dataclasses import dataclass, field
            from typing import ClassVar, NamedTuple
            @dataclass(frozen=True)
            class Case:
                KIND: ClassVar[str] = "case"
                id: str
                value: int = 0
                tags: list = field(default_factory=list)
            class Spec(NamedTuple):
                name: str
                kind: str = "k"
            class Plain:
                def __init__(self, name, *, code="c"):
                    \"\"\"Holds a name.\"\"\"
                    self.label = name
                    self.code = code
            @dataclasses.dataclass
            class Checked:
                id: str
                def __post_init__(self):
                    self.id = self.id.upper()
            class Computed:
                def __init__(self, name):
                    self.label = name.upper()
            class Status:
                DONE = "done"
            @dataclass(frozen=True)
            class Child(Case):
                extra: str = ""
        """,
        "reexport.py": """
            from records import Status
            Status = Status
        """,
        "test_static_records.py": """
            import pytest
            from records import Case, Plain, Spec
            def compute(): return 3
            CASES = [Case("a", 1), Case(id="b", value=compute())]
            SPECS = [Spec("x"), Spec(name="y", kind="z")]
            PLAINS = [Plain("p"), Plain(name="q", code="r")]
            FIRST = Case("first")
            @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
            def test_lambda(case): pass
            @pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
            def test_comprehension(case): pass
            @pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.kind)
            def test_named_tuple(spec): pass
            @pytest.mark.parametrize("plain", PLAINS, ids=lambda plain: plain.label)
            def test_init_attribute(plain): pass
            @pytest.mark.parametrize("plain", PLAINS, ids=lambda plain: plain.code)
            def test_keyword_default(plain): pass
            @pytest.mark.parametrize("case", CASES, ids=lambda case: None)
            def test_none(case): pass
            @pytest.mark.parametrize("cls", [Case, Spec, Plain], ids=lambda cls: cls.__name__)
            def test_class_names(cls): pass
            @pytest.mark.parametrize("case", [Case("i"), pytest.param(Case("j"), id="own")], ids=lambda c: c.id)
            def test_inline(case): pass
            @pytest.mark.parametrize("value", [FIRST.id, FIRST.value])
            def test_constant_attribute(value): pass
            from reexport import Status
            @pytest.mark.parametrize("value", [Status.DONE])
            def test_alias_attribute(value): pass
        """,
        "test_runtime_records.py": """
            import pytest
            from records import Case, Checked, Child, Computed
            def compute(x="x"): return x
            @pytest.mark.parametrize("case", [Checked("a")], ids=lambda case: case.id)
            def test_post_init(case): pass
            @pytest.mark.parametrize("case", [Computed("a")], ids=lambda case: case.label)
            def test_computed_init(case): pass
            @pytest.mark.parametrize("case", [Child("a")], ids=lambda case: case.id)
            def test_inherited(case): pass
            @pytest.mark.parametrize("case", [Case(compute())], ids=lambda case: case.id)
            def test_unknown_field(case): pass
            @pytest.mark.parametrize("case", [Case("a")], ids=lambda case: case)
            def test_object_id(case): pass
        """,
    }
    with create_test_project({name: textwrap.dedent(source) for name, source in files.items()}) as root:
        assert_hybrid_matches_pytest(
            root,
            ["test_static_records.py", "test_runtime_records.py"],
            {
                "test_runtime_records.py::test_post_init",
                "test_runtime_records.py::test_computed_init",
                "test_runtime_records.py::test_unknown_field",
                "test_runtime_records.py::test_object_id",
            },
        )


def test_cannot_expand_report_destinations_and_lifecycle() -> None:
    files = {
        "test_dynamic.py": (
            "import pytest\ndef values(n=2): return list(range(n))\n"
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
