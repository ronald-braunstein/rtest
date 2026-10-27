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
        "test_strings.py": """
            import pytest
            NAME = "n"
            ONE = 1
            class Ids:
                PREFIX = "id"
                FLAG = False
            @pytest.mark.parametrize(
                "value",
                [f"x-{ONE}-{NAME}", f"{Ids.PREFIX}:{Ids.FLAG}:{None}", f"{NAME!s}" "-tail", {1, 2}, {"a": 1}],
            )
            def test_values(value): pass
            class TestScoped:
                LABEL = "scoped"
                @pytest.mark.parametrize("value", [f"{LABEL}!"])
                def test_class_attribute(self, value): pass
        """,
        # Expressions pytest IDs by their runtime value, one file each.
        **{
            f"test_runtime_{name}.py": f"import pytest\n{source}\n"
            f'@pytest.mark.parametrize("value", {argvalues})\ndef test_item(value): pass\n'
            for name, source, argvalues in [
                ("fstring_format", "ONE = 1", '[f"{ONE:03}"]'),
                ("fstring_repr", "NAME = 'n'", '[f"{NAME!r}"]'),
                ("fstring_runtime", "import os", '[f"{os.sep}x"]'),
                ("fstring_bytes", "", "[f\"{b'x'}\"]"),
                ("conditional", "ONE = 1", '["a" if ONE else "b"]'),
                ("comparison", "ONE = 1", "[ONE == 1]"),
                ("subscript", "LOOKUP = {'k': 'v'}", "[LOOKUP['k']]"),
                ("bool_op", "ONE = 1", "[ONE or 'x']"),
            ]
        },
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
        """,
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
        static_files = {"test_static.py", "test_strings.py"} | (
            {"test_duplicates.py"} if int(version.split(".")[0]) >= 8 else set()
        )
        assert fallback == set(test_files) - static_files
        collected = [normalize_nodeid(line) for line in extract_test_lines(result.output_lines)]
        assert reported == {nodeid for nodeid in collected if nodeid.split("::", 1)[0] in fallback}
        static = [nodeid for nodeid in collected if nodeid not in reported]
        hybrid = static + pytest_collect(root, sorted(fallback))
        assert sorted(hybrid) == sorted(pytest_collect(root, test_files))
        # Compare multisets: pytest 7 itself can emit duplicates for colliding explicit IDs.


def test_cannot_expand_report_destinations_and_lifecycle() -> None:
    files = {
        "test_dynamic.py": "import pytest\n@pytest.mark.parametrize('x', range(2))\ndef test_collected(x): pass\n",
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
