"""
Notebook hygiene, and static analysis of the code the notebooks contain.

A notebook that does not parse, that ships a saved traceback, or that reads a
name nothing defines is broken for every reader before they run a single cell.
None of that is visible in a diff and all of it is cheap to check, so check it.

Nothing here starts a kernel: these tests are about what can be known from the
file alone, and they run in well under a second.
"""

import ast
import builtins
import json
import re
import symtable
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parent.parent / "notebooks").glob("*.ipynb"))


def saved_errors(notebook):
    """
    Names of exceptions stored in a notebook's saved outputs.

    Args:
        notebook: Parsed notebook dict

    Returns:
        List of "cell <index>: <ExceptionName>" strings
    """
    found = []
    for index, cell in enumerate(notebook.get("cells", [])):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                found.append(f"cell {index}: {output.get('ename', 'error')}")
    return found


class TestSavedErrorDetection:
    """The checker itself, so a green suite below means something."""

    def test_finds_a_saved_traceback(self):
        notebook = {
            "cells": [
                {"cell_type": "code", "outputs": []},
                {
                    "cell_type": "code",
                    "outputs": [
                        {"output_type": "error", "ename": "ValueError"}
                    ],
                },
            ]
        }

        assert saved_errors(notebook) == ["cell 1: ValueError"]

    def test_clean_notebook_reports_nothing(self):
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "outputs": [{"output_type": "stream", "text": "ok"}],
                }
            ]
        }

        assert saved_errors(notebook) == []

    def test_markdown_only_notebook_reports_nothing(self):
        assert saved_errors({"cells": [{"cell_type": "markdown"}]}) == []


class TestEveryNotebook:
    def test_there_are_notebooks_to_check(self):
        """Guards against a path typo silently making the suite below vacuous."""
        assert len(NOTEBOOKS) >= 26

    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
    def test_parses_as_json(self, path):
        json.loads(path.read_text())

    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
    def test_has_no_saved_traceback(self, path):
        errors = saved_errors(json.loads(path.read_text()))

        assert errors == [], f"{path.name} ships a saved traceback: {errors}"

    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
    def test_declares_a_kernel(self, path):
        """Without kernelspec metadata, Jupyter cannot pick an interpreter."""
        notebook = json.loads(path.read_text())

        assert "kernelspec" in notebook.get("metadata", {}), (
            f"{path.name} has no kernelspec"
        )


# ---------------------------------------------------------------------------
# Static analysis of the notebook code itself.
#
# The checks above catch a notebook that is broken as a *file*. These catch one
# that is broken as a *program*: a cell that does not parse, a name that is
# never defined, a name used in an earlier cell than the one that defines it.
# All three survive a re-save and none of them show up until a reader runs the
# cell and hits the traceback.
#
# Nothing here executes notebook code, so the suite stays fast and needs no
# kernel.
# ---------------------------------------------------------------------------

# `%magic`, `!shell` and `?help` are IPython syntax, not Python. Blanking those
# lines (rather than dropping them) keeps line numbers aligned with the cell.
IPYTHON_LINE = re.compile(r"^\s*[%!?]")

# Names a notebook may use without defining: real builtins plus what IPython
# injects into the namespace.
ALLOWED_FREE_NAMES = set(dir(builtins)) | {
    "get_ipython", "display", "In", "Out", "__file__", "__name__", "__doc__",
}


def cell_code(cell):
    """A code cell's source, with IPython-only lines neutralized."""
    source = "".join(cell.get("source", []))
    return "\n".join(
        "pass" if IPYTHON_LINE.match(line) else line
        for line in source.split("\n")
    )


def code_cells(notebook):
    """(index, source) for each code cell, in document order."""
    return [
        (index, cell_code(cell))
        for index, cell in enumerate(notebook.get("cells", []))
        if cell["cell_type"] == "code"
    ]


def as_one_module(notebook):
    """
    Every code cell concatenated, which is what running top-to-bottom means.

    Returns (source, [(first_line, cell_index), ...]) so a line number in the
    combined source can be reported as a cell number.
    """
    pieces, offsets, line = [], [], 1
    for index, source in code_cells(notebook):
        pieces.append(source)
        offsets.append((line, index))
        line += source.count("\n") + 2
    return "\n\n".join(pieces), offsets


def undefined_names(source):
    """
    Names read but never bound anywhere in the notebook.

    symtable does the scope analysis, so closures, comprehensions, `global`
    and class bodies are handled by Python's own rules rather than by a
    hand-rolled approximation of them.
    """
    top = symtable.symtable(source, "<notebook>", "exec")
    module_level = {
        symbol.get_name()
        for symbol in top.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_parameter()
    }

    missing = set()

    def visit(table, is_module):
        for symbol in table.get_symbols():
            name = symbol.get_name()
            if not symbol.is_referenced() or name in ALLOWED_FREE_NAMES:
                continue
            if is_module:
                bound = (symbol.is_assigned() or symbol.is_imported()
                         or symbol.is_parameter())
                if not bound:
                    missing.add(name)
            elif symbol.is_global() and name not in module_level:
                # Resolves to module scope, but module scope has no such name.
                missing.add(name)
        for child in table.get_children():
            visit(child, False)

    visit(top, True)
    return sorted(missing)


class TestUndefinedNameDetection:
    """The checker itself, so a green suite below means something."""

    def test_flags_a_name_that_is_never_defined(self):
        assert undefined_names("x = undefined_thing + 1") == ["undefined_thing"]

    def test_accepts_a_name_defined_in_an_earlier_cell(self):
        assert undefined_names("a = 1\n\nprint(a)") == []

    def test_accepts_builtins(self):
        assert undefined_names("print(len([1, 2]))") == []

    def test_looks_inside_function_bodies(self):
        assert undefined_names("def f():\n    return nope") == ["nope"]

    def test_accepts_a_global_defined_after_the_function(self):
        # Legal: the body runs after `later` exists.
        assert undefined_names("def f():\n    return later\n\nlater = 1") == []

    def test_handles_comprehension_scope(self):
        assert undefined_names("xs = [1]\nys = [v * 2 for v in xs]") == []


class TestEveryNotebookParses:
    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
    def test_every_code_cell_is_valid_python(self, path):
        notebook = json.loads(path.read_text())

        for index, source in code_cells(notebook):
            try:
                ast.parse(source)
            except SyntaxError as error:
                pytest.fail(
                    f"{path.name} cell {index} line {error.lineno}: {error.msg}"
                )

    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
    def test_no_undefined_names(self, path):
        notebook = json.loads(path.read_text())
        source, _ = as_one_module(notebook)

        missing = undefined_names(source)

        assert missing == [], (
            f"{path.name} reads names that are never defined: {missing}. "
            f"Either the defining cell was deleted, or it is a typo."
        )


class TestNotebookHygiene:
    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
    def test_has_no_empty_cells(self, path):
        """An empty cell is always an editing leftover, never intent."""
        notebook = json.loads(path.read_text())

        empty = [
            index
            for index, cell in enumerate(notebook["cells"])
            if not "".join(cell.get("source", [])).strip()
        ]

        assert empty == [], f"{path.name} has empty cells at {empty}"

    @pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
    def test_code_cells_have_no_trailing_whitespace(self, path):
        notebook = json.loads(path.read_text())

        offenders = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            for line in "".join(cell["source"]).split("\n"):
                if line != line.rstrip():
                    offenders.append(index)
                    break

        assert offenders == [], (
            f"{path.name} has trailing whitespace in cells {offenders}"
        )
