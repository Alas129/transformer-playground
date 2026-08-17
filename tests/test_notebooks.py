"""
Notebook hygiene.

A notebook that does not parse, or that ships a saved traceback, is broken for
every reader before they run a single cell. Cheap to check, so check it.
"""

import json
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
