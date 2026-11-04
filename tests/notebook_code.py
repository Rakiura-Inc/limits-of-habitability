"""Run a notebook's own definitions in a test, so the method a member reads is the method tested.

Only the imports, functions and plain constants of each code cell run: nothing that downloads,
fits or draws.
"""

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import habtrain as ht

NOTEBOOKS = Path(__file__).parent.parent / "notebooks"


SAFE_CALLS = {"len", "enumerate", "range", "zip", "dict", "list", "sorted", "set"}


def _is_safe(part) -> bool:
    """False for anything that could download, fit or draw: any call but a few builtins, any lookup."""
    if isinstance(part, ast.Call):
        return isinstance(part.func, ast.Name) and part.func.id in SAFE_CALLS
    return not isinstance(part, ast.Subscript)


def _is_definition(node) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
        return True
    if isinstance(node, ast.Assign):  # a constant, or a table of names built from constants
        return all(_is_safe(part) for part in ast.walk(node.value))
    return False


def definitions(notebook: str, **names) -> dict:
    """The notebook's functions and constants, with `names` (and np, pd, plt, ht) already defined."""
    namespace = {"np": np, "pd": pd, "plt": plt, "ht": ht, **names}
    for cell in json.loads((NOTEBOOKS / notebook).read_text())["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or source.lstrip().startswith(("%", "#!")) or "%pip" in source:
            continue
        for node in ast.parse(source).body:
            if not _is_definition(node):
                continue
            code = compile(ast.Module(body=[node], type_ignores=[]), notebook, "exec")
            try:
                exec(code, namespace)  # noqa: S102 - our own notebooks, read in a test
            except NameError:
                continue  # a value worked out from a result, such as converged = N_STEPS > 50 * tau
    return namespace
