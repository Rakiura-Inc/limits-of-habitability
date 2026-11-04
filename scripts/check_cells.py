"""Teaching-cell checks ruff cannot make: cell size, state probing, semicolons, nesting.

Usage: python scripts/check_cells.py notebooks/*.ipynb   (exit 1 if anything blocks)
"""

import ast
import json
import re
import sys
from pathlib import Path

WARN_LINES = 15  # teaching-code-style C3
BLOCK_LINES = 25


COMPREHENSIONS = (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)


def contains_comprehension(node):
    """True if another comprehension sits inside this one."""
    return any(child is not node and isinstance(child, COMPREHENSIONS) for child in ast.walk(node))


def code_lines(source):
    """The lines that are code: not blank, not comments, not %pip or !shell lines."""
    return [line for line in source.splitlines() if line.strip() and not line.strip().startswith(("#", "%", "!"))]


def check_cell(source):
    """(blocks, warnings) for one code cell."""
    blocks, warnings = [], []
    lines = code_lines(source)
    if len(lines) > BLOCK_LINES:
        blocks.append(f"{len(lines)} lines of code (C3: split it)")
    elif len(lines) > WARN_LINES:
        warnings.append(f"{len(lines)} lines of code (C3)")
    if re.search(r"""["']\w+["']\s+(not\s+)?in\s+(dir|globals|locals)\(\)""", source):
        blocks.append("probes earlier cells with dir() or globals() (C4)")
    if any(line.rstrip().endswith(";") for line in lines):
        blocks.append("a line ends in ; (F4)")
    tree = ast.parse(re.sub(r"^(\s*)[%!].*$", r"\1pass", source, flags=re.MULTILINE))
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            warnings.append(f"lambda on line {node.lineno} (F5; ruff E731 blocks a named one)")
        if isinstance(node, ast.NamedExpr):
            warnings.append(f":= on line {node.lineno} (F1)")
        if isinstance(node, COMPREHENSIONS) and contains_comprehension(node):
            warnings.append(f"nested comprehension on line {node.lineno} (F1)")
    return blocks, warnings


def main(paths):
    blocked = False
    for path in paths:
        notebook = json.loads(Path(path).read_text())
        for number, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            blocks, warnings = check_cell("".join(cell["source"]))
            for message in blocks:
                print(f"BLOCK {path} cell {number}: {message}")
            for message in warnings:
                print(f"warn  {path} cell {number}: {message}")
            blocked = blocked or bool(blocks)
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
