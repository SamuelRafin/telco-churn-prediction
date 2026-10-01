"""Execute a notebook in place, embedding its outputs.

Used to produce a notebook that renders with charts visible on GitHub.
Reports the first failing cell clearly instead of a long traceback.

Usage:
    python scripts/run_notebook.py notebooks/01_eda.ipynb
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def run(path: Path) -> int:
    notebook = nbformat.read(path, as_version=4)
    # Ensure every cell has an id (required by nbformat >= 5.1.4).
    nbformat.validator.normalize(notebook)

    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )

    try:
        client.execute()
    except CellExecutionError as error:
        print(f"FAILED: {path.name}")
        print(str(error)[:3000])
        nbformat.write(notebook, path)
        return 1

    nbformat.write(notebook, path)

    n_outputs = sum(
        1
        for cell in notebook.cells
        if cell.get("cell_type") == "code" and cell.get("outputs")
    )
    n_images = sum(
        1
        for cell in notebook.cells
        if cell.get("cell_type") == "code"
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    )
    print(f"OK: {path.name}")
    print(f"  code cells with output: {n_outputs}")
    print(f"  embedded images:        {n_images}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(run(Path(sys.argv[1]).resolve()))
