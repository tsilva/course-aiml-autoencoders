#!/usr/bin/env python3
"""Bundle exact course sources and lessons into an uploadable Colab harness."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zlib

import nbformat
import yaml


ROOT = Path(__file__).resolve().parents[4]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_payload(root: Path, selection: list[str] | None = None) -> dict:
    lessons = yaml.safe_load((root / "course/curriculum.yaml").read_text())["lessons"]
    selected = []
    for selector in selection or [lesson["id"] for lesson in lessons]:
        name = Path(selector).name
        matches = [lesson for lesson in lessons if name in {
            lesson["id"], str(int(lesson["id"])),
            Path(lesson["notebook"]).name,
            Path(lesson["notebook_source"]).name,
        }]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous lesson: {selector}")
        if matches[0] not in selected:
            selected.append(matches[0])
    notebooks = []
    for lesson in selected:
        path = root / "course" / lesson["notebook"]
        raw = path.read_bytes()
        notebook = nbformat.reads(raw.decode(), as_version=4)
        code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        if not code or any(cell.outputs or cell.execution_count is not None for cell in code):
            raise ValueError(f"Expected unexecuted generated notebook: {path}")
        for index, cell in enumerate(code, 1):
            # exec is appropriate for this course's Python-only code cells.
            # Fail before upload if future lessons need IPython magic handling.
            compile(cell.source, f"{path.name}:cell:{index}", "exec")
        notebooks.append({"id": lesson["id"], "path": str(path.relative_to(root)),
                          "sha256": digest(raw), "code_cells": len(code),
                          "notebook": json.loads(json.dumps(notebook))})
    bootstraps = [next(cell["source"] for cell in item["notebook"]["cells"]
                       if cell["cell_type"] == "code") for item in notebooks]
    if len(set(bootstraps)) != 1:
        raise ValueError("Setup cells differ; prepare and validate each lesson separately")
    paths = sorted({
        *root.glob("src/course_aiml_autoencoders/**/*.py"),
        *root.glob("recipes/**/*.yaml"),
        root / "course/curriculum.yaml", root / "pyproject.toml", root / "uv.lock",
    })
    files = {str(path.relative_to(root)): path.read_text() for path in paths}
    snapshot = {"files": files, "lessons": notebooks}
    fingerprint = digest(json.dumps(snapshot, sort_keys=True).encode())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return {"snapshot": snapshot, "fingerprint": fingerprint, "head": head}


def prepare(root: Path, output: Path, selection: list[str] | None = None) -> Path:
    subprocess.run([sys.executable, str(root / "scripts/build_notebooks.py"), "--check"],
                   cwd=root, check=True)
    payload = make_payload(root, selection)
    output.resolve().relative_to((root / "runs").resolve())
    output.mkdir(parents=True, exist_ok=False)
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    setup = next(cell["source"] for cell in payload["snapshot"]["lessons"][0]["notebook"]["cells"]
                 if cell["cell_type"] == "code")
    runtime = Path(__file__).with_name("colab_runtime.py").read_text()
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(
            "# Course Colab validation\nRun all in a **fresh Google Colab runtime**. "
            "This copy embeds the tested working tree; it does not publish changes.\n\n"
            f"Snapshot: `{payload['fingerprint']}`\n\n"
            "Execution success still requires numerical and visual result review."),
        nbformat.v4.new_code_cell(
            "import google.colab\nfrom pathlib import Path\nimport sys\n"
            "if any(name.startswith('course_aiml_autoencoders') for name in sys.modules):\n"
            "    raise RuntimeError('Use a fresh runtime: course modules already loaded')\n"
            "if any((p / 'course/curriculum.yaml').exists() for p in (Path.cwd(), *Path.cwd().parents)) "
            "or (Path.cwd() / 'course-aiml-autoencoders').exists():\n"
            "    raise RuntimeError('Use a fresh runtime: course checkout already present')"),
        nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_code_cell(f"ENCODED_SNAPSHOT = {encoded!r}\n" + runtime),
    ], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    destination = output / "validate-colab.ipynb"
    nbformat.write(notebook, destination)
    manifest = {key: value for key, value in payload.items() if key != "snapshot"}
    manifest["lessons"] = [{key: value for key, value in item.items() if key != "notebook"}
                           for item in payload["snapshot"]["lessons"]]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", action="append", help="Lesson ID, filename, or path; default: all")
    parser.add_argument("--output-dir", type=Path, help="New directory under runs/")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or ROOT / "runs/colab-validation" / stamp
    if not output.is_absolute():
        output = ROOT / output
    try:
        path = prepare(ROOT, output, args.notebook)
    except (ValueError, SyntaxError) as error:
        parser.error(str(error))
    print(f"Upload to Colab: {path}\nLocal evidence directory: {output}")


if __name__ == "__main__":
    main()
