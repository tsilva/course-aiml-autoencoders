"""Verify snapshot provenance and that the harness records real cell failures."""

import base64
import importlib.util
import json
from pathlib import Path
import zlib

import nbformat
import pytest


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / ".agents/skills/validate-colab-notebooks/scripts"
SPEC = importlib.util.spec_from_file_location("prepare_colab", SCRIPTS / "prepare_validation.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def test_selection_and_exact_snapshot():
    whole = prepare.make_payload(ROOT)
    assert [item["id"] for item in whole["snapshot"]["lessons"]] == [f"{i:02d}" for i in range(14)]
    assert sum(item["code_cells"] for item in whole["snapshot"]["lessons"]) == 45
    by_id = prepare.make_payload(ROOT, ["02"])
    by_path = prepare.make_payload(ROOT, ["course/notebooks/02-linear-ae.ipynb"])
    assert by_id == by_path
    item = by_id["snapshot"]["lessons"][0]
    assert item["sha256"] == prepare.digest((ROOT / item["path"]).read_bytes())
    assert by_id["snapshot"]["files"]["src/course_aiml_autoencoders/course/learning.py"] == (
        ROOT / "src/course_aiml_autoencoders/course/learning.py").read_text()
    assert "recipes/ae/ae-001-linear.yaml" in by_id["snapshot"]["files"]
    assert prepare.make_payload(ROOT, ["02", "02"])["fingerprint"] == by_id["fingerprint"]


def test_invalid_selector_does_not_fall_back_to_all():
    with pytest.raises(ValueError, match="Unknown or ambiguous"):
        prepare.make_payload(ROOT, ["missing.ipynb"])


def test_generated_harness_is_uploadable_and_preserves_sources(tmp_path):
    # Use an ignored repository directory because the helper enforces evidence placement.
    import tempfile
    (ROOT / "runs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
        output = Path(directory) / "check"
        path = prepare.prepare(ROOT, output, ["00"])
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert all(not cell.outputs for cell in notebook.cells if cell.cell_type == "code")
        namespace = {}
        exec(notebook.cells[-1].source.split("\n", 1)[0], namespace)
        payload = json.loads(zlib.decompress(base64.b64decode(namespace["ENCODED_SNAPSHOT"])))
        manifest = json.loads((output / "manifest.json").read_text())
        assert payload["fingerprint"] == manifest["fingerprint"]
        original = nbformat.read(ROOT / "course/notebooks/00-laboratory.ipynb", as_version=4)
        assert notebook.cells[2].source == next(c.source for c in original.cells if c.cell_type == "code")
        assert [c["source"] for c in payload["snapshot"]["lessons"][0]["notebook"]["cells"]] == [
            c.source for c in original.cells]
    with pytest.raises(ValueError):
        prepare.prepare(ROOT, tmp_path / "outside-runs", ["00"])


def test_runtime_preserves_failure_and_keeps_semantic_review_pending(tmp_path, monkeypatch):
    root = tmp_path / "course-aiml-autoencoders"
    for directory in (".git", "src/course_aiml_autoencoders", "recipes"):
        (root / directory).mkdir(parents=True)
    monkeypatch.chdir(root)
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("print('real output')"),
        nbformat.v4.new_code_cell("raise ValueError('expected failure')"),
        nbformat.v4.new_code_cell("raise AssertionError('must not execute after failure')"),
    ])
    snapshot = {"files": {"src/course_aiml_autoencoders/__init__.py": ""}, "lessons": [
        {"id": "test", "path": "course/notebooks/test.ipynb", "sha256": "test-sha",
         "code_cells": 3, "notebook": json.loads(json.dumps(notebook))}]}
    payload = {"snapshot": snapshot, "head": "test-head",
               "fingerprint": prepare.digest(json.dumps(snapshot, sort_keys=True).encode())}
    namespace = {"ENCODED_SNAPSHOT": base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()}
    with pytest.raises(RuntimeError, match="Some lessons failed"):
        exec((SCRIPTS / "colab_runtime.py").read_text(), namespace)
    evidence = root / "runs/colab-validation" / payload["fingerprint"][:12]
    report = json.loads((evidence / "report.json").read_text())
    assert report["semantic_review"] == "pending"
    result = report["lessons"][0]
    assert result["status"] == "execution_fail"
    assert result["completed_code_cells"] == 1
    assert len(result["cells"]) == 2
    assert "expected failure" in result["traceback"]
    executed = nbformat.read(evidence / "test.ipynb", as_version=4)
    assert executed.cells[0].outputs[0].text == "real output\n"
    assert executed.cells[1].outputs[0].output_type == "error"
    assert executed.cells[2].execution_count is None
