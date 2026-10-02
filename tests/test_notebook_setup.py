"""Regression checks for standalone Colab notebooks and local setup."""

import subprocess
import sys
import types
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).parents[1]
NOTEBOOKS = sorted((ROOT / "course/notebooks").glob("*.ipynb"))


def setup_source(path=NOTEBOOKS[0]):
    notebook = nbformat.read(path, as_version=4)
    return next(cell.source for cell in notebook.cells if cell.cell_type == "code")


def checkout(path):
    (path / "course").mkdir(parents=True)
    (path / "course/curriculum.yaml").write_text("lessons: []\n")
    (path / "src/course_aiml_autoencoders").mkdir(parents=True)


def colab_modules(monkeypatch):
    google = types.ModuleType("google")
    google.__path__ = []
    colab = types.ModuleType("google.colab")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


def test_every_notebook_bootstraps_before_course_imports():
    assert len(NOTEBOOKS) == 14
    expected = setup_source()
    for path in NOTEBOOKS:
        assert setup_source(path) == expected
        notebook = nbformat.read(path, as_version=4)
        # The actual imports follow setup, preventing the reported Colab error.
        code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
        assert "from course_aiml_autoencoders" in code[1]


def test_local_setup_finds_checkout_without_network(tmp_path, monkeypatch):
    checkout(tmp_path)
    child = tmp_path / "course/notebooks"
    child.mkdir()
    monkeypatch.chdir(child)
    monkeypatch.setattr(sys, "path", sys.path.copy())

    def unexpected_clone(*args, **kwargs):
        pytest.fail("Local setup must not clone a repository")

    monkeypatch.setattr(subprocess, "run", unexpected_clone)
    exec(setup_source(), {})
    assert Path.cwd() == tmp_path
    assert sys.path[0] == str(tmp_path / "src")


def test_fresh_colab_clones_once_and_reruns_safely(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    colab_modules(monkeypatch)
    calls = []

    def clone(command, *, check):
        assert check is True
        assert command[:-1] == [
            "git", "clone", "--depth", "1", "--branch", "main",
            "https://github.com/tsilva/course-aiml-autoencoders.git",
        ]
        calls.append(command)
        checkout(Path(command[-1]))

    monkeypatch.setattr(subprocess, "run", clone)
    exec(setup_source(), {})
    expected = tmp_path / "course-aiml-autoencoders"
    assert Path.cwd() == expected
    assert sys.path[0] == str(expected / "src")
    exec(setup_source(), {})
    assert len(calls) == 1
    assert sys.path.count(str(expected / "src")) == 1


def test_colab_preserves_incomplete_existing_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    colab_modules(monkeypatch)
    existing = tmp_path / "course-aiml-autoencoders"
    existing.mkdir()
    sentinel = existing / "personal.txt"
    sentinel.write_text("preserve this")
    with pytest.raises(RuntimeError, match="Incomplete course checkout"):
        exec(setup_source(), {})
    assert sentinel.read_text() == "preserve this"


def test_colab_clone_failure_is_not_hidden(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    colab_modules(monkeypatch)

    def failed_clone(command, **kwargs):
        raise subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(subprocess, "run", failed_clone)
    with pytest.raises(subprocess.CalledProcessError):
        exec(setup_source(), {})
