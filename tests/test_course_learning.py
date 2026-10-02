from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch

from course_aiml_autoencoders.config import load_yaml, save_yaml
from course_aiml_autoencoders.course import learn, learn_prior, lesson_config
from course_aiml_autoencoders.course import learning


def _write_run(config, root, count, checkpoint=None):
    run_dir = root / config["id"] / str(count)
    run_dir.mkdir(parents=True)
    resolved = dict(config)
    if checkpoint:
        resolved.update(vq_checkpoint=str(checkpoint), token_map_shape=[2, 2], codebook_size=8)
    save_yaml(resolved, run_dir / "resolved-config.yaml")
    (run_dir / "summary.json").write_text(json.dumps({"elapsed_seconds": 1, "device": "cpu"}))
    (run_dir / "metrics.jsonl").write_text('{}\n')
    (run_dir / "checkpoint-best.pt").write_bytes(b"completed checkpoint")
    figures = run_dir / "figures"
    figures.mkdir()
    for name in ("reconstructions.png", "latent-space.png", "random-latent-samples.png",
                 "kl-per-dimension.png", "token-maps.png", "codebook-usage.png",
                 "uniform-random-token-samples.png", "corrupted-input-reconstructions.png",
                 "learned-prior-samples.png"):
        (figures / name).write_bytes(b"figure")
    return SimpleNamespace(run_dir=run_dir)


def test_quick_config_preserves_scientific_settings_and_bounds_the_budget():
    recipe = load_yaml("recipes/vae/vae-002-warmup.yaml")
    config = lesson_config(recipe)
    assert config["model"] == recipe["model"]
    assert config["objective"]["beta"] == recipe["objective"]["beta"]
    assert config["training"]["epochs"] == 8
    assert config["training"]["device"] == "cpu"
    assert config["objective"]["kl_warmup_epochs"] <= 8
    assert (config["dataset"]["train_size"], config["dataset"]["validation_size"]) == (2048, 512)
    assert recipe["training"]["epochs"] == 20
    assert lesson_config(recipe, profile="full")["training"] == recipe["training"]
    with pytest.raises(ValueError, match="profile"):
        lesson_config(recipe, profile="unknown")


def test_reuse_requires_exact_config_complete_artifacts_and_current_source(tmp_path, monkeypatch):
    calls = []
    def train(config, *, run_root):
        calls.append(config)
        return _write_run(config, run_root, len(calls))
    monkeypatch.setattr(learning, "run_training", train)
    monkeypatch.setattr(learning, "_source_digest", lambda: "version-one")
    recipe = load_yaml("recipes/ae/ae-001-linear.yaml")
    first = learn(recipe, run_root=tmp_path)
    assert learn(recipe, run_root=tmp_path) == first
    assert len(calls) == 1
    changed = {**recipe, "objective": {**recipe["objective"], "latent_l1_weight": 0.1}}
    assert learn(changed, run_root=tmp_path) != first
    assert learn(recipe, run_root=tmp_path, rerun=True) != first
    monkeypatch.setattr(learning, "_source_digest", lambda: "version-two")
    current = learn(recipe, run_root=tmp_path)
    (current / "figures/reconstructions.png").unlink()
    assert learn(recipe, run_root=tmp_path) != current
    assert len(calls) == 5
    manifest = json.loads((tmp_path / "course/session.json").read_text())
    assert all(Path(item["run_dir"]).is_absolute() for item in manifest.values())


def test_prior_reuse_tracks_checkpoint_contents_and_pairing(tmp_path, monkeypatch):
    vq = tmp_path / "vq"
    vq.mkdir()
    (vq / "checkpoint-best.pt").write_bytes(b"vq version one")
    calls = []
    def prior(config, checkpoint, *, run_root):
        calls.append(config)
        return _write_run(config, run_root, len(calls), checkpoint)
    monkeypatch.setattr(learning, "run_code_prior_training", prior)
    monkeypatch.setattr(learning, "_source_digest", lambda: "fixed-source")
    first = learn_prior("recipes/prior/prior-001-gru.yaml", vq, run_root=tmp_path)
    assert learn_prior("recipes/prior/prior-001-gru.yaml", vq, run_root=tmp_path) == first
    (vq / "checkpoint-best.pt").write_bytes(b"vq version two")
    assert learn_prior("recipes/prior/prior-001-gru.yaml", vq, run_root=tmp_path) != first
    assert len(calls) == 2
