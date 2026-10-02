"""Short, reusable course experiments with explicit configuration provenance."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from course_aiml_autoencoders.config import load_yaml
from course_aiml_autoencoders.course.probes import load_run_summary, repository_root
from course_aiml_autoencoders.training import run_code_prior_training, run_training


def _source_digest() -> str:
    source_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256(torch.__version__.encode())
    for path in sorted(source_root.rglob("*.py")):
        digest.update(str(path.relative_to(source_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def lesson_config(
    recipe: str | Path | dict[str, Any], *, profile: str = "quick"
) -> dict[str, Any]:
    """Keep architectures/objectives intact; bound data and training in quick mode."""

    if profile not in {"quick", "full"}:
        raise ValueError("profile must be 'quick' or 'full'")
    config = copy.deepcopy(recipe) if isinstance(recipe, dict) else load_yaml(
        repository_root() / recipe
    )
    config["id"] = f"course/{profile}/{config['id']}"
    config["course"] = {"profile": profile, "source_digest": _source_digest(),
                        "data_protocol": "training-holdout-v1"}
    if "dataset" in config:
        config["dataset"]["root"] = str(
            (repository_root() / config["dataset"].get("root", "data")).resolve()
        )
    if profile == "quick":
        training = config["training"]
        training["epochs"] = min(8, int(training["epochs"]))
        training["device"] = "cpu"
        training["num_workers"] = 0
        if "dataset" in config:
            dataset = config["dataset"]
            dataset.setdefault("split_seed", 0)
            dataset["train_size"] = min(2048, int(dataset.get("train_size", 2048)))
            dataset["validation_size"] = min(512, int(dataset.get("validation_size", 512)))
            config.setdefault("diagnostics", {})["max_latent_points"] = dataset["validation_size"]
        objective = config.get("objective", {})
        if objective.get("kl_warmup_epochs", 0):
            objective["kl_warmup_epochs"] = min(
                int(objective["kl_warmup_epochs"]), training["epochs"]
            )
    return config


def _matching_run(config: dict[str, Any], run_root: Path, *, prior: bool) -> Path | None:
    directory = run_root / config["id"]
    if not directory.is_dir():
        return None
    for candidate in sorted(directory.iterdir(), reverse=True):
        required = ("resolved-config.yaml", "summary.json", "checkpoint-best.pt", "metrics.jsonl")
        if prior:
            figures = ["learned-prior-samples.png"]
        else:
            figures = ["reconstructions.png"]
            kind = config["model"]["kind"]
            if kind in {"ae", "vae"}:
                figures += ["latent-space.png", "random-latent-samples.png"]
            if kind == "vae":
                figures.append("kl-per-dimension.png")
            if kind == "vqvae":
                figures += ["token-maps.png", "codebook-usage.png", "uniform-random-token-samples.png"]
            if config["training"].get("evaluation_corruption", config["training"].get("input_corruption")):
                figures.append("corrupted-input-reconstructions.png")
        if not all((candidate / "figures" / name).is_file() for name in figures):
            continue
        if not all((candidate / name).is_file() for name in required):
            continue
        try:
            resolved = load_yaml(candidate / "resolved-config.yaml")
            if prior:
                resolved = {key: value for key, value in resolved.items()
                            if key not in {"vq_checkpoint", "token_map_shape", "codebook_size"}}
            summary = load_run_summary(candidate)
        except (ValueError, OSError, yaml.YAMLError):
            continue
        if resolved == config and "elapsed_seconds" in summary:
            return candidate
    return None


def learn(
    recipe: str | Path | dict[str, Any], *, profile: str = "quick",
    rerun: bool = False, run_root: str | Path | None = None,
) -> Path:
    """Return the exact matching completed run, or train once and print its timing."""

    return _learn(recipe, profile=profile, rerun=rerun, run_root=run_root)


def learn_prior(
    recipe: str | Path | dict[str, Any], vq_run: str | Path, *,
    profile: str = "quick", rerun: bool = False, run_root: str | Path | None = None,
) -> Path:
    """Reuse only a prior paired with this exact VQ checkpoint's contents."""

    checkpoint = (Path(vq_run) / "checkpoint-best.pt").resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Run Lesson 10 first to create a VQ checkpoint")
    return _learn(recipe, profile=profile, rerun=rerun, run_root=run_root,
                  checkpoint=checkpoint)


def _learn(recipe, *, profile, rerun, run_root, checkpoint=None) -> Path:
    config = lesson_config(recipe, profile=profile)
    if checkpoint is not None:
        config["course"]["vq_checkpoint"] = str(checkpoint)
        config["course"]["vq_checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    root = Path(run_root) if run_root is not None else repository_root() / "runs"
    cached = None if rerun else _matching_run(config, root, prior=checkpoint is not None)
    if cached is not None:
        run_dir = cached
        action = "Reusing"
    else:
        print(f"Training {config['id']} ({config['training']['epochs']} epochs; "
              f"device={config['training']['device']}).", flush=True)
        previous_threads = torch.get_num_threads()
        try:
            if profile == "quick":
                torch.set_num_threads(min(previous_threads, 4))
            result = (run_training(config, run_root=root) if checkpoint is None else
                      run_code_prior_training(config, checkpoint, run_root=root))
        finally:
            torch.set_num_threads(previous_threads)
        run_dir = result.run_dir.resolve()
        action = "Created"
    summary = load_run_summary(run_dir)
    print(f"{action}: {run_dir}\nMeasured training + diagnostics: "
          f"{summary['elapsed_seconds']:.1f} s on {summary['device']}", flush=True)
    manifest = root / "course" / "session.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    runs = json.loads(manifest.read_text()) if manifest.is_file() else {}
    key = config["id"]
    if checkpoint is not None:
        key += ":" + config["course"]["vq_checkpoint_sha256"]
    runs[key] = {"run_dir": str(run_dir), "config_sha256": hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()).hexdigest()}
    manifest.write_text(json.dumps(runs, indent=2, sort_keys=True) + "\n")
    return run_dir
