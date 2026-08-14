# Autoencoders: From Bottlenecks to Discrete Latents

This repository contains our experiment-first course on autoencoders,
variational autoencoders, and vector-quantized autoencoders. It pairs concise,
intuition-first lessons with inspectable implementations and controlled
experiments so learners can connect model behavior to evidence.

The step-by-step executable course starts at
[course/README.md](course/README.md). Its canonical lessons are agent-editable
Jupytext `py:percent` files; deterministic `.ipynb` builds are committed for
Jupyter and GitHub readers.

List lessons from the terminal with:

```bash
uv run course-aiml-autoencoders course
uv run course-aiml-autoencoders course 00
```

The progression is deliberate:

1. **AE:** what information survives a bottleneck?
2. **VAE:** how does a prior make latent-space sampling meaningful?
3. **VQ-VAE:** what changes when the representation becomes discrete?

## Quick start

```bash
uv sync --locked
uv run pytest
uv run python scripts/build_notebooks.py --check
uv run jupyter lab course/notebooks/01-mean-baseline.ipynb
uv run course-aiml-autoencoders train recipes/smoke/fake-ae.yaml
uv run course-aiml-autoencoders train recipes/smoke/fake-vae.yaml
uv run course-aiml-autoencoders train recipes/smoke/fake-vqvae.yaml
uv run course-aiml-autoencoders course
uv run course-aiml-autoencoders train recipes/ae/ae-001-linear.yaml
uv run course-aiml-autoencoders study studies/ae/ae-003-latent-capacity.yaml
```

Fashion-MNIST is downloaded into `data/` on first use. Generated runs,
checkpoints, and raw figures are stored in `runs/` and ignored by Git.

## Repository contract

- `src/course_aiml_autoencoders/` contains reusable implementation code.
- `recipes/` contains complete executable configurations.
- `studies/` contains questions, hypotheses, controlled variants, and required
  evidence.
- `reports/` contains conclusions worth preserving.
- `runs/` contains disposable generated artifacts.
- `course/notebook_sources/` contains the canonical executable lessons.
- `course/notebooks/` contains deterministic generated Jupyter notebooks.
- Notebooks probe the package, but never own reusable model or training logic.

Every run writes its resolved configuration, JSONL metrics, summary, best
checkpoint, and a fixed reconstruction grid. A study changes one declared
variable at a time and can repeat variants across seeds.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the learning sequence and
[docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) for the experiment
rules.
