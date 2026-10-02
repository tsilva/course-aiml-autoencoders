<div align="center">
  <img src="./logo.png" alt="Autoencoders: From Bottlenecks to Discrete Latents" width="760" />

  **🧪 Learn autoencoders by predicting, running, and inspecting experiments. 🧪**
</div>

Autoencoders: From Bottlenecks to Discrete Latents is an executable Python
course for learners and ML practitioners who want to understand AEs, VAEs, and
VQ-VAEs through evidence as well as theory. Work through 14 Jupyter lessons,
make one prediction, inspect one small experiment, and answer one transfer
check per lesson. Short CPU runs are reused automatically; derivations, full
sweeps, and research reports are optional.

The curriculum progresses from deterministic bottlenecks to continuous and
discrete latent spaces. Recipes define complete runs, studies vary one declared
factor, generated runs hold evidence, and reports preserve conclusions. Start
with the [course guide](course/README.md).

## Install

```bash
git clone https://github.com/tsilva/course-aiml-autoencoders.git
cd course-aiml-autoencoders
uv sync --frozen
```

Open the first laboratory lesson:

```bash
uv run --frozen python -m jupyterlab course/notebooks/00-laboratory.ipynb
```

Jupyter prints the local URL to open in your browser.

## Commands

```bash
uv run --frozen course-aiml-autoencoders course                          # list lessons
uv run --frozen course-aiml-autoencoders course 00                       # show one lesson
uv run --frozen course-aiml-autoencoders train recipes/smoke/fake-ae.yaml --device cpu
uv run --frozen course-aiml-autoencoders train recipes/ae/ae-001-linear.yaml
uv run --frozen course-aiml-autoencoders study studies/ae/ae-003-latent-capacity.yaml --seeds 0
uv run --frozen course-aiml-autoencoders inspect <RUN_DIR>                # inspect evidence
uv run --frozen python scripts/build_notebooks.py --check                 # verify notebooks
uv run --frozen python -m pytest                                          # run tests
```

## Notes

- Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/) are required.
- Fashion-MNIST downloads into ignored `data/` storage on first use. The fake
  AE, VAE, and VQ-VAE smoke recipes exercise the pipeline without that dataset.
- Each ignored `runs/` directory records the resolved configuration, metrics,
  best checkpoint, summary, diagnostics, and figures needed to inspect a claim.
- Canonical lessons live in Jupytext sources. Generated notebooks are
  deterministic and output-free; training evidence belongs in `runs/`.
- The core path uses up to eight CPU epochs on 2,048 training and 512 validation
  images, with exact run reuse and measured timings. Set the notebook profile
  to `full` for the original recipes.
- Validation comes from a fixed holdout of the training split; the official test
  split is reserved for optional final confirmation.
- Use one seed while exploring and multiple seeds only before promoting an
  important close numerical result into an optional report.
- See the [learning roadmap](docs/ROADMAP.md),
  [experiment protocol](docs/EXPERIMENT_PROTOCOL.md), and
  [troubleshooting guide](course/TROUBLESHOOTING.md) for deeper guidance.

## Architecture

![Course and experiment architecture](./architecture.png)
