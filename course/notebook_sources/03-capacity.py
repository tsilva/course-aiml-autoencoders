# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lesson 03 — Let the reconstruction sheet bend
#
# **Learning objective:** separate nonlinear geometry from bottleneck size.
#
# A linear decoder builds everything from one flat set of directions. Nonlinear
# layers let the mapping bend and respond differently in different regions.
# Keep the eight-number bottleneck fixed and change only the hidden-layer setup.
# The larger parameter count remains a confound; this tests two architectures,
# not the isolated effect of an activation function.

# %% [markdown]
# ## Setup
#
# Run this cell first. In Colab it fetches the course code automatically;
# locally it uses your checkout. The lessons use short CPU experiments.

# %%
import os
from pathlib import Path
import subprocess
import sys

# Find an existing checkout before fetching one in a fresh Colab session.
_start = Path.cwd().resolve()
_course_root = next((
    candidate for candidate in (_start, *_start.parents)
    if (candidate / "course/curriculum.yaml").is_file()
    and (candidate / "src/course_aiml_autoencoders").is_dir()
), None)
if _course_root is None:
    try:
        import google.colab
    except ImportError as error:
        raise RuntimeError("Open this notebook from the course checkout or in Google Colab.") from error
    _course_root = _start / "course-aiml-autoencoders"
    if not _course_root.exists():
        subprocess.run([
            "git", "clone", "--depth", "1", "--branch", "main",
            "https://github.com/tsilva/course-aiml-autoencoders.git",
            str(_course_root),
        ], check=True)
    if not (_course_root / "course/curriculum.yaml").is_file():
        raise RuntimeError(f"Incomplete course checkout at {_course_root}; rename it and rerun setup.")
os.chdir(_course_root)
_course_src = str(_course_root / "src")
if _course_src not in sys.path:
    sys.path.insert(0, _course_src)
print("Course ready:", _course_root)

# %%
import matplotlib.pyplot as plt
import torch
from IPython.display import Image, display
from course_aiml_autoencoders.config import load_yaml, with_overrides
from course_aiml_autoencoders.course import (
    balanced_class_batch, learn, learn_prior, lesson_config,
    load_metrics, load_run_summary, load_trained_model,
    plot_metric_history, repository_root,
)
from course_aiml_autoencoders.data import build_dataloaders, class_names
from course_aiml_autoencoders.diagnostics import (
    per_example_mse, plot_image_grid, plot_reconstruction_grid,
)

ROOT = repository_root()
torch.manual_seed(0)
PROFILE = "quick"

# %% [markdown]
# The default uses eight CPU epochs on 2,048 training images and 512 held-out
# images. A matching completed run is reused automatically; its exact path and
# measured duration are printed. These short runs expose mechanisms, not settled
# rankings. The first Fashion-MNIST lesson downloads the dataset once.
#
# Set `PROFILE = "full"` for the original training budget; `learn(..., rerun=True)`
# creates fresh evidence. You can return to the prediction while training runs.

# %% [markdown]
# ## Predict before running
#
# With eight sliders in both models, can nonlinear layers still improve reconstruction?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Yes: the mapping can represent more complex structure at the same latent dimension. Improvement is empirical, and added parameters also contribute.
# </details>

# %%
linear_dir = learn("recipes/ae/ae-001-linear.yaml", profile=PROFILE)
nonlinear_dir = learn("recipes/ae/ae-002-nonlinear.yaml", profile=PROFILE)
for name, run_dir in (("linear", linear_dir), ("nonlinear", nonlinear_dir)):
    summary = load_run_summary(run_dir)
    print(name, "MSE:", summary["best_validation_metrics"]["validation/reconstruction_loss"],
          "parameters:", summary["parameter_count"])
    display(Image(filename=str(run_dir / "figures/reconstructions.png")))

# %% [markdown]
# Compare the same validation examples. Did edges or class-specific shape
# improve? A better pixel score supports reconstruction usefulness under this
# setup, not automatic disentanglement or downstream utility.
#
# More latent coordinates are a separate change: they let more information pass.
# A bottleneck of 784 with sufficient flexibility may allow copying, removing
# the pressure to find a compact summary.

# %% [markdown]
# ## Advancement gate — transfer check
#
# A 128-dimensional model beats an eight-dimensional one. Is it the better representation?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# It is better on the measured reconstruction task. Compression, robustness, generation, and downstream usefulness still require separate evidence.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# Run the full latent-size sweep:
# `uv run course-aiml-autoencoders study studies/ae/ae-003-latent-capacity.yaml --seeds 0`.
# Vary only latent size. Expect possible diminishing returns; inspect the result
# before concluding. The original nonlinearity study is also available in
# `studies/ae/ae-002-nonlinearity.yaml`.
# </details>

# %% [markdown]
# **Next:** [Lesson 04](04-regularized-ae.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/04-regularized-ae.ipynb). No worksheet is required.
