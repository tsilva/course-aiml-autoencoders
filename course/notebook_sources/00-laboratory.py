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
# # Lesson 00 — See what an autoencoder does
#
# **Learning objective:** follow an image through an encoder, a bottleneck, and a decoder.
#
# Imagine sending a picture through a tiny note. The encoder writes the note;
# the decoder rebuilds a picture using only that note. Training teaches both sides
# the same shorthand. Today we inspect the system *before* it learns.
#
# You need basic Python, tensors as arrays of numbers, and the idea that training
# adjusts weights to reduce an error. Probability and gradient details are
# introduced when needed. Run cells in order; make a mental prediction, then
# expand its explanation. The course moves from reconstruction to sampling and
# finally to discrete visual tokens.

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
# ## Predict before running
#
# An untrained model already has an encoder and decoder. Will its reconstruction resemble the input?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Usually not: the shape of the system is in place, but the weights have not learned a shared shorthand.
# </details>

# %%
from course_aiml_autoencoders.models import build_model
from course_aiml_autoencoders.data.datasets import dataset_spec

config = load_yaml(ROOT / "recipes/smoke/fake-ae.yaml")
spec = dataset_spec(config["dataset"])
model = build_model(config["model"], spec)
inputs = torch.zeros(2, *spec.input_shape)
inputs[0, :, 5:23, 10:18] = 1
inputs[1, :, 10:18, 5:23] = 1
model.eval()
with torch.no_grad():
    output = model(inputs)
print("Image → note → image:", tuple(inputs.shape), "→", tuple(output.latent.shape), "→", tuple(output.reconstruction.shape))
_ = plot_reconstruction_grid(inputs, output.reconstruction, include_error=True)

# %% [markdown]
# The output has the right shape without having useful content. This is why a
# working command or a finite loss does not prove learning. In Lesson 01 we build
# a deliberately simple competitor; in Lesson 02 a trained model must beat it.
#
# AE: writes a continuous note. VAE: writes a distribution of possible notes.
# VQ-VAE: writes a grid of symbols from a learned vocabulary. Each changes what
# can pass through the bottleneck and how new notes can be generated.

# %% [markdown]
# ## Advancement gate — transfer check
#
# A colleague shows you a model with perfect tensor shapes and no training history. What evidence is still missing?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Evidence that training produces input-dependent reconstructions on held-out images and improves on an input-ignoring baseline.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# For implementation trust checks, run `uv run --frozen python -m pytest`.
# The model returns reconstruction, latent, and specialized extras so one generic
# trainer can serve all three families. Models and objectives own their math;
# notebooks configure experiments and inspect evidence.
# </details>

# %% [markdown]
# **Next:** [Lesson 01](01-mean-baseline.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/01-mean-baseline.ipynb). No worksheet is required.
