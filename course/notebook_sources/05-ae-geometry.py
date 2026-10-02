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
# # Lesson 05 — Reconstruction does not choose new codes
#
# **Learning objective:** distinguish interpolation from sampling.
#
# An AE learns to read notes written by its encoder. Inventing random notes
# asks the decoder to read a distribution it was never trained to expect.
# A smooth path between two valid notes is a different claim from a reliable
# method for drawing new notes.

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
# Should a smooth interpolation guarantee that random standard-normal codes produce plausible garments?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# No. Continuity gives smooth changes; it says nothing about how probable the visited codes are. The AE has not matched its encoded distribution to a standard normal.
# </details>

# %%
from course_aiml_autoencoders.diagnostics.interpolations import linear_interpolation

run_dir = learn("recipes/ae/ae-002-nonlinear.yaml", profile=PROFILE)
model, config = load_trained_model(run_dir)
_, validation_loader, spec = build_dataloaders(config["dataset"], config["training"])
inputs, labels = balanced_class_batch(validation_loader, spec.num_classes)
with torch.no_grad():
    output = model(inputs)
    path = linear_interpolation(output.latent[0], output.latent[-1], 9)
    interpolated = model.decode(path)
    random_images = model.decode(torch.randn(9, config["model"]["latent_dim"]))
_ = plot_image_grid(interpolated, title="Walk between two encoded examples", max_items=9)
_ = plot_image_grid(random_images, title="Invent standard-normal codes", max_items=9)

# %% [markdown]
# Inspect plausibility as well as smoothness. Encoded means, scales, correlations,
# and occupied regions may differ from the convenient standard normal. A few
# plausible random images would not establish a good sampling distribution.
#
# Use the AE for reconstruction or a separately evaluated representation task.
# For generation, we need training pressure or an additional model that tells us
# where valid codes come from. That motivates the VAE.

# %% [markdown]
# ## Advancement gate — transfer check
#
# You rescale every AE latent coordinate to unit variance. Is generation now solved?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# No. Matching coordinate scales does not match correlations, shapes, or low-density holes. Sampling requires a distribution that models the joint encoded data.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# Plot a PCA projection of encoded validation examples, colored by class.
# A projection helps reveal structure but cannot map every hole in a
# higher-dimensional space. See the [geometry reference](../lessons/05-ae-geometry.md).
# </details>

# %% [markdown]
# **Next:** [Lesson 06](06-vae-control.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/06-vae-control.ipynb). No worksheet is required.
