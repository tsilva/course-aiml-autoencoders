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
# # Lesson 01 — Give reconstruction a reference
#
# **Learning objective:** explain what MSE notices and why reconstruction needs a baseline.
#
# MSE is a pixel accountant: it squares each brightness mismatch, then averages.
# It knows locations and numbers, not garments. A shifted picture can score poorly
# even when you recognize it. First use a tiny symbol; no training is needed.

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
# Will shifting a symbol one pixel cost more than deleting one bright pixel?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Often yes: a shift creates errors where the symbol used to be and where it moves. Pixel agreement differs from semantic similarity.
# </details>

# %%
target = torch.zeros(1, 1, 5, 5)
target[0, 0, 1:4, 2] = 1
missing = target.clone()
missing[0, 0, 3, 2] = 0
shifted = torch.roll(target, shifts=1, dims=-1)
predictions = torch.cat([target, missing, shifted, torch.zeros_like(target)])
_ = plot_reconstruction_grid(target.expand_as(predictions), predictions, include_error=True)
print("Exact, missing pixel, shifted, all black:", per_example_mse(target.expand_as(predictions), predictions).tolist())

# %% [markdown]
# Now ask an intentionally lazy competitor to predict one image for every input.
# The mean training image is the best constant prediction under squared error.
# Compute it from training images and score it on separate validation images.

# %%
from course_aiml_autoencoders.training import compute_mean_image, constant_reconstruction_errors

config = lesson_config("recipes/ae/ae-001-linear.yaml", profile=PROFILE)
train_loader, validation_loader, spec = build_dataloaders(config["dataset"], config["training"])
inputs, labels = balanced_class_batch(validation_loader, spec.num_classes)
mean_image, _ = compute_mean_image(train_loader, torch.device("cpu"))
black_errors = constant_reconstruction_errors(validation_loader, torch.zeros_like(mean_image), torch.device("cpu"))
mean_errors = constant_reconstruction_errors(validation_loader, mean_image, torch.device("cpu"))
print(f"All-black MSE: {float(black_errors.mean()):.4f}; mean-image MSE: {float(mean_errors.mean()):.4f}")
_ = plot_reconstruction_grid(inputs, mean_image.expand_as(inputs), labels=labels,
    class_names=class_names(config["dataset"]["name"]), include_error=True)

# %% [markdown]
# The mean image ignores the input yet may earn a respectable score because
# background and broad structure repeat. A learned autoencoder must beat this
# competitor and visibly react to different inputs. Even then, reconstruction
# alone does not establish classification or generation quality.

# %% [markdown]
# ## Advancement gate — transfer check
#
# A model predicts the same blurry image for every garment and earns a low MSE. What comparison would expose the problem?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Compare with the mean training image on the same held-out examples. Check whether changing the input changes the reconstruction.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# Under squared error, averaging minimizes the sum of deviations at each pixel.
# MSE is useful for aligned pixel fidelity; it is sensitive to shifts and can
# reward blur. Use a task-specific measure when asking a different question.
# </details>

# %% [markdown]
# **Next:** [Lesson 02](02-linear-ae.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/02-linear-ae.ipynb). No worksheet is required.
