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
# # Lesson 04 — Learn to remove noise
#
# **Learning objective:** distinguish reconstructing an input from recovering its clean target.
#
# Give an ordinary AE a noisy image and copying some noise may help its task.
# Give a denoising AE the same noisy image but score against the clean image,
# and copying noise is penalized. Change only training corruption; hold the
# architecture, clean targets, split, optimizer, and evaluation noise fixed.

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
# Which model should better recover a clean image from noise? Must it also win on clean inputs?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# The denoising model is trained for noisy-input recovery. It can trade some clean-input fidelity for robustness, so clean MSE alone cannot answer the denoising question.
# </details>

# %%
recipe = load_yaml(ROOT / "recipes/ae/ae-003-denoising.yaml")
clean_recipe = with_overrides(recipe, {"training.input_corruption": None})
clean_recipe["id"] = "ae/ae-003-denoising-clean-control"
clean_dir = learn(clean_recipe, profile=PROFILE)
denoising_dir = learn(recipe, profile=PROFILE)
for name, run_dir in (("clean-trained", clean_dir), ("denoising", denoising_dir)):
    metrics = load_run_summary(run_dir)["best_validation_metrics"]
    print(name, "clean MSE:", metrics["validation/reconstruction_loss"],
          "noisy-input → clean-target MSE:", metrics["validation/corrupted_mse"],
          "unprocessed noisy-input MSE:", metrics["validation/noisy_input_mse"])
    display(Image(filename=str(run_dir / "figures/corrupted-input-reconstructions.png")))

# %% [markdown]
# Read the four rows: clean target, noisy input, reconstruction, clean-target
# error. Both models see identical validation corruption. The noise-only score
# also shows how much improvement comes from processing the image at all.
#
# If the short experiment contradicts the prediction, check whether both models
# learned enough before claiming denoising is ineffective. Use this approach for
# noise resembling the training corruption; robustness to a new noise type is a
# separate question.

# %% [markdown]
# ## Advancement gate — transfer check
#
# A denoiser improves clean-input MSE but leaves noisy images untouched. Has it demonstrated denoising?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# No. Score noisy-input reconstructions against the clean targets on the same corruptions used for the control. Clean-input improvement answers another question.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# Sparse AEs add a cost for latent activity rather than changing input noise:
# `L = reconstruction + lambda * mean(abs(z))`. Smaller mean activity alone does
# not prove more zero activations or useful sparsity; latent rescaling can also
# reduce that number. Inspect activity patterns and decoder weights.
#
# Full studies: `studies/ae/ae-004-denoising.yaml` and
# `studies/ae/ae-005-sparsity.yaml`. Run them only if these extensions answer your
# next question.
# </details>

# %% [markdown]
# **Next:** [Lesson 05](05-ae-geometry.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/05-ae-geometry.ipynb). No worksheet is required.
