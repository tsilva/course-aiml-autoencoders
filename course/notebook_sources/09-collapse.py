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
# # Lesson 09 — Check whether the decoder uses the note
#
# **Learning objective:** diagnose latent dependence before trying collapse remedies.
#
# A decoder can appear to work while ignoring the note. To test dependence,
# compare normal reconstruction with decoding one fixed zero note for every
# input. The fixed-note output is a deliberately broken control: it illustrates
# loss of information and does not prove the trained model collapsed.

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
# If every input receives the same note, can the decoder reconstruct their different details?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# A deterministic decoder receives identical inputs and produces identical outputs. Normal reconstructions should preserve more input-specific information if the learned latent is used.
# </details>

# %%
run_dir = learn("recipes/vae/vae-001-basic.yaml", profile=PROFILE)
model, config = load_trained_model(run_dir)
_, validation_loader, spec = build_dataloaders(config["dataset"], config["training"])
inputs, labels = balanced_class_batch(validation_loader, spec.num_classes)
with torch.no_grad():
    output = model(inputs)
    fixed_note = model.decode(torch.zeros_like(output.latent))
_ = plot_reconstruction_grid(inputs, output.reconstruction, include_error=True)
_ = plot_reconstruction_grid(inputs, fixed_note, include_error=True)
print("Normal MSE:", float(per_example_mse(inputs, output.reconstruction).mean()),
      "fixed-note MSE:", float(per_example_mse(inputs, fixed_note).mean()))
display(Image(filename=str(run_dir / "figures/kl-per-dimension.png")))

# %% [markdown]
# Posterior collapse combines weak input information in the encoder with a
# decoder that does not use it. Low KL alone is insufficient. KL thresholds also
# can count prior mismatch unrelated to input information.
#
# Warm-up gradually adds KL pressure so reconstruction can establish a useful
# path first. Free bits removes the incentive to compress a dimension below an
# allowance. Neither guarantees useful information or deserves credit for fixing
# a control that never collapsed.

# %% [markdown]
# ## Advancement gate — transfer check
#
# Run A has low KL and distinct, accurate reconstructions; Run B has low KL and constant reconstructions. Which deserves a collapse investigation?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Run B. In A, investigate whether little information is being used efficiently. Confirm B with latent interventions and posterior variation; the shared low KL number is not enough.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# Free bits optimizes a per-dimension floor $\max(\lambda, KL_j)$.
# Below the floor, the compression term supplies no further gradient; it does not
# force a dimension to encode information. Raw KL and optimized effective KL can
# differ. Run `studies/vae/vae-002-collapse-remedies.yaml` to compare immediate KL,
# warm-up, and free bits. The small decoder may not collapse: report that honestly.
# </details>

# %% [markdown]
# **Next:** [Lesson 10](10-vqvae.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/10-vqvae.ipynb). No worksheet is required.
