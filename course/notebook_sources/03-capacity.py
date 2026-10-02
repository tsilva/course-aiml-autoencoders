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
# **Next:** [Lesson 04](04-regularized-ae.ipynb). No worksheet is required.
