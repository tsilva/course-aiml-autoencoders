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
# # Lesson 13 — Choose a representation for a task
#
# **Learning objective:** select a model from required behavior and diagnose contradictory evidence.
#
# Choose the mechanism the task needs: an AE for input-conditioned
# reconstruction, a VAE for regularized continuous sampling, or a VQ-VAE for
# visual tokens plus a learned prior for generation. Finish with one choice and
# one unfamiliar failure case. A report, seed sweep, and extra application are
# optional extensions.

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
# A product needs compact discrete image tokens for another model. Which family fits, and what evidence would you inspect?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# VQ-VAE. Check reconstruction, token maps, and global vocabulary usage. If the product also generates images, evaluate a prior paired with that exact representation.
# </details>

# %%
# These calls reuse the same configurations inspected in earlier lessons.
finalists = {
    "AE": learn("recipes/ae/ae-002-nonlinear.yaml", profile=PROFILE),
    "VAE": learn("recipes/vae/vae-001-basic.yaml", profile=PROFILE),
    "VQ-VAE": learn("recipes/vqvae/vqvae-001-basic.yaml", profile=PROFILE),
}
for family, run_dir in finalists.items():
    print(family, "exact run:", run_dir)
    display(Image(filename=str(run_dir / "figures/reconstructions.png")))

# %% [markdown]
# The pictures expose representation tradeoffs; they do not define a fair
# cross-family leaderboard. Architectures and training objectives differ.
#
# | Need | Starting choice | First evidence |
# |---|---|---|
# | Recover clean images from matched noise | Denoising AE | Noisy-input output scored against clean targets |
# | Generate using a known continuous prior | VAE | Reconstruction, latent dependence, prior samples |
# | Encode reusable discrete visual symbols | VQ-VAE | Reconstruction, token maps, global code usage |
# | Generate coherent token arrangements | VQ-VAE + prior | Validation likelihood and sampled arrangements |
#
# Explain your chosen mechanism in one sentence, then name the evidence that
# would make you change your choice. No written worksheet is required.

# %% [markdown]
# ## Advancement gate — transfer check
#
# An unfamiliar VAE has high KL, excellent reconstruction, and poor prior samples. What single next check would distinguish useful input information from prior mismatch?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Inspect the encoded aggregate distribution versus the sampling prior, or test a controlled change in beta. High KL alone does not mean useful representation capacity; it also includes aggregate mismatch. Avoid changing architecture, data, and beta together.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# For a durable claim, run only the relevant two variants over seeds 0, 1,
# and 2, report mean and variation, and inspect seed-specific failures. Use
# [WORKSHEET.md](../WORKSHEET.md) only when preserving a research conclusion.
#
# For a downstream extension, freeze the encoder and compare a classifier or
# retrieval probe on latent features with raw pixels and PCA. Keep identical
# splits and assess the task rather than assuming reconstruction implies utility.
#
# Use `build_test_dataloader` for final confirmation after choosing the model.
# The official test split is reserved; routine training and selection use a fixed
# holdout from the original training split. Do not tune again on test results.
# </details>

# %% [markdown]
# **Finished:** revisit the one concept you could not explain, or choose an optional extension in the [course guide](../README.md).
