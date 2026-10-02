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
# # Lesson 12 — Learn which token arrangements belong together
#
# **Learning objective:** separate learning a visual vocabulary from learning a sampling distribution.
#
# A vocabulary tells us what symbols mean. A prior learns which symbol should
# come next, given the symbols already present. Scanning the token grid row by
# row lets a small GRU learn these dependencies while the VQ encoder and decoder
# stay frozen. Training supplies the true previous tokens; sampling supplies the
# prior's own previous choices.

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
# Will beating a uniform token predictor prove that the prior learned spatial dependencies?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# No. Frequent symbols alone can beat uniform prediction. Compare with a training-frequency unigram baseline as well as inspecting generated arrangements.
# </details>

# %%
vq_run = learn("recipes/vqvae/vqvae-001-basic.yaml", profile=PROFILE)
prior_run = learn_prior("recipes/prior/prior-001-gru.yaml", vq_run, profile=PROFILE)
summary = load_run_summary(prior_run)
print("Uniform CE:", summary["uniform_cross_entropy"],
      "training-frequency unigram CE:", summary["unigram_cross_entropy"],
      "learned-prior validation CE:", summary["best_validation_cross_entropy"])
_ = plot_metric_history(load_metrics(prior_run), ["train/cross_entropy", "validation/cross_entropy"])
display(Image(filename=str(vq_run / "figures/uniform-random-token-samples.png")))
display(Image(filename=str(prior_run / "figures/learned-prior-samples.png")))

# %% [markdown]
# Better validation likelihood than the unigram baseline supports learning
# structure beyond overall symbol frequency, but is not proof of good generated
# images. Inspect arrangements and diversity: sampling consumes its own history
# and can accumulate errors. Short training may reveal only frequency learning.
#
# Prior perplexity measures next-token uncertainty given history; codebook
# perplexity measures assignment diversity. They share a name but answer
# different questions. The prior cannot restore information discarded by the
# frozen VQ representation.

# %% [markdown]
# ## Advancement gate — transfer check
#
# The prior improves dramatically, but generated images still lack fine detail. Which stage might be the bottleneck?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# The VQ encoder/codebook/decoder may have discarded that detail. A better token distribution cannot recover information absent from the representation.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# Raster order gives the factorization
# $p(k_1,\ldots,k_N)=\prod_i p(k_i\mid k_{<i})$.
# Teacher forcing uses true history during training; sampling uses sampled
# history. The small GRU is chosen for clarity, not top-tier image quality.
# Temperature and larger priors are optional investigations; preserve the exact
# paired VQ checkpoint when comparing them.
# </details>

# %% [markdown]
# **Next:** [Lesson 13](13-synthesis.ipynb). No worksheet is required.
