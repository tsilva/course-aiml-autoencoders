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
# # Lesson 02 — Rebuild a garment from eight sliders
#
# **Learning objective:** understand a linear encoder, decoder, and bottleneck through visible reconstruction.
#
# The encoder turns 784 pixel values into eight slider settings. The decoder
# mixes eight learned patterns to rebuild the image. These are broad patterns,
# not eight selected pixels. Training teaches the encoder and decoder a shared
# language; the bottleneck forces them to prioritize what that language can express.

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
# Will eight input-dependent sliders beat one mean poster? What details should survive?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Usually the model beats the mean poster by tailoring the sliders to each input. Broad shape and brightness should survive; fine detail is expensive to express with eight linear directions.
# </details>

# %%
from course_aiml_autoencoders.training import compute_mean_image, constant_reconstruction_errors
from course_aiml_autoencoders.training.evaluator import evaluate_reconstruction_mse

run_dir = learn("recipes/ae/ae-001-linear.yaml", profile=PROFILE)
model, config = load_trained_model(run_dir)
train_loader, validation_loader, spec = build_dataloaders(config["dataset"], config["training"])
inputs, labels = balanced_class_batch(validation_loader, spec.num_classes)
with torch.no_grad():
    output = model(inputs)
_ = plot_reconstruction_grid(inputs, output.reconstruction, labels=labels,
    class_names=class_names(config["dataset"]["name"]), include_error=True)
mean_image, _ = compute_mean_image(train_loader, torch.device("cpu"))
baseline = float(constant_reconstruction_errors(validation_loader, mean_image, torch.device("cpu")).mean())
model_mse = evaluate_reconstruction_mse(model, validation_loader, torch.device("cpu"))["mse"]
print(f"Mean poster: {baseline:.4f}; linear AE: {model_mse:.4f}; error removed: {1-model_mse/baseline:.1%}")
_ = plot_metric_history(load_metrics(run_dir), ["train/reconstruction_loss", "validation/reconstruction_loss"])

# %% [markdown]
# Look for input-specific silhouettes and the details lost along the way. If it
# fails to beat the mean, check the learning curves before interpreting its latent
# space. A short run need not reach the best possible solution.
#
# Use a linear AE as a transparent reconstruction baseline. Its flat set of
# directions limits image complexity; latent coordinates need not correspond to
# named concepts, and no distribution of valid new slider settings is learned.

# %% [markdown]
# ## Advancement gate — transfer check
#
# A smooth blend between a shirt and a shoe appears on screen. Does that establish a useful generator?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# No. A linear decoder creates smooth blends automatically. We still need a way to draw plausible new latent codes; reconstruction does not teach a sampling distribution.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# The encoder computes `z = W_e x + b_e`; the decoder computes
# `x_hat = W_d z + b_d`. Their composition is affine. With eight latent
# coordinates, all reconstructions lie in an at-most-eight-dimensional affine
# subspace. Under appropriate optimization, the reconstruction subspace matches
# PCA, although individual latent axes may be rotated or rescaled.
#
# Explore decoder directions, a one-slider traversal, and interpolation using the
# [linear AE reference](../lessons/02-linear-ae.md).
# </details>

# %% [markdown]
# **Next:** [Lesson 03](03-capacity.ipynb). No worksheet is required.
