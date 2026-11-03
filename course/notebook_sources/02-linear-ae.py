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
# # 🎚️ Lesson 02 — Rebuild a garment from eight sliders
#
# **Learning objective:** understand a linear encoder, decoder, and bottleneck through visible reconstruction.
#
# Imagine sending a picture using just eight numbers. The **encoder** reads
# 784 pixel values and sets eight sliders. The **decoder** mixes eight learned
# patterns according to those settings to rebuild the picture. Training tunes
# both halves so the sliders carry information that helps reconstruction.
#
# Run one cell at a time. At each 🧪 **Try** prompt, change the suggested value
# and rerun the indicated cell. To inspect an intermediate value, put its
# variable name on the last line of its cell and rerun it.

# %% [markdown]
# ## ⚙️ Setup
#
# Run **Set up the course** once per session. It lists the tools you'll use.

# %% tags=["course-setup"]
from pathlib import Path
from runpy import run_path
from subprocess import run

setup_file = Path("course/notebook_sources/_setup.py")
here = Path.cwd()
checkout = next((root for root in (here, *here.parents) if (root / setup_file).is_file()), None)
if checkout is None:
    import google.colab

    checkout = here / "course-aiml-autoencoders"
    if not checkout.exists():
        run(["git", "clone", "--depth", "1", "--branch", "main",
             "https://github.com/tsilva/course-aiml-autoencoders.git", str(checkout)], check=True)
run_path(str(checkout / setup_file))

import torch
from IPython.display import Markdown, display
from course_aiml_autoencoders.course import (
    learn, load_metrics, load_trained_model, plot_metric_history,
)
from course_aiml_autoencoders.course.linear_lesson import prepare_linear_lesson_data
from course_aiml_autoencoders.diagnostics import plot_image_grid, plot_reconstruction_grid
from course_aiml_autoencoders.training import compute_mean_image, constant_reconstruction_errors
from course_aiml_autoencoders.training.evaluator import evaluate_reconstruction_mse

display(Markdown("""
| Tool | What it does in this lesson |
| --- | --- |
| `torch` | Works with image and slider tensors and turns off gradients during inspection. |
| `learn` | Trains the recipe or reuses an exact matching completed run. |
| `load_trained_model` | Restores the trained model and its experiment settings. |
| `prepare_linear_lesson_data` | Restores the same data splits and picks validation examples to display. |
| `plot_image_grid` | Displays images side by side. |
| `plot_reconstruction_grid` | Compares originals with reconstructions and shows squared error. |
| `compute_mean_image` | Averages the training images into one fixed prediction. |
| `constant_reconstruction_errors` | Scores a fixed prediction on each held-out image. |
| `evaluate_reconstruction_mse` | Scores the trained model on the held-out images. |
| `load_metrics` | Reads the loss values recorded during training. |
| `plot_metric_history` | Plots the recorded losses by epoch. |
"""))

# %% [markdown]
# ## 🧪 Predict before training
#
# Will eight sliders that change with each input beat the fixed mean image from
# Lesson 01? What details do you expect to survive?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# The sliders can adapt the reconstruction to each garment. Broad shape and
# brightness should survive more readily than small details, which must share
# the same eight directions.
# </details>
#
# ## 1. 🏋️ Train the linear autoencoder
#
# The quick profile uses eight CPU epochs, 2,048 training images, and 512
# validation images. The first Fashion-MNIST run downloads the dataset. The
# cell prints the run path and measured duration, and reuses exact matching
# evidence when available. Use `PROFILE = "full"` for the original budget or
# pass `rerun=True` to `learn` for a fresh run.

# %%
# Choose the quick training profile.
PROFILE = "quick"

# Train the linear autoencoder or reuse the same experiment.
run_dir = learn("recipes/ae/ae-001-linear.yaml", profile=PROFILE)

# %% [markdown]
# ## 2. 👕 Meet the held-out garments
#
# Load the best checkpoint, then look at garments the encoder will receive.

# %%
# Restore the model and the settings from this run.
model, config = load_trained_model(run_dir)

# Show the two layers of the trained model.
print(model)

# %%
# Load the same splits and choose validation garments for display.
train_loader, validation_loader, images, labels, garment_names = prepare_linear_lesson_data(config)

# Show the encoder's inputs.
_ = plot_image_grid(images, labels=labels, class_names=garment_names, title="Validation garments")

# %% [markdown]
# ## 3. 🎚️ Encode an image into sliders
#
# The encoder reads each image and returns its settings. 🧪 **Try** changing
# `example_index` in the second cell below. Rerun it and look for differences
# between the settings for two garments.

# %%
# Turn off gradient tracking while inspecting the trained model.
with torch.no_grad():

    # Encode every displayed garment.
    slider_settings = model.encode(images)

# Read the number of garments and sliders in the resulting tensor.
slider_shape = slider_settings.shape

# Show the tensor dimensions.
print(f"Slider tensor shape: {slider_shape}")

# %%
# Choose which garment's settings to inspect.
example_index = 0

# Select the corresponding row of slider settings.
example_settings = slider_settings[example_index]

# Show the selected settings.
display(example_settings)

# %% [markdown]
# ## 4. 🖼️ Decode the sliders into images
#
# Decode the settings for each garment. Look at what the model keeps and what
# disappears. The error maps show where pixel brightness differs most.

# %%
# Turn off gradient tracking during reconstruction.
with torch.no_grad():

    # Decode every garment's slider settings into an image.
    reconstructions = model.decode(slider_settings)

# Compare the originals, their reconstructions, and their error maps.
_ = plot_reconstruction_grid(
    images, reconstructions, labels=labels, class_names=garment_names, include_error=True,
)

# %% [markdown]
# ### 🧪 Try changing one slider
#
# Change `slider_offset` and rerun this cell. Which parts of the image respond?
# Set it to zero to check that the two images match. The slider may affect many
# pixels at once.

# %%
# Choose one of the eight sliders.
slider_index = 0

# Choose how far to move it.
slider_offset = 2.0

# Copy the selected settings so the original stays available for comparison.
edited_settings = example_settings.clone()

# Move only the chosen slider.
edited_settings[slider_index] += slider_offset

# Select the original reconstruction for this garment.
original_image = reconstructions[example_index]

# Keep the original reconstruction in a batch of one for plotting.
original_decoding = original_image.unsqueeze(0)

# Give the edited settings a batch dimension.
edited_batch = edited_settings.unsqueeze(0)

# Turn off gradient tracking during decoding.
with torch.no_grad():

    # Decode the edited settings into a new image.
    edited_decoding = model.decode(edited_batch)

# Put the original and edited decodings in display order.
decoding_pair = torch.cat((original_decoding, edited_decoding))

# Show the effect of the slider change.
_ = plot_image_grid(decoding_pair, title="Original decoding (left) and edited slider (right)")

# %% [markdown]
# ## 5. 📊 Compare against the mean-image baseline
#
# Fit the mean image on training garments, then score both predictors on the
# same held-out garments. The few images above help you see *where* the model
# makes errors. The full validation split measures *how much* error it makes.

# %%
# Score on the same CPU device used to load the checkpoint.
device = torch.device("cpu")

# Build the fixed mean image from training garments.
mean_image, _ = compute_mean_image(train_loader, device)

# Show the prediction that the baseline gives every garment.
_ = plot_image_grid(mean_image, title="Mean training image")

# %%
# Score the mean image on each held-out garment.
baseline_errors = constant_reconstruction_errors(validation_loader, mean_image, device)

# Average the scores across garments.
baseline_average = baseline_errors.mean()

# Convert the average to a Python number.
baseline_mse = baseline_average.item()

# Show the fixed predictor's validation score.
print(f"Mean-image validation MSE: {baseline_mse:.4f}")

# %%
# Score the trained model on the same validation garments.
model_scores = evaluate_reconstruction_mse(model, validation_loader, device)

# Get the model's mean squared error.
model_mse = model_scores["mse"]

# Compute the share of baseline error still present.
error_remaining = model_mse / baseline_mse

# Compute the share of baseline error removed.
error_removed = 1 - error_remaining

# Compare the model with the baseline.
print(f"Linear AE validation MSE: {model_mse:.4f} · baseline error removed: {error_removed:.1%}")

# %% [markdown]
# ## 6. 📉 Check what training changed
#
# Compare the training and validation curves. If your model fails to beat the
# mean, look for a curve that is still falling or one where the validation
# error rises while training error falls.

# %%
# Read the loss values saved by the training run.
history = load_metrics(run_dir)

# Choose the two reconstruction curves to compare.
metrics = ["train/reconstruction_loss", "validation/reconstruction_loss"]

# Show how the losses change over the epochs.
_ = plot_metric_history(history, metrics)

# %% [markdown]
# Use a linear autoencoder as a reconstruction baseline. Eight directions can
# preserve coarse structure, while fine details tend to be lost. Individual
# sliders need not correspond to named features such as sleeves or collars.
#
# ## ✅ Advancement gate — transfer check
#
# A smooth blend between a shirt and a shoe appears on screen. Does that
# establish a useful generator?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# A linear decoder creates smooth blends automatically. We still need a way
# to draw plausible new slider settings. Reconstruction alone does not teach
# a distribution to sample from.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# The encoder computes `z = W_e x + b_e` and the decoder computes
# `x_hat = W_d z + b_d`. Their composition is affine. With eight latent
# coordinates, all reconstructions lie in an at-most-eight-dimensional affine
# subspace. Under appropriate optimization, the reconstruction subspace matches
# PCA, although individual latent axes may be rotated or rescaled.
#
# See the [linear AE reference](../lessons/02-linear-ae.md) for further reading.
# </details>
#
# ➡️ **Next:** [Lesson 03](03-capacity.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/03-capacity.ipynb).
