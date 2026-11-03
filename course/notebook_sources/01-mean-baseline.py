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
# How can a model get a decent reconstruction score by handing out the same
# blurry picture every time? Start with a five-by-five image and build up to
# a baseline for real garments.
#
# Run one cell at a time. Where you see **Try**, edit the suggested value and
# rerun that cell. If you change an earlier result, rerun the cells that use it.

# %% [markdown]
# ## Setup
#
# Run the collapsed setup cell, then the imports below it. You only need
# to do this once per session.

# %% tags=["course-setup"]

# %%
import torch
from course_aiml_autoencoders.course.baseline_lesson import load_fashion_image_splits
from course_aiml_autoencoders.diagnostics import (
    per_example_mse, plot_image_grid, plot_reconstruction_grid,
)

# %% [markdown]
# ## 1. Look at the target
#
# This line of three white pixels is the image we want to reconstruct.

# %%
target = torch.zeros(1, 1, 5, 5)
target[0, 0, 1:4, 2] = 1
_ = plot_image_grid(target, title="Target")

# %% [markdown]
# ## 2. Make two imperfect predictions
#
# First erase the bottom pixel. **Try** changing `missing_row` to 1 or 2
# and see which pixel disappears. Rows are numbered from 0.

# %%
missing_row = 3
missing = target.clone()
missing[0, 0, missing_row, 2] = 0
_ = plot_image_grid(missing, title="One pixel missing")

# %% [markdown]
# Now move the whole line one column to the right. **Try** a shift of 0,
# 2, or −1.
#
# With the default settings, which prediction do you expect to have more
# mismatched pixels: the shorter line or the shifted line?

# %%
shift_pixels = 1
shifted = target.roll(shifts=shift_pixels, dims=-1)
_ = plot_image_grid(shifted, title=f"Shift of {shift_pixels} columns")

# %% [markdown]
# ## 3. See what MSE counts
#
# Subtract the prediction from the target at each location. Squaring makes
# a missing white pixel and an extra white pixel count equally.
# **Try** changing `prediction` to `missing` and count the bright spots again.

# %%
prediction = shifted
squared_error = (target - prediction).square()
_ = plot_image_grid(squared_error, title="Squared error at each pixel")
print(f"{squared_error.count_nonzero().item()} of {target.numel()} pixels differ")

# %% [markdown]
# Average the squared errors across all 25 pixels, including the ones that
# match. That is the **mean squared error**, or MSE. With a one-column shift,
# can you predict the score from the image above?

# %%
mse = squared_error.mean().item()
print(f"MSE = {squared_error.sum().item():g} / {target.numel()} = {mse:.2f}")

# %% [markdown]
# Compare the scores side by side. A perfect copy and an all-black image
# give us two more reference points.

# %%
candidates = {
    "Exact": target,
    "One pixel missing": missing,
    "Shifted": shifted,
    "All black": torch.zeros_like(target),
}
for name, candidate in candidates.items():
    print(f"{name:>17}  MSE {per_example_mse(target, candidate).item():.2f}")

# %% [markdown]
# <details>
# <summary>Why does the shifted line score worse?</summary>
#
# With the default settings, a shift leaves three pixels missing and adds
# three elsewhere. Erasing one pixel creates only one mismatch. MSE compares
# brightness at fixed locations, so a recognizable shape can still score poorly.
# </details>

# %% [markdown]
# ## 4. Load real garments
#
# We will build one prediction from the training images and judge it on
# different garments from the validation split. This keeps the images used
# to fit the prediction separate from the ones used to score it.

# %%
PROFILE = "quick"
train_images, validation_images = load_fashion_image_splits(profile=PROFILE)
print(f"Training: {len(train_images):,} images")
print(f"Validation: {len(validation_images):,} images")

# %% [markdown]
# Look at a few training images before averaging them. Which bright regions
# do several garments have in common?

# %%
_ = plot_image_grid(train_images[:4], title="Training images")

# %% [markdown]
# ## 5. Average the training images
#
# Imagine stacking the training images like transparencies. At each pixel
# location, average the brightness through the stack. The first tensor axis
# holds the images, so `mean(dim=0)` averages across garments.
#
# **Try** changing `average_count` to 1 or 16. What changes as more garments
# go into the average? Restore `len(train_images)` for the baseline.

# %%
average_count = len(train_images)
mean_image = train_images[:average_count].mean(dim=0, keepdim=True)
_ = plot_image_grid(mean_image, title=f"Average of {average_count:,} training images")

# %% [markdown]
# ## 6. Give every garment the same prediction
#
# Predict the mean image for each garment, regardless of what it looks like.
# **Try** changing `start` to see a different group of validation garments.
# Which parts of the error map change when the prediction stays fixed?

# %%
start = 0
examples = validation_images[start:start + 4]
_ = plot_reconstruction_grid(examples, mean_image.expand_as(examples), include_error=True)

# %% [markdown]
# ## 7. Score both baselines on the same held-out images
#
# Compare two constant predictions across the whole validation split.
# Predict which one will score lower before running the cell.

# %%
black_image = torch.zeros_like(mean_image)
black_mse = per_example_mse(validation_images, black_image.expand_as(validation_images)).mean()
mean_mse = per_example_mse(validation_images, mean_image.expand_as(validation_images)).mean()
print(f"All-black MSE:  {black_mse.item():.4f}")
print(f"Mean-image MSE: {mean_mse.item():.4f}")

# %% [markdown]
# The mean image can score well because the garments share a dark background
# and tend to occupy the middle of the frame. Under squared error, averaging
# the training images gives the best single fixed prediction on that set.
# An autoencoder should beat this baseline on held-out images and visibly
# respond to changes in its input.
#
# **Try** two fixed brightness scales, such as 0.5 and 1.5. Predict what
# each will do to the score, then rerun just this cell to see the effect.

# %%
brightness = 1.0
adjusted_mean = (brightness * mean_image).clamp(0, 1)
adjusted_mse = per_example_mse(validation_images, adjusted_mean.expand_as(validation_images)).mean()
_ = plot_image_grid(adjusted_mean, title=f"Mean image × {brightness:g}")
print(f"Validation MSE: {adjusted_mse.item():.4f}")

# %% [markdown]
# ## Advancement gate — transfer check
#
# A model predicts the same blurry image for every garment and earns a low
# MSE. What comparison would expose the problem?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Compare against the mean training image on the same held-out examples.
# Then swap the input and look for changes in the reconstruction. An
# input-independent prediction can score well by capturing shared structure.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: why the mean wins for squared error</summary>
#
# Consider one pixel across all the training images. Moving a candidate
# brightness away from the mean increases its average squared error by the
# square of that move. The deviations around the mean sum to zero. Apply
# that argument at each location to get the mean image.
# </details>

# %% [markdown]
# **Next:** [Lesson 02](02-linear-ae.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/02-linear-ae.ipynb).
