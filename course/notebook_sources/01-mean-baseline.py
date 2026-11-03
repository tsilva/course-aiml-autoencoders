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
# To inspect an intermediate result, put its variable name on the last line
# of the cell and rerun it.

# %% [markdown]
# ## Setup
#
# Run **Set up the course** once per session. It shows a reference table for
# the tools you'll use below. Then go straight to the first image.

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
from course_aiml_autoencoders.course.baseline_lesson import load_fashion_image_splits
from course_aiml_autoencoders.diagnostics import (
    per_example_mse, plot_image_grid, plot_reconstruction_grid,
)

display(Markdown("""
| Tool | What it does in this lesson |
| --- | --- |
| `torch` | Creates and manipulates tensors that hold pixel values. |
| `load_fashion_image_splits` | Loads separate Fashion-MNIST training and validation images. |
| `per_example_mse` | Computes one mean squared pixel error per image. |
| `plot_image_grid` | Displays images side by side. |
| `plot_reconstruction_grid` | Compares original images with their reconstructions and can show where pixels differ. |
"""))

# %% [markdown]
# ## 1. Look at the target
#
# This line of three white pixels is the image we want to reconstruct.

# %%
# Start with a black 5 × 5 image.
target = torch.zeros(1, 1, 5, 5)

# Turn three pixels in the center column white.
target[0, 0, 1:4, 2] = 1

# Show the image we'll try to reconstruct.
_ = plot_image_grid(target, title="Target")

# %% [markdown]
# ## 2. Make two imperfect predictions
#
# First erase the bottom pixel. **Try** changing `missing_row` to 1 or 2
# and see which pixel disappears. Rows are numbered from 0.

# %%
# Choose which row loses its white pixel.
missing_row = 3

# Copy the target so we can edit the prediction independently.
missing = target.clone()

# Turn the chosen pixel black.
missing[0, 0, missing_row, 2] = 0

# Show the shortened line.
_ = plot_image_grid(missing, title="One pixel missing")

# %% [markdown]
# Now move the whole line one column to the right. **Try** a shift of 0,
# 2, or −1.
#
# With the default settings, which prediction do you expect to have more
# mismatched pixels: the shorter line or the shifted line?

# %%
# Choose how many columns to shift.
shift_pixels = 1

# Shift the target along its horizontal axis.
shifted = target.roll(shifts=shift_pixels, dims=-1)

# Put the chosen shift in the plot title.
shift_title = f"Shift of {shift_pixels} columns"

# Show the shifted line.
_ = plot_image_grid(shifted, title=shift_title)

# %% [markdown]
# ## 3. See what MSE counts
#
# Subtract the prediction from the target at each location. Squaring makes
# a missing white pixel and an extra white pixel count equally.
# **Try** changing `prediction` to `missing` and count the bright spots again.

# %%
# Choose which prediction to score.
prediction = shifted

# Subtract the prediction at every pixel.
pixel_difference = target - prediction

# Square the differences so both signs contribute to the error.
squared_error = pixel_difference.square()

# Show where the squared errors occur.
_ = plot_image_grid(squared_error, title="Squared error at each pixel")

# Count the pixels that differ.
mismatch_count = squared_error.count_nonzero()

# Convert the count to a Python number.
mismatches = mismatch_count.item()

# Count all pixels in the target.
pixel_count = target.numel()

# Report how many pixels differ.
print(f"{mismatches} of {pixel_count} pixels differ")

# %% [markdown]
# Average the squared errors across all 25 pixels, including the ones that
# match. That is the **mean squared error**, or MSE. With a one-column shift,
# can you predict the score from the image above?

# %%
# Average the squared errors over every pixel.
mse_tensor = squared_error.mean()

# Convert the average to a Python number for display.
mse = mse_tensor.item()

# Add up the squared errors to show the numerator.
total_error = squared_error.sum()

# Convert the total to a Python number for display.
error_sum = total_error.item()

# Show the calculation as a fraction and a decimal.
print(f"MSE = {error_sum:g} / {pixel_count} = {mse:.2f}")

# %% [markdown]
# Compare the scores side by side. A perfect copy and an all-black image
# give us two more reference points.

# %%
# Create a black image with the same shape as the target.
black_target = torch.zeros_like(target)

# Give each prediction a label.
candidates = {
    "Exact": target,
    "One pixel missing": missing,
    "Shifted": shifted,
    "All black": black_target,
}

# Get the names and images to compare.
named_candidates = candidates.items()

# Score each image against the same target.
for name, candidate in named_candidates:

    # Compute the MSE for this prediction.
    candidate_error = per_example_mse(target, candidate)

    # Convert the single score to a Python number.
    candidate_mse = candidate_error.item()

    # Print the label and score together.
    print(f"{name:>17}  MSE {candidate_mse:.2f}")

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
# Use the smaller split to keep the lesson quick.
PROFILE = "quick"

# Load separate image sets for fitting and scoring.
train_images, validation_images = load_fashion_image_splits(profile=PROFILE)

# Count the training garments.
train_count = len(train_images)

# Count the validation garments.
validation_count = len(validation_images)

# Report how many images can go into the average.
print(f"Training: {train_count:,} images")

# Report how many images are available for scoring.
print(f"Validation: {validation_count:,} images")

# %% [markdown]
# Look at a few training images before averaging them. Which bright regions
# do several garments have in common?

# %%
# Pick the first four training garments.
training_examples = train_images[:4]

# Show them side by side.
_ = plot_image_grid(training_examples, title="Training images")

# %% [markdown]
# ## 5. Average the training images
#
# Imagine stacking the training images like transparencies. At each pixel
# location, average the brightness through the stack. The first tensor axis
# holds the images, so `mean(dim=0)` averages across garments.
#
# **Try** changing `average_count` to 1 or 16. What changes as more garments
# go into the average? Restore `train_count` for the baseline.

# %%
# Start by using all the training images.
average_count = train_count

# Select the images that will contribute to the average.
images_to_average = train_images[:average_count]

# Average each pixel across garments, keeping the image axis.
mean_image = images_to_average.mean(dim=0, keepdim=True)

# Put the chosen image count in the plot title.
average_title = f"Average of {average_count:,} training images"

# Show the resulting mean image.
_ = plot_image_grid(mean_image, title=average_title)

# %% [markdown]
# ## 6. Give every garment the same prediction
#
# Predict the mean image for each garment, regardless of what it looks like.
# **Try** changing `start` to see a different group of validation garments.
# Which parts of the error map change when the prediction stays fixed?

# %%
# Choose the first validation garment to show.
start = 0

# Set the slice boundary four garments later.
stop = start + 4

# Select this group of held-out garments.
examples = validation_images[start:stop]

# Use the same mean image as the prediction for each garment.
example_predictions = mean_image.expand_as(examples)

# Compare the garments, their predictions, and their error maps.
_ = plot_reconstruction_grid(examples, example_predictions, include_error=True)

# %% [markdown]
# ## 7. Score both baselines on the same held-out images
#
# Compare two constant predictions across the whole validation split.
# Predict which one will score lower before running the cell.

# %%
# Make a black image with the same shape as the mean image.
black_image = torch.zeros_like(mean_image)

# Use the black image for every validation garment.
black_predictions = black_image.expand_as(validation_images)

# Score the black prediction separately for each garment.
black_errors = per_example_mse(validation_images, black_predictions)

# Average the black baseline scores across garments.
black_average = black_errors.mean()

# Convert the average to a Python number for display.
black_mse = black_average.item()

# Use the mean image for every validation garment.
mean_predictions = mean_image.expand_as(validation_images)

# Score the mean prediction separately for each garment.
mean_errors = per_example_mse(validation_images, mean_predictions)

# Average the mean baseline scores across garments.
mean_average = mean_errors.mean()

# Convert the average to a Python number for display.
mean_mse = mean_average.item()

# Report the black baseline score.
print(f"All-black MSE:  {black_mse:.4f}")

# Report the mean baseline score on the same garments.
print(f"Mean-image MSE: {mean_mse:.4f}")

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
# Choose a brightness multiplier.
brightness = 1.0

# Scale the mean image's pixel values.
scaled_mean = brightness * mean_image

# Keep each pixel within the valid brightness range.
adjusted_mean = scaled_mean.clamp(0, 1)

# Use the adjusted image for every validation garment.
adjusted_predictions = adjusted_mean.expand_as(validation_images)

# Score the adjusted prediction separately for each garment.
adjusted_errors = per_example_mse(validation_images, adjusted_predictions)

# Average those scores over the validation split.
adjusted_average = adjusted_errors.mean()

# Convert the average to a Python number for display.
adjusted_mse = adjusted_average.item()

# Put the multiplier in the plot title.
brightness_title = f"Mean image × {brightness:g}"

# Show the adjusted image.
_ = plot_image_grid(adjusted_mean, title=brightness_title)

# Report how well it predicts the validation images.
print(f"Validation MSE: {adjusted_mse:.4f}")

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
