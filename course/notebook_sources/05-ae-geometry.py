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
# # 🧪 Lesson 05 — Reconstruction does not choose new codes
#
# **Learning objective:** distinguish interpolation from sampling.
#
# An AE learns to read notes written by its encoder. Inventing random notes
# asks the decoder to read a distribution it was never trained to expect.
# A smooth path between two valid notes is a different claim from a reliable
# method for drawing new notes.

# %% [markdown]
# ## ⚙️ Setup
#
# Run **Set up the course** once per session to load the tools below.
# To inspect an intermediate value, put its name on the last line of
# its cell and rerun the cell.

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

from IPython.display import Markdown, display
import torch
from course_aiml_autoencoders.course import learn
from course_aiml_autoencoders.course import load_trained_model
from course_aiml_autoencoders.diagnostics import plot_image_grid
from course_aiml_autoencoders.diagnostics.interpolations import linear_interpolation
from course_aiml_autoencoders.course.notebook_views import validation_examples

torch.manual_seed(0)

display(Markdown("""
| Tool | What it does in this lesson |
| --- | --- |
| `torch` | Creates tensors and controls gradient tracking. |
| `learn` | Trains a recipe or reuses an exact matching run. |
| `load_trained_model` | Restores the trained model and its resolved settings. |
| `plot_image_grid` | Displays images side by side. |
| `linear_interpolation` | Creates evenly spaced codes between two endpoints. |
| `validation_examples` | Selects balanced examples from the run's held-out split. |
"""))

# %% [markdown]
# The default uses eight CPU epochs on 2,048 training images and 512 held-out
# images. A matching completed run is reused automatically; its exact path and
# measured duration are printed. These short runs expose mechanisms, not settled
# rankings. The first Fashion-MNIST lesson downloads the dataset once.
#
# Set `PROFILE = "full"` for the original training budget; `learn(..., rerun=True)`
# creates fresh evidence. You can return to the prediction while training runs.

# %% [markdown]
# ## 🔮 Predict before running
#
# Should a smooth interpolation guarantee that random standard-normal codes produce plausible garments?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# No. Continuity gives smooth changes; it says nothing about how probable the visited codes are. The AE has not matched its encoded distribution to a standard normal.
# </details>

# %% [markdown]
# ## 1. 🏋️ Restore the nonlinear experiment

# %%
# Use the quick training budget.
PROFILE = "quick"

# Run this recipe or reuse matching evidence.
run_dir = learn("recipes/ae/ae-002-nonlinear.yaml", profile=PROFILE)

# %% [markdown]
# ## 2. 👕 Choose validation garments

# %%
# Restore the trained model and its settings.
model, config = load_trained_model(run_dir)

# Select balanced validation examples.
inputs = validation_examples(config)

# Display the available endpoints.
_ = plot_image_grid(inputs, title="Available interpolation endpoints")

# %% [markdown]
# ## 3. 🎚️ Encode the garments

# %%
# Disable gradients for inspection.
with torch.no_grad():

    # Encode each displayed garment.
    codes = model.encode(inputs)

# Read the code dimensions.
code_shape = codes.shape

# Show the number and width of the codes.
print(code_shape)

# %% [markdown]
# ## 4. 🛤️ Walk between two codes
#
# 🧪 **Try** changing `end` to another garment index and rerunning this cell.
# Does the walk stay smooth even when the endpoints look different?

# %%
# Choose the first endpoint.
start = 0

# Choose the other endpoint.
end = -1

# Select the first code.
start_code = codes[start]

# Select the last code.
end_code = codes[end]

# Build nine evenly spaced codes.
path = linear_interpolation(start_code, end_code, 9)

# Disable gradients for inspection.
with torch.no_grad():

    # Decode the walk.
    interpolated = model.decode(path)

# Show how the decoded image changes along the walk.
_ = plot_image_grid(interpolated, title="Walk between two encoded examples", max_items=9)

# %% [markdown]
# ## 5. 🎲 Invent codes and decode them

# %%
# Read the code width.
latent_width = codes.shape[1]

# Draw nine standard-normal codes.
random_codes = torch.randn(9, latent_width)

# Disable gradients for inspection.
with torch.no_grad():

    # Decode the invented codes.
    random_images = model.decode(random_codes)

# Show the samples.
_ = plot_image_grid(random_images, title="Invent standard-normal codes", max_items=9)

# %% [markdown]
# Inspect plausibility as well as smoothness. Encoded means, scales, correlations,
# and occupied regions may differ from the convenient standard normal. A few
# plausible random images would not establish a good sampling distribution.
#
# Use the AE for reconstruction or a separately evaluated representation task.
# For generation, we need training pressure or an additional model that tells us
# where valid codes come from. That motivates the VAE.

# %% [markdown]
# ## ✅ Advancement gate — transfer check
#
# You rescale every AE latent coordinate to unit variance. Is generation now solved?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# No. Matching coordinate scales does not match correlations, shapes, or low-density holes. Sampling requires a distribution that models the joint encoded data.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# Plot a PCA projection of encoded validation examples, colored by class.
# A projection helps reveal structure but cannot map every hole in a
# higher-dimensional space. See the [geometry reference](../lessons/05-ae-geometry.md).
# </details>

# %% [markdown]
# ➡️ **Next:** [Lesson 06](06-vae-control.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/06-vae-control.ipynb). No worksheet is required.
