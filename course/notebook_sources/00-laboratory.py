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
# # 🧪 Lesson 00 — See what an autoencoder does
#
# **Learning objective:** follow an image through an encoder, a bottleneck, and a decoder.
#
# Imagine sending a picture through a tiny note. The encoder writes the note;
# the decoder rebuilds a picture using only that note. Training teaches both sides
# the same shorthand. Today we inspect the system *before* it learns.
#
# You need basic Python, tensors as arrays of numbers, and the idea that training
# adjusts weights to reduce an error. Probability and gradient details are
# introduced when needed. Run cells in order; make a mental prediction, then
# expand its explanation. The course moves from reconstruction to sampling and
# finally to discrete visual tokens.

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
from course_aiml_autoencoders.diagnostics import plot_image_grid
from course_aiml_autoencoders.diagnostics import plot_reconstruction_grid
from course_aiml_autoencoders.course.notebook_views import untrained_example_model

torch.manual_seed(0)

display(Markdown("""
| Tool | What it does in this lesson |
| --- | --- |
| `torch` | Creates tensors and controls gradient tracking. |
| `plot_image_grid` | Displays images side by side. |
| `plot_reconstruction_grid` | Compares inputs, reconstructions, and pixel errors. |
| `untrained_example_model` | Builds a small model with untrained encoder and decoder weights. |
"""))

# %% [markdown]
# ## 🔮 Predict before running
#
# An untrained model already has an encoder and decoder. Will its reconstruction resemble the input?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Usually not: the shape of the system is in place, but the weights have not learned a shared shorthand.
# </details>

# %% [markdown]
# ## 1. 🎯 Make two toy images

# %%
# Build an encoder and decoder with untrained weights.
model, image_shape = untrained_example_model()

# Allocate two black images.
inputs = torch.zeros(2, *image_shape)

# Draw a vertical bar in the first image.
inputs[0, :, 5:23, 10:18] = 1

# Draw a horizontal bar in the second image.
inputs[1, :, 10:18, 5:23] = 1

# Display the two images.
_ = plot_image_grid(inputs, title="Toy inputs")

# %% [markdown]
# ## 2. ✉️ Encode the images

# %%
# Turn off gradient tracking for the demonstration.
with torch.no_grad():

    # Encode both bars.
    notes = model.encode(inputs)

# Read the input dimensions.
input_shape = inputs.shape

# Read the note dimensions.
note_shape = notes.shape

# Compare the image and note shapes.
print(f"Images {input_shape} → notes {note_shape}")

# %% [markdown]
# ## 3. 🖼️ Decode the notes
#
# 🧪 **Try** changing the first bar's slice and rerunning all three cells.
# Check whether changing the input is enough to produce a useful reconstruction.

# %%
# Turn off gradient tracking during decoding.
with torch.no_grad():

    # Decode both notes.
    reconstructions = model.decode(notes)

# Read the output dimensions.
reconstruction_shape = reconstructions.shape

# Show the output dimensions.
print(f"Reconstructed images: {reconstruction_shape}")

# Show the inputs and the untrained outputs.
_ = plot_reconstruction_grid(inputs, reconstructions, include_error=True)

# %% [markdown]
# The output has the right shape without having useful content. This is why a
# working command or a finite loss does not prove learning. In Lesson 01 we build
# a deliberately simple competitor; in Lesson 02 a trained model must beat it.
#
# AE: writes a continuous note. VAE: writes a distribution of possible notes.
# VQ-VAE: writes a grid of symbols from a learned vocabulary. Each changes what
# can pass through the bottleneck and how new notes can be generated.

# %% [markdown]
# ## ✅ Advancement gate — transfer check
#
# A colleague shows you a model with perfect tensor shapes and no training history. What evidence is still missing?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Evidence that training produces input-dependent reconstructions on held-out images and improves on an input-ignoring baseline.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# For implementation trust checks, run `uv run --frozen python -m pytest`.
# The model returns reconstruction, latent, and specialized extras so one generic
# trainer can serve all three families. Models and objectives own their math;
# notebooks configure experiments and inspect evidence.
# </details>

# %% [markdown]
# ➡️ **Next:** [Lesson 01](01-mean-baseline.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/01-mean-baseline.ipynb). No worksheet is required.
