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
# # 🧪 Lesson 10 — Snap notes to a learned vocabulary
#
# **Learning objective:** understand vector quantization as choosing symbols for image regions.
#
# Instead of any continuous note, use one of a limited set of learned symbols.
# A VQ-VAE encoder describes each image region as a vector; quantization snaps
# that vector to its nearest codebook entry. The decoder rebuilds the image from
# a grid of those entries. First watch four points snap to three fixed symbols.

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
from course_aiml_autoencoders.course import load_run_summary
from course_aiml_autoencoders.models.quantizer import VectorQuantizer
from course_aiml_autoencoders.course.notebook_views import show_run_figure
from course_aiml_autoencoders.course.notebook_views import plot_codebook_snap

torch.manual_seed(0)

display(Markdown("""
| Tool | What it does in this lesson |
| --- | --- |
| `torch` | Creates tensors and controls gradient tracking. |
| `learn` | Trains a recipe or reuses an exact matching run. |
| `load_run_summary` | Reads recorded validation results and run metadata. |
| `VectorQuantizer` | Replaces an encoder vector with its closest codebook entry. |
| `show_run_figure` | Displays a diagnostic saved by the selected run. |
| `plot_codebook_snap` | Connects each encoder vector to its selected symbol. |
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
# If two nearby vectors choose the same symbol, will the decoder still see their small difference?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# No. Quantization replaces both with the same embedding. That loses detail but creates a discrete vocabulary.
# </details>

# %% [markdown]
# ## 1. 📍 Make a tiny vocabulary

# %%
# Build a quantizer with three two-dimensional entries.
quantizer = VectorQuantizer(codebook_size=3, embedding_dim=2, commitment_weight=0.25)

# Choose three fixed codebook locations.
embeddings = torch.tensor([[-1., -1.], [1., -1.], [0., 1.]])

# Disable gradients while setting up the example.
with torch.no_grad():

    # Set the toy codebook entries.
    quantizer.codebook.weight.copy_(embeddings)

# Show the available symbol locations.
print(embeddings)

# %% [markdown]
# ## 2. 🧲 Snap encoder vectors to symbols
#
# 🧪 **Try** moving the first point toward the right-hand symbol.
# Rerun this cell and the next one to see when its selected index changes.

# %%
# Choose four example encoder vectors.
points = torch.tensor([[-0.9, -0.7], [-0.6, -0.8], [0.9, -0.6], [0.2, 0.7]])

# Put the two feature channels first.
channels_first = points.T

# Reshape the points as a one-row spatial grid.
encoder_grid = channels_first.reshape(1, 2, 1, 4)

# Replace each point with its nearest codebook vector.
quantized, extras = quantizer(encoder_grid)

# Read the chosen symbol indices.
indices = extras["indices"]

# Flatten the grid of indices into a list-shaped tensor.
chosen_indices = indices.flatten()

# Convert the indices to a Python list.
symbols = chosen_indices.tolist()

# Show the symbol chosen for each vector.
print("Chosen symbols:", symbols)

# %% [markdown]
# ## 3. 🗺️ See where each vector snapped

# %%
# Detach the snapped grid for display.
snapped_grid = quantized.detach()

# Move the two feature channels to the last axis.
features_last = snapped_grid.permute(0, 2, 3, 1)

# Recover one row per snapped vector.
chosen = features_last.reshape(-1, 2)

# Draw the movement from each vector to its chosen symbol.
plot_codebook_snap(points, embeddings, chosen)

# %% [markdown]
# The real codebook is learned. Reconstruction trains the encoder/decoder,
# codebook loss moves symbols toward encoder outputs, and commitment loss keeps
# encoder outputs near their chosen symbols. Straight-through learning passes an
# approximate reconstruction gradient through the discrete choice.

# %% [markdown]
# ## 4. 🏋️ Train a VQ-VAE

# %%
# Use the quick training budget.
PROFILE = "quick"

# Run this recipe or reuse matching evidence.
run_dir = learn("recipes/vqvae/vqvae-001-basic.yaml", profile=PROFILE)

# %% [markdown]
# ## 5. 📊 Read the learned codebook statistics

# %%
# Open the run summary.
summary = load_run_summary(run_dir)

# Get the codebook statistics.
usage = summary["codebook"]

# Show the number and diversity of used codes.
print(usage)

# %% [markdown]
# ## 6. 🖼️ Inspect reconstructions

# %%
# Display the diagnostic from this run.
show_run_figure(run_dir, "reconstructions.png")

# %% [markdown]
# ## 7. 🗺️ Inspect token grids

# %%
# Display the diagnostic from this run.
show_run_figure(run_dir, "token-maps.png")

# %% [markdown]
# ## 8. 🎲 Inspect samples from uniform tokens

# %%
# Display the diagnostic from this run.
show_run_figure(run_dir, "uniform-random-token-samples.png")

# %% [markdown]
# Each 7-by-7 position is a code index. Indices name symbols; neighboring numeric
# IDs need not mean similar image features. VQ-VAE learns reusable discrete visual
# tokens, but neither good reconstruction nor a vocabulary teaches which token
# arrangements make plausible new images. That is the prior's job.

# %% [markdown]
# ## ✅ Advancement gate — transfer check
#
# The codebook and decoder are trained. Why can a grid of uniformly random tokens still look incoherent?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Knowing word meanings does not teach sentence probabilities. The model has learned symbols and decoding, not a distribution over coherent spatial arrangements.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# Nearest-neighbor selection is $k=\arg\min_j\|z_e-e_j\|^2$.
# Straight-through uses the snapped value forward and an identity surrogate
# gradient backward; this is biased. Reconstruction gradients reach the encoder,
# codebook loss updates embeddings, and commitment loss updates the encoder.
# For isolated gradient probes, see the [VQ reference](../lessons/10-vqvae.md).
# </details>

# %% [markdown]
# ➡️ **Next:** [Lesson 11](11-codebook.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/11-codebook.ipynb). No worksheet is required.
