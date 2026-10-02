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
# # Lesson 10 — Snap notes to a learned vocabulary
#
# **Learning objective:** understand vector quantization as choosing symbols for image regions.
#
# Instead of any continuous note, use one of a limited set of learned symbols.
# A VQ-VAE encoder describes each image region as a vector; quantization snaps
# that vector to its nearest codebook entry. The decoder rebuilds the image from
# a grid of those entries. First watch four points snap to three fixed symbols.

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
# If two nearby vectors choose the same symbol, will the decoder still see their small difference?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# No. Quantization replaces both with the same embedding. That loses detail but creates a discrete vocabulary.
# </details>

# %%
from course_aiml_autoencoders.models.quantizer import VectorQuantizer

quantizer = VectorQuantizer(codebook_size=3, embedding_dim=2, commitment_weight=0.25)
with torch.no_grad():
    quantizer.codebook.weight.copy_(torch.tensor([[-1., -1.], [1., -1.], [0., 1.]]))
points = torch.tensor([[-0.9, -0.7], [-0.6, -0.8], [0.9, -0.6], [0.2, 0.7]])
quantized, extras = quantizer(points.T.reshape(1, 2, 1, 4))
chosen = quantized.detach().permute(0, 2, 3, 1).reshape(-1, 2)
plt.scatter(points[:, 0], points[:, 1], label="encoder vectors")
embeddings = quantizer.codebook.weight.detach()
plt.scatter(embeddings[:, 0], embeddings[:, 1], marker="s", s=100, label="codebook")
for point, symbol in zip(points, chosen):
    plt.plot([point[0], symbol[0]], [point[1], symbol[1]], color="gray")
plt.legend()
plt.axis("equal")
plt.title("Snapping to symbols; illustrative fixed codebook")
plt.show()
print("Chosen symbols:", extras["indices"].flatten().tolist())

# %% [markdown]
# The real codebook is learned. Reconstruction trains the encoder/decoder,
# codebook loss moves symbols toward encoder outputs, and commitment loss keeps
# encoder outputs near their chosen symbols. Straight-through learning passes an
# approximate reconstruction gradient through the discrete choice.

# %%
run_dir = learn("recipes/vqvae/vqvae-001-basic.yaml", profile=PROFILE)
summary = load_run_summary(run_dir)
print(summary["codebook"])
for figure in ("reconstructions.png", "token-maps.png", "uniform-random-token-samples.png"):
    display(Image(filename=str(run_dir / "figures" / figure)))

# %% [markdown]
# Each 7-by-7 position is a code index. Indices name symbols; neighboring numeric
# IDs need not mean similar image features. VQ-VAE learns reusable discrete visual
# tokens, but neither good reconstruction nor a vocabulary teaches which token
# arrangements make plausible new images. That is the prior's job.

# %% [markdown]
# ## Advancement gate — transfer check
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
# **Next:** [Lesson 11](11-codebook.ipynb). No worksheet is required.
