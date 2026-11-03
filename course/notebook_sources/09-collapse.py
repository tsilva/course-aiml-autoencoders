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
# # 🧪 Lesson 09 — Check whether the decoder uses the note
#
# **Learning objective:** diagnose latent dependence before trying collapse remedies.
#
# A decoder can appear to work while ignoring the note. To test dependence,
# compare normal reconstruction with decoding one fixed zero note for every
# input. The fixed-note output is a deliberately broken control: it illustrates
# loss of information and does not prove the trained model collapsed.

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
from course_aiml_autoencoders.diagnostics import plot_reconstruction_grid
from course_aiml_autoencoders.diagnostics import per_example_mse
from course_aiml_autoencoders.course.notebook_views import validation_examples
from course_aiml_autoencoders.course.notebook_views import show_run_figure

torch.manual_seed(0)

display(Markdown("""
| Tool | What it does in this lesson |
| --- | --- |
| `torch` | Creates tensors and controls gradient tracking. |
| `learn` | Trains a recipe or reuses an exact matching run. |
| `load_trained_model` | Restores the trained model and its resolved settings. |
| `plot_reconstruction_grid` | Compares inputs, reconstructions, and pixel errors. |
| `per_example_mse` | Computes one mean squared pixel error per image. |
| `validation_examples` | Selects balanced examples from the run's held-out split. |
| `show_run_figure` | Displays a diagnostic saved by the selected run. |
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
# If every input receives the same note, can the decoder reconstruct their different details?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# A deterministic decoder receives identical inputs and produces identical outputs. Normal reconstructions should preserve more input-specific information if the learned latent is used.
# </details>

# %% [markdown]
# ## 1. 🏋️ Restore the standard VAE

# %%
# Use the quick training budget.
PROFILE = "quick"

# Run this recipe or reuse matching evidence.
run_dir = learn("recipes/vae/vae-001-basic.yaml", profile=PROFILE)

# %% [markdown]
# ## 2. 🖼️ Reconstruct distinct inputs

# %%
# Restore the trained model and settings.
model, config = load_trained_model(run_dir)

# Choose held-out garments.
inputs = validation_examples(config)

# Disable gradients for the inspection.
with torch.no_grad():

    # Get the ordinary reconstructions and latent codes.
    output = model(inputs)

# Select the ordinary reconstructions.
reconstructions = output.reconstruction

# Show how each input is reconstructed.
_ = plot_reconstruction_grid(inputs, reconstructions, include_error=True)

# %% [markdown]
# ## 3. 📌 Give every input the same note
#
# 🧪 **Try** changing `fixed_value` to `1.0` and rerunning.
# Can the shared note change the output without making it depend on the input?

# %%
# Choose a shared coordinate value.
fixed_value = 0.0

# Read the input-dependent latent codes.
codes = output.latent

# Replace every coordinate with the same value.
fixed_codes = torch.full_like(codes, fixed_value)

# Disable gradients for decoding.
with torch.no_grad():

    # Decode the shared note for every garment.
    fixed_reconstructions = model.decode(fixed_codes)

# Compare those outputs against the original garments.
_ = plot_reconstruction_grid(inputs, fixed_reconstructions, include_error=True)

# %% [markdown]
# ## 4. 📊 Compare reconstruction errors

# %%
# Score ordinary reconstruction for each garment.
normal_errors = per_example_mse(inputs, reconstructions)

# Average the ordinary errors.
normal_mse = normal_errors.mean()

# Score fixed-note reconstruction for each garment.
fixed_errors = per_example_mse(inputs, fixed_reconstructions)

# Average the fixed-note errors.
fixed_mse = fixed_errors.mean()

# Compare errors on the same examples.
print(f"Normal MSE: {normal_mse:.4f} · fixed-note MSE: {fixed_mse:.4f}")

# %% [markdown]
# ## 5. 🔍 Inspect KL by dimension

# %%
# Display the diagnostic from this run.
show_run_figure(run_dir, "kl-per-dimension.png")

# %% [markdown]
# Posterior collapse combines weak input information in the encoder with a
# decoder that does not use it. Low KL alone is insufficient. KL thresholds also
# can count prior mismatch unrelated to input information.
#
# Warm-up gradually adds KL pressure so reconstruction can establish a useful
# path first. Free bits removes the incentive to compress a dimension below an
# allowance. Neither guarantees useful information or deserves credit for fixing
# a control that never collapsed.

# %% [markdown]
# ## ✅ Advancement gate — transfer check
#
# Run A has low KL and distinct, accurate reconstructions; Run B has low KL and constant reconstructions. Which deserves a collapse investigation?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Run B. In A, investigate whether little information is being used efficiently. Confirm B with latent interventions and posterior variation; the shared low KL number is not enough.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# Free bits optimizes a per-dimension floor $\max(\lambda, KL_j)$.
# Below the floor, the compression term supplies no further gradient; it does not
# force a dimension to encode information. Raw KL and optimized effective KL can
# differ. Run `studies/vae/vae-002-collapse-remedies.yaml` to compare immediate KL,
# warm-up, and free bits. The small decoder may not collapse: report that honestly.
# </details>

# %% [markdown]
# ➡️ **Next:** [Lesson 10](10-vqvae.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/10-vqvae.ipynb). No worksheet is required.
