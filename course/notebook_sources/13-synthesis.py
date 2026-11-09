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
# # 🧪 Lesson 13 — Choose a representation for a task
#
# **Learning objective:** select a model from required behavior and diagnose contradictory evidence.
#
# Choose the mechanism the task needs: an AE for input-conditioned
# reconstruction, a VAE for regularized continuous sampling, or a VQ-VAE for
# visual tokens plus a learned prior for generation. Finish with one choice and
# one unfamiliar failure case. A report, seed sweep, and extra application are
# optional extensions.

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
from course_aiml_autoencoders.course import learn
from course_aiml_autoencoders.course.notebook_views import show_run_figure

display(Markdown("""
| Tool | What it does in this lesson |
| --- | --- |
| `learn` | Trains a recipe or reuses an exact matching run. |
| `show_run_figure` | Displays a diagnostic saved by the selected run. |
"""))

# %% [markdown]
# ## 🔮 Predict before running
#
# A product needs compact discrete image tokens for another model. Which family fits, and what evidence would you inspect?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# VQ-VAE. Check reconstruction, token maps, and global vocabulary usage. If the product also generates images, evaluate a prior paired with that exact representation.
# </details>

# %% [markdown]
# ## 1. 🏋️ Restore the continuous AE

# %%
# Use the quick training budget.
PROFILE = "quick"

# Run this recipe or reuse matching evidence.
ae_run = learn("recipes/ae/ae-002-nonlinear.yaml", profile=PROFILE)

# %% [markdown]
# ## 2. 🏋️ Restore the VAE

# %%
# Run this recipe or reuse matching evidence.
vae_run = learn("recipes/vae/vae-001-basic.yaml", profile=PROFILE)

# %% [markdown]
# ## 3. 🏋️ Restore the VQ-VAE

# %%
# Run this recipe or reuse matching evidence.
vq_run = learn("recipes/vqvae/vqvae-001-basic.yaml", profile=PROFILE)

# %% [markdown]
# ## 4. 🖼️ Inspect continuous AE reconstructions

# %%
# Display the diagnostic from this run.
show_run_figure(ae_run, "reconstructions.png")

# %% [markdown]
# ## 5. 🖼️ Inspect VAE reconstructions

# %%
# Display the diagnostic from this run.
show_run_figure(vae_run, "reconstructions.png")

# %% [markdown]
# ## 6. 🖼️ Inspect VQ-VAE reconstructions

# %%
# Display the diagnostic from this run.
show_run_figure(vq_run, "reconstructions.png")

# %% [markdown]
# ## 7. 🔁 Choose the evidence for your task
#
# 🧪 **Try** changing `selected_run` to the family that fits your task.
# Rerun the cell and name one property these reconstructions can help you assess.

# %%
# Group the runs by their labels.
gallery_runs = {"AE": ae_run, "VAE": vae_run, "VQ-VAE": vq_run}

# Choose which result to view.
selected_run = "AE"

# Look up the exact run for this label.
gallery_run = gallery_runs[selected_run]

# Display the selected run's figure.
show_run_figure(gallery_run, "reconstructions.png")

# %% [markdown]
# The pictures expose representation tradeoffs; they do not define a fair
# cross-family leaderboard. Architectures and training objectives differ.
#
# | Need | Starting choice | First evidence |
# |---|---|---|
# | Recover clean images from matched noise | Denoising AE | Noisy-input output scored against clean targets |
# | Generate using a known continuous prior | VAE | Reconstruction, latent dependence, prior samples |
# | Encode reusable discrete visual symbols | VQ-VAE | Reconstruction, token maps, global code usage |
# | Generate coherent token arrangements | VQ-VAE + prior | Validation likelihood and sampled arrangements |
#
# Explain your chosen mechanism in one sentence, then name the evidence that
# would make you change your choice. No written worksheet is required.

# %% [markdown]
# ## ✅ Advancement gate — transfer check
#
# An unfamiliar VAE has high KL, excellent reconstruction, and poor prior samples. What single next check would distinguish useful input information from prior mismatch?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Inspect the encoded aggregate distribution versus the sampling prior, or test a controlled change in beta. High KL alone does not mean useful representation capacity; it also includes aggregate mismatch. Avoid changing architecture, data, and beta together.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# For a durable claim, run only the relevant two variants over seeds 0, 1,
# and 2, report mean and variation, and inspect seed-specific failures. Use
# [WORKSHEET.md](../WORKSHEET.md) only when preserving a research conclusion.
#
# For a downstream extension, freeze the encoder and compare a classifier or
# retrieval probe on latent features with raw pixels and PCA. Keep identical
# splits and assess the task rather than assuming reconstruction implies utility.
#
# Use `build_test_dataloader` for final confirmation after choosing the model.
# The official test split is reserved; routine training and selection use a fixed
# holdout from the original training split. Do not tune again on test results.
# </details>

# %% [markdown]
# **Finished:** revisit the one concept you could not explain, or choose an optional extension in the [course guide](../README.md).
