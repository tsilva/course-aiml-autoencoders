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
# # 🧪 Lesson 07 — Give the clouds a shared address system
#
# **Learning objective:** explain why prior pressure trades reconstruction detail for easier sampling.
#
# Imagine every input's cloud using a different, far-away address system.
# Random notes drawn near zero will miss many of them. KL pressure charges for
# moving or narrowing a cloud away from the shared standard-normal prior.
# That encourages overlap with places our sampler knows how to visit.

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
from course_aiml_autoencoders.course import load_run_summary
from course_aiml_autoencoders.course.notebook_views import show_run_figure

display(Markdown("""
| Tool | What it does in this lesson |
| --- | --- |
| `learn` | Trains a recipe or reuses an exact matching run. |
| `load_run_summary` | Reads recorded validation results and run metadata. |
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
# Compared with beta zero, should prior pressure improve every reconstruction and every sample?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# No. It can improve prior compatibility while sacrificing reconstruction information. Too much pressure can make clouds lose their input-specific information.
# </details>

# %% [markdown]
# ## 1. 🏋️ Run the beta-zero control

# %%
# Use the quick training budget.
PROFILE = "quick"

# Run this recipe or reuse matching evidence.
zero_dir = learn("recipes/vae/vae-000-kl-off.yaml", profile=PROFILE)

# %% [markdown]
# ## 2. 🏋️ Add prior pressure

# %%
# Run this recipe or reuse matching evidence.
one_dir = learn("recipes/vae/vae-001-basic.yaml", profile=PROFILE)

# %% [markdown]
# ## 3. 📊 Compare reconstruction and KL

# %%
# Name the runs to compare.
runs = (("beta zero", zero_dir), ("beta one", one_dir))

# Read the same evidence from each run.
for name, run_dir in runs:

    # Open the run summary.
    summary = load_run_summary(run_dir)

    # Select the held-out metrics.
    metrics = summary["best_validation_metrics"]

    # Read reconstruction BCE.
    bce = metrics["validation/reconstruction_loss"]

    # Read raw KL.
    raw_kl = metrics["validation/kl_loss"]

    # Print the scores together.
    print(name, "reconstruction BCE:", bce, "raw KL:", raw_kl)

# %% [markdown]
# ## 4. 🎲 View beta-zero samples

# %%
# Display the diagnostic from this run.
show_run_figure(zero_dir, "random-latent-samples.png")

# %% [markdown]
# ## 5. 🎲 View beta-one samples

# %%
# Display the diagnostic from this run.
show_run_figure(one_dir, "random-latent-samples.png")

# %% [markdown]
# ## 6. 🔁 Switch sample galleries
#
# 🧪 **Try** setting `selected_run` to `"beta one"` and rerunning.
# Look for changes in coherence and diversity, alongside the measured KL.

# %%
# Group the runs by their labels.
gallery_runs = {"beta zero": zero_dir, "beta one": one_dir}

# Choose which result to view.
selected_run = "beta zero"

# Look up the exact run for this label.
gallery_run = gallery_runs[selected_run]

# Display the selected run's figure.
show_run_figure(gallery_run, "random-latent-samples.png")

# %% [markdown]
# Check whether samples become more coherent and whether reconstruction cost
# rises. Neither trend is guaranteed by eight epochs. This VAE uses BCE summed per
# image; earlier AEs used mean pixel MSE. Their loss magnitudes cannot be compared
# directly. Center-decoded validation reconstruction is also not a Monte Carlo
# estimate of the full stochastic ELBO.
#
# A VAE gives a known sampling prior and continuous probabilistic latents. The
# tradeoff is reconstruction fidelity, imperfect prior fit, and collapse risk.

# %% [markdown]
# ## ✅ Advancement gate — transfer check
#
# All input clouds match the prior perfectly, and every reconstruction looks alike. Is zero KL a success?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# It indicates input information was lost: the decoder is not reconstructing distinct inputs. Check latent dependence and reconstruction evidence, not KL alone.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# The beta-one negative ELBO combines expected negative log likelihood with
# $D_{KL}(q(z|x)\|p(z))$. Beta changes their relative price.
# Average posterior-to-prior KL includes input information **and** aggregate
# posterior mismatch; it is not an exact measurement of mutual information.
# Nonzero per-dimension KL is a diagnostic, not proof a coordinate carries useful
# input information. See the [objective reference](../lessons/07-standard-vae.md).
# </details>

# %% [markdown]
# ➡️ **Next:** [Lesson 08](08-rate-distortion.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/08-rate-distortion.ipynb). No worksheet is required.
