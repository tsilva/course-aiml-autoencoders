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
# # 🧪 Lesson 08 — Change the price of information
#
# **Learning objective:** interpret beta through a reconstruction/prior tradeoff.
#
# Beta is the price charged for moving a cloud away from the shared prior.
# Compare one familiar price with a higher one. Use the same architecture,
# split, optimizer, and training budget; change only beta.

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
from course_aiml_autoencoders.course import repository_root
from course_aiml_autoencoders.config import load_yaml
from course_aiml_autoencoders.config import with_overrides
from course_aiml_autoencoders.course.notebook_views import show_run_figure
from course_aiml_autoencoders.course.notebook_views import plot_rate_distortion

display(Markdown("""
| Tool | What it does in this lesson |
| --- | --- |
| `learn` | Trains a recipe or reuses an exact matching run. |
| `load_run_summary` | Reads recorded validation results and run metadata. |
| `repository_root` | Finds the course checkout so recipe paths work from any notebook folder. |
| `load_yaml` | Reads the experiment recipe. |
| `with_overrides` | Copies a recipe with a named setting changed. |
| `show_run_figure` | Displays a diagnostic saved by the selected run. |
| `plot_rate_distortion` | Plots reconstruction cost against raw KL for each observed beta. |
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
# If beta rises from 1 to 4, which should usually decrease: reconstruction error or raw KL?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Raw KL should tend to decrease. Reconstruction error may rise because retaining input-specific detail costs more. Optimization can violate the trend.
# </details>

# %% [markdown]
# ## 1. ⚙️ Set up a beta comparison
#
# 🧪 **Try** changing `higher_beta` to `2.0`. Rerun this cell and the
# subsequent experiment cells to see how the tradeoff moves.

# %%
# Use the quick training budget.
PROFILE = "quick"

# Find the course root.
root = repository_root()

# Locate the beta-one recipe.
recipe_path = root / "recipes/vae/vae-001-basic.yaml"

# Read the control recipe.
base = load_yaml(recipe_path)

# Choose the higher beta.
higher_beta = 4.0

# Change only the KL weight.
expensive = with_overrides(base, {"objective.beta": higher_beta})

# Name the variant by its weight.
expensive["id"] = f"vae/vae-beta-{higher_beta:g}"

# Show the two beta values.
print(f"Comparing beta 1 with beta {higher_beta:g}")

# %% [markdown]
# ## 2. 🏋️ Run the beta-one control

# %%
# Train or reuse the reference run.
base_dir = learn(base, profile=PROFILE)

# %% [markdown]
# ## 3. 🏋️ Run the higher-beta variant

# %%
# Train or reuse the altered KL weight.
higher_dir = learn(expensive, profile=PROFILE)

# %% [markdown]
# ## 4. 📊 Collect the two points

# %%
# Pair each beta with its run.
runs = ((1.0, base_dir), (higher_beta, higher_dir))

# Start an empty list of points.
observations = []

# Read both runs.
for beta, run_dir in runs:

    # Open this run's summary.
    summary = load_run_summary(run_dir)

    # Select the validation metrics.
    metrics = summary["best_validation_metrics"]

    # Read raw KL for the rate axis.
    rate = metrics["validation/kl_loss"]

    # Read reconstruction BCE for the distortion axis.
    distortion = metrics["validation/reconstruction_loss"]

    # Read the latent diagnostics.
    latent_summary = summary["vae_latent"]

    # Read the count of dimensions above the KL threshold.
    active = latent_summary["active_dimensions"]

    # Save one point for this beta.
    observations.append((beta, rate, distortion))

    # Report both loss terms and the diagnostic count.
    print("beta", beta, "raw KL", rate, "reconstruction BCE", distortion, "KL-threshold dimensions", active)

# %% [markdown]
# ## 5. 📉 Plot the observed tradeoff

# %%
# Plot the two points with their beta labels.
plot_rate_distortion(observations)

# %% [markdown]
# ## 6. 🎲 Inspect the control samples

# %%
# Display the diagnostic from this run.
show_run_figure(base_dir, "random-latent-samples.png")

# %% [markdown]
# ## 7. 🎲 Inspect the higher-beta samples

# %%
# Display the diagnostic from this run.
show_run_figure(higher_dir, "random-latent-samples.png")

# %% [markdown]
# A point toward the left uses less posterior-to-prior KL; a point lower down
# reconstructs better under the chosen BCE convention. Neither axis defines a
# universally best model. Inspect samples and input dependence to decide whether
# the exchange helps the task. Total losses are unsuitable for ranking because
# beta changes what they mean.

# %% [markdown]
# ## ✅ Advancement gate — transfer check
#
# The beta-four model has a lower total loss but worse reconstruction. Is it the winner?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# The totals optimize differently weighted objectives. Compare named components and the behavior required by the task; lower total loss does not settle the choice.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# For beta 0, 0.1, 1, and 4, run
# `uv run course-aiml-autoencoders study studies/vae/vae-001-beta-sweep.yaml --seeds 0`.
# Repeat only a meaningful finalist contrast across seeds if you want a durable
# numerical conclusion. Predict sample diversity as well as prior compatibility.
# </details>

# %% [markdown]
# ➡️ **Next:** [Lesson 09](09-collapse.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/09-collapse.ipynb). No worksheet is required.
