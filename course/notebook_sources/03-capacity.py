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
# # 🧪 Lesson 03 — Let the reconstruction sheet bend
#
# **Learning objective:** separate nonlinear geometry from bottleneck size.
#
# A linear decoder builds everything from one flat set of directions. Nonlinear
# layers let the mapping bend and respond differently in different regions.
# Keep the eight-number bottleneck fixed and change only the hidden-layer setup.
# The larger parameter count remains a confound; this tests two architectures,
# not the isolated effect of an activation function.

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
# With eight sliders in both models, can nonlinear layers still improve reconstruction?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Yes: the mapping can represent more complex structure at the same latent dimension. Improvement is empirical, and added parameters also contribute.
# </details>

# %% [markdown]
# ## 1. 🏋️ Run the linear control

# %%
# Use the quick training budget.
PROFILE = "quick"

# Run this recipe or reuse matching evidence.
linear_dir = learn("recipes/ae/ae-001-linear.yaml", profile=PROFILE)

# %% [markdown]
# ## 2. 🏋️ Run the nonlinear variant

# %%
# Run this recipe or reuse matching evidence.
nonlinear_dir = learn("recipes/ae/ae-002-nonlinear.yaml", profile=PROFILE)

# %% [markdown]
# ## 3. 📊 Compare held-out MSE and model size

# %%
# Name the runs to compare.
runs = (("linear", linear_dir), ("nonlinear", nonlinear_dir))

# Read the same evidence from each run.
for name, run_dir in runs:

    # Open the run summary.
    summary = load_run_summary(run_dir)

    # Select the held-out metrics.
    metrics = summary["best_validation_metrics"]

    # Read MSE.
    mse = metrics["validation/reconstruction_loss"]

    # Read parameter count.
    parameters = summary["parameter_count"]

    # Print the scores together.
    print(name, "MSE:", mse, "parameter count:", parameters)

# %% [markdown]
# ## 4. 🖼️ Inspect the linear reconstruction

# %%
# Display the diagnostic from this run.
show_run_figure(linear_dir, "reconstructions.png")

# %% [markdown]
# ## 5. 🖼️ Inspect the nonlinear reconstruction

# %%
# Display the diagnostic from this run.
show_run_figure(nonlinear_dir, "reconstructions.png")

# %% [markdown]
# ## 6. 🔁 Switch between the same examples
#
# 🧪 **Try** setting `selected_run` to `"nonlinear"` and rerunning this cell.
# Compare the same garment and look for one edge that changes.

# %%
# Group the runs by their labels.
gallery_runs = {"linear": linear_dir, "nonlinear": nonlinear_dir}

# Choose which result to view.
selected_run = "linear"

# Look up the exact run for this label.
gallery_run = gallery_runs[selected_run]

# Display the selected run's figure.
show_run_figure(gallery_run, "reconstructions.png")

# %% [markdown]
# Compare the same validation examples. Did edges or class-specific shape
# improve? A better pixel score supports reconstruction usefulness under this
# setup, not automatic disentanglement or downstream utility.
#
# More latent coordinates are a separate change: they let more information pass.
# A bottleneck of 784 with sufficient flexibility may allow copying, removing
# the pressure to find a compact summary.

# %% [markdown]
# ## ✅ Advancement gate — transfer check
#
# A 128-dimensional model beats an eight-dimensional one. Is it the better representation?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# It is better on the measured reconstruction task. Compression, robustness, generation, and downstream usefulness still require separate evidence.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# Run the full latent-size sweep:
# `uv run course-aiml-autoencoders study studies/ae/ae-003-latent-capacity.yaml --seeds 0`.
# Vary only latent size. Expect possible diminishing returns; inspect the result
# before concluding. The original nonlinearity study is also available in
# `studies/ae/ae-002-nonlinearity.yaml`.
# </details>

# %% [markdown]
# ➡️ **Next:** [Lesson 04](04-regularized-ae.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/04-regularized-ae.ipynb). No worksheet is required.
