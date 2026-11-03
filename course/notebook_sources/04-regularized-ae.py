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
# # 🧪 Lesson 04 — Learn to remove noise
#
# **Learning objective:** distinguish reconstructing an input from recovering its clean target.
#
# Give an ordinary AE a noisy image and copying some noise may help its task.
# Give a denoising AE the same noisy image but score against the clean image,
# and copying noise is penalized. Change only training corruption; hold the
# architecture, clean targets, split, optimizer, and evaluation noise fixed.

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

display(Markdown("""
| Tool | What it does in this lesson |
| --- | --- |
| `learn` | Trains a recipe or reuses an exact matching run. |
| `load_run_summary` | Reads recorded validation results and run metadata. |
| `repository_root` | Finds the course checkout so recipe paths work from any notebook folder. |
| `load_yaml` | Reads the experiment recipe. |
| `with_overrides` | Copies a recipe with a named setting changed. |
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
# Which model should better recover a clean image from noise? Must it also win on clean inputs?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# The denoising model is trained for noisy-input recovery. It can trade some clean-input fidelity for robustness, so clean MSE alone cannot answer the denoising question.
# </details>

# %% [markdown]
# ## 1. ⚙️ Make a clean-input control

# %%
# Use the quick training budget.
PROFILE = "quick"

# Find the course root.
root = repository_root()

# Locate the denoising recipe.
recipe_path = root / "recipes/ae/ae-003-denoising.yaml"

# Read the denoising recipe.
recipe = load_yaml(recipe_path)

# Remove training corruption for the control.
clean_recipe = with_overrides(recipe, {"training.input_corruption": None})

# Give the clean control its own run ID.
clean_recipe["id"] = "ae/ae-003-denoising-clean-control"

# Show the intervention.
print("Control: clean training inputs. Variant: corrupted training inputs.")

# %% [markdown]
# ## 2. 🏋️ Run the clean-input control

# %%
# Train with clean inputs.
clean_dir = learn(clean_recipe, profile=PROFILE)

# %% [markdown]
# ## 3. 🏋️ Run the denoiser

# %%
# Train with corrupted inputs.
denoising_dir = learn(recipe, profile=PROFILE)

# %% [markdown]
# ## 4. 📊 Score both on the same corruption

# %%
# Name the runs to compare.
runs = (("clean-trained", clean_dir), ("denoising", denoising_dir))

# Read the same evidence from each run.
for name, run_dir in runs:

    # Open the run summary.
    summary = load_run_summary(run_dir)

    # Select the held-out metrics.
    metrics = summary["best_validation_metrics"]

    # Read clean MSE.
    clean_mse = metrics["validation/reconstruction_loss"]

    # Read noisy-input → clean-target MSE.
    corrupted_mse = metrics["validation/corrupted_mse"]

    # Read unprocessed noisy-input MSE.
    noisy_mse = metrics["validation/noisy_input_mse"]

    # Print the scores together.
    print(name, "clean MSE:", clean_mse, "noisy-input → clean-target MSE:", corrupted_mse, "unprocessed noisy-input MSE:", noisy_mse)

# %% [markdown]
# ## 5. 🖼️ Inspect the clean-trained control

# %%
# Display the diagnostic from this run.
show_run_figure(clean_dir, "corrupted-input-reconstructions.png")

# %% [markdown]
# ## 6. 🖼️ Inspect the denoiser

# %%
# Display the diagnostic from this run.
show_run_figure(denoising_dir, "corrupted-input-reconstructions.png")

# %% [markdown]
# ## 7. 🔁 Toggle the prediction
#
# 🧪 **Try** setting `selected_run` to `"denoising"` and rerunning.
# Look at the error row for the same noisy input.

# %%
# Group the runs by their labels.
gallery_runs = {"clean-trained": clean_dir, "denoising": denoising_dir}

# Choose which result to view.
selected_run = "clean-trained"

# Look up the exact run for this label.
gallery_run = gallery_runs[selected_run]

# Display the selected run's figure.
show_run_figure(gallery_run, "corrupted-input-reconstructions.png")

# %% [markdown]
# Read the four rows: clean target, noisy input, reconstruction, clean-target
# error. Both models see identical validation corruption. The noise-only score
# also shows how much improvement comes from processing the image at all.
#
# If the short experiment contradicts the prediction, check whether both models
# learned enough before claiming denoising is ineffective. Use this approach for
# noise resembling the training corruption; robustness to a new noise type is a
# separate question.

# %% [markdown]
# ## ✅ Advancement gate — transfer check
#
# A denoiser improves clean-input MSE but leaves noisy images untouched. Has it demonstrated denoising?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# No. Score noisy-input reconstructions against the clean targets on the same corruptions used for the control. Clean-input improvement answers another question.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# Sparse AEs add a cost for latent activity rather than changing input noise:
# `L = reconstruction + lambda * mean(abs(z))`. Smaller mean activity alone does
# not prove more zero activations or useful sparsity; latent rescaling can also
# reduce that number. Inspect activity patterns and decoder weights.
#
# Full studies: `studies/ae/ae-004-denoising.yaml` and
# `studies/ae/ae-005-sparsity.yaml`. Run them only if these extensions answer your
# next question.
# </details>

# %% [markdown]
# ➡️ **Next:** [Lesson 05](05-ae-geometry.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/05-ae-geometry.ipynb). No worksheet is required.
