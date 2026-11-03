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
# # 🧪 Lesson 12 — Learn which token arrangements belong together
#
# **Learning objective:** separate learning a visual vocabulary from learning a sampling distribution.
#
# A vocabulary tells us what symbols mean. A prior learns which symbol should
# come next, given the symbols already present. Scanning the token grid row by
# row lets a small GRU learn these dependencies while the VQ encoder and decoder
# stay frozen. Training supplies the true previous tokens; sampling supplies the
# prior's own previous choices.

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
from course_aiml_autoencoders.course import learn_prior
from course_aiml_autoencoders.course import load_run_summary
from course_aiml_autoencoders.course import load_metrics
from course_aiml_autoencoders.course import plot_metric_history
from course_aiml_autoencoders.course.notebook_views import show_run_figure

display(Markdown("""
| Tool | What it does in this lesson |
| --- | --- |
| `learn` | Trains a recipe or reuses an exact matching run. |
| `learn_prior` | Trains a token prior paired with a specific VQ-VAE checkpoint. |
| `load_run_summary` | Reads recorded validation results and run metadata. |
| `load_metrics` | Reads the loss values recorded during training. |
| `plot_metric_history` | Shows how the selected losses change with training. |
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
# Will beating a uniform token predictor prove that the prior learned spatial dependencies?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# No. Frequent symbols alone can beat uniform prediction. Compare with a training-frequency unigram baseline as well as inspecting generated arrangements.
# </details>

# %% [markdown]
# ## 1. 🧱 Restore the tokenizer

# %%
# Use the quick training budget.
PROFILE = "quick"

# Run this recipe or reuse matching evidence.
vq_run = learn("recipes/vqvae/vqvae-001-basic.yaml", profile=PROFILE)

# %% [markdown]
# ## 2. 🏋️ Learn token sequences

# %%
# Train a prior paired with this exact VQ checkpoint.
prior_run = learn_prior("recipes/prior/prior-001-gru.yaml", vq_run, profile=PROFILE)

# %% [markdown]
# ## 3. 📊 Compare held-out token predictions

# %%
# Read the prior run summary.
summary = load_run_summary(prior_run)

# Read the uniform predictor's cross-entropy.
uniform_ce = summary["uniform_cross_entropy"]

# Read the frequency baseline's cross-entropy.
unigram_ce = summary["unigram_cross_entropy"]

# Read the learned prior's held-out cross-entropy.
prior_ce = summary["best_validation_cross_entropy"]

# Compare the three predictors.
print("Uniform CE:", uniform_ce, "frequency CE:", unigram_ce, "learned-prior CE:", prior_ce)

# %% [markdown]
# ## 4. 📉 Inspect learning over time

# %%
# Load the recorded losses.
history = load_metrics(prior_run)

# Choose training and validation cross-entropy.
metrics = ["train/cross_entropy", "validation/cross_entropy"]

# Display the learning curves.
_ = plot_metric_history(history, metrics)

# %% [markdown]
# ## 5. 🎲 View samples from uniform tokens

# %%
# Display the diagnostic from this run.
show_run_figure(vq_run, "uniform-random-token-samples.png")

# %% [markdown]
# ## 6. 🖼️ View samples from the learned prior
#
# 🧪 **Try** setting `sample_source` to `"uniform"` and rerunning this cell.
# Compare the same visual feature in both galleries, such as coherent edges.

# %%
# Choose which sampling method to inspect.
sample_source = "learned"

# Pair each method with its run and figure.
sample_galleries = {
    "learned": (prior_run, "learned-prior-samples.png"),
    "uniform": (vq_run, "uniform-random-token-samples.png"),
}

# Look up the gallery for the selected sampling method.
sample_run, sample_figure = sample_galleries[sample_source]

# Show the selected gallery.
show_run_figure(sample_run, sample_figure)

# %% [markdown]
# Better validation likelihood than the unigram baseline supports learning
# structure beyond overall symbol frequency, but is not proof of good generated
# images. Inspect arrangements and diversity: sampling consumes its own history
# and can accumulate errors. Short training may reveal only frequency learning.
#
# Prior perplexity measures next-token uncertainty given history; codebook
# perplexity measures assignment diversity. They share a name but answer
# different questions. The prior cannot restore information discarded by the
# frozen VQ representation.

# %% [markdown]
# ## ✅ Advancement gate — transfer check
#
# The prior improves dramatically, but generated images still lack fine detail. Which stage might be the bottleneck?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# The VQ encoder/codebook/decoder may have discarded that detail. A better token distribution cannot recover information absent from the representation.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# Raster order gives the factorization
# $p(k_1,\ldots,k_N)=\prod_i p(k_i\mid k_{<i})$.
# Teacher forcing uses true history during training; sampling uses sampled
# history. The small GRU is chosen for clarity, not top-tier image quality.
# Temperature and larger priors are optional investigations; preserve the exact
# paired VQ checkpoint when comparing them.
# </details>

# %% [markdown]
# ➡️ **Next:** [Lesson 13](13-synthesis.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/13-synthesis.ipynb). No worksheet is required.
