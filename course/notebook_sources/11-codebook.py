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
# # 🧪 Lesson 11 — Count the symbols actually used
#
# **Learning objective:** distinguish vocabulary size, coverage, and balanced usage.
#
# A dictionary with 128 words is not a 128-word conversation if almost every
# sentence repeats one word. Codebook size counts available symbols; codes used
# counts observed symbols; perplexity summarizes how evenly they are used.
# Inspect the run from Lesson 10 before spending time training larger codebooks.

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
from course_aiml_autoencoders.diagnostics.codebook import code_usage
from course_aiml_autoencoders.course.notebook_views import show_run_figure

torch.manual_seed(0)

display(Markdown("""
| Tool | What it does in this lesson |
| --- | --- |
| `torch` | Creates tensors and controls gradient tracking. |
| `learn` | Trains a recipe or reuses an exact matching run. |
| `load_run_summary` | Reads recorded validation results and run metadata. |
| `code_usage` | Measures code coverage and the effective vocabulary size. |
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
# Two encoders use all eight symbols. One uses them evenly; the other uses one symbol 99% of the time. Should their perplexities match?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# No. Both have full coverage, but the concentrated encoder has much lower effective vocabulary size.
# </details>

# %% [markdown]
# ## 1. 🔢 Use all eight codes evenly

# %%
# List each of the eight symbols once.
symbols = torch.arange(8)

# Repeat all eight symbols equally.
balanced = symbols.repeat(100)

# Measure the balanced vocabulary.
balanced_usage = code_usage(balanced, 8)

# Read how many codes appear.
balanced_count = balanced_usage["codes_used"]

# Read the effective vocabulary size.
balanced_perplexity = balanced_usage["perplexity"]

# Show the coverage and perplexity.
print(f"Balanced: {balanced_count} codes · perplexity {balanced_perplexity:.2f}")

# %% [markdown]
# ## 2. ⚖️ Concentrate usage in one code
#
# 🧪 **Try** changing `extra_zeros` to `8` or `0` and rerunning.
# Which metric responds to the change in concentration?

# %%
# Choose the number of extra copies of code zero.
extra_zeros = 792

# Make those additional assignments.
dominant = torch.zeros(extra_zeros, dtype=torch.long)

# Keep at least one assignment to every code.
dominated = torch.cat((dominant, symbols))

# Measure the concentrated vocabulary.
dominated_usage = code_usage(dominated, 8)

# Read how many codes still appear.
dominated_count = dominated_usage["codes_used"]

# Read the new effective vocabulary size.
dominated_perplexity = dominated_usage["perplexity"]

# Compare the coverage and perplexity.
print(f"Dominated: {dominated_count} codes · perplexity {dominated_perplexity:.2f}")

# %% [markdown]
# ## 3. 🏋️ Restore the trained VQ-VAE

# %%
# Use the quick training budget.
PROFILE = "quick"

# Run this recipe or reuse matching evidence.
run_dir = learn("recipes/vqvae/vqvae-001-basic.yaml", profile=PROFILE)

# %% [markdown]
# ## 4. 📊 Read actual validation usage

# %%
# Read the trained run summary.
summary = load_run_summary(run_dir)

# Find the codebook statistics.
usage = summary["codebook"]

# Show actual usage on held-out images.
print("Actual validation usage:", usage)

# %% [markdown]
# ## 5. 📉 Inspect the assignment histogram

# %%
# Display the diagnostic from this run.
show_run_figure(run_dir, "codebook-usage.png")

# %% [markdown]
# Compare the toy contrast with the actual histogram. Some dead entries do not
# by themselves establish failure: the spatial grid and decoder can reconstruct
# well with a smaller vocabulary. Conversely, high coverage can hide domination
# by a few codes. Inspect reconstruction alongside usage.
#
# Commitment pressure keeps encoder vectors near symbols. Its useful strength
# is empirical; stronger pressure need not improve either reconstruction or
# vocabulary use.

# %% [markdown]
# ## ✅ Advancement gate — transfer check
#
# A model offers 512 codes, uses 40, and has perplexity 9. What do those three numbers establish?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# 512 is nominal vocabulary, 40 received assignments in the evaluated data, and the uneven usage has effective diversity about 9. None alone establishes reconstruction or downstream quality.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# Full size and commitment sweeps live in
# `studies/vqvae/vqvae-001-codebook-size.yaml` and
# `studies/vqvae/vqvae-002-commitment.yaml`. Change one factor and compare raw and
# weighted loss terms separately. Dead codes are unobserved in the evaluated
# sample, not proof they can never be used. EMA updates and dead-code recovery
# are further optional experiments.
# </details>

# %% [markdown]
# ➡️ **Next:** [Lesson 12](12-code-prior.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/12-code-prior.ipynb). No worksheet is required.
