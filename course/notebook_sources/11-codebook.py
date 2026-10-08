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
# # Lesson 11 — Count the symbols actually used
#
# **Learning objective:** distinguish vocabulary size, coverage, and balanced usage.
#
# A dictionary with 128 words is not a 128-word conversation if almost every
# sentence repeats one word. Codebook size counts available symbols; codes used
# counts observed symbols; perplexity summarizes how evenly they are used.
# Inspect the run from Lesson 10 before spending time training larger codebooks.

# %% [markdown]
# ## Setup
#
# Run this cell first. In Colab it fetches the course code automatically;
# locally it uses your checkout. The lessons use short CPU experiments.

# %%
import os
from pathlib import Path
import subprocess
import sys

# Find an existing checkout before fetching one in a fresh Colab session.
_start = Path.cwd().resolve()
_course_root = next((
    candidate for candidate in (_start, *_start.parents)
    if (candidate / "course/curriculum.yaml").is_file()
    and (candidate / "src/course_aiml_autoencoders").is_dir()
), None)
if _course_root is None:
    try:
        import google.colab
    except ImportError as error:
        raise RuntimeError("Open this notebook from the course checkout or in Google Colab.") from error
    _course_root = _start / "course-aiml-autoencoders"
    if not _course_root.exists():
        subprocess.run([
            "git", "clone", "--depth", "1", "--branch", "main",
            "https://github.com/tsilva/course-aiml-autoencoders.git",
            str(_course_root),
        ], check=True)
    if not (_course_root / "course/curriculum.yaml").is_file():
        raise RuntimeError(f"Incomplete course checkout at {_course_root}; rename it and rerun setup.")
os.chdir(_course_root)
_course_src = str(_course_root / "src")
if _course_src not in sys.path:
    sys.path.insert(0, _course_src)
print("Course ready:", _course_root)

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
# Two encoders use all eight symbols. One uses them evenly; the other uses one symbol 99% of the time. Should their perplexities match?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# No. Both have full coverage, but the concentrated encoder has much lower effective vocabulary size.
# </details>

# %%
from course_aiml_autoencoders.diagnostics.codebook import code_usage

for name, tokens in (("balanced", torch.arange(8).repeat(100)),
                     ("dominated", torch.cat([torch.zeros(792, dtype=torch.long), torch.arange(8)]))):
    usage = code_usage(tokens, 8)
    print(name, "codes used:", int(usage["codes_used"]), "perplexity:", float(usage["perplexity"]))
run_dir = learn("recipes/vqvae/vqvae-001-basic.yaml", profile=PROFILE)
usage = load_run_summary(run_dir)["codebook"]
print("Actual validation usage:", usage)
display(Image(filename=str(run_dir / "figures/codebook-usage.png")))

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
# ## Advancement gate — transfer check
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
# **Next:** [Lesson 12](12-code-prior.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/12-code-prior.ipynb). No worksheet is required.
