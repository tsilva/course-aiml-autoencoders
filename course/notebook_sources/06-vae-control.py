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
# # Lesson 06 — Write a cloud of possible notes
#
# **Learning objective:** understand stochastic encoding before introducing a prior penalty.
#
# The VAE encoder writes a small cloud around an image's note instead of one
# fixed note. Its center says where the note belongs; its spread says how much
# it can vary. The decoder practices reading sampled notes from that cloud.
# Start with two dimensions so you can see randomness before seeing equations.

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
# If the center stays fixed and the spread grows, what changes when we encode the same input repeatedly?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Samples spread farther from the center. The input still has the same predicted center, but each sampled note can differ.
# </details>

# %%
from course_aiml_autoencoders.models.vae import reparameterize

mu = torch.tensor([[2.0, -1.0]]).expand(200, -1)
for spread in (0.1, 0.8):
    logvar = torch.full_like(mu, 2 * torch.log(torch.tensor(spread)))
    samples = reparameterize(mu, logvar)
    plt.scatter(samples[:, 0], samples[:, 1], s=8, alpha=0.4, label=f"spread {spread}")
plt.scatter([2], [-1], marker="x", c="black", label="center")
plt.axis("equal")
plt.legend()
plt.title("Same center, two different clouds; illustrative, not trained")
plt.show()

# %% [markdown]
# Sampling adds variation but does not tell the encoder where all clouds should
# live. Turn off the prior penalty (`beta=0`) to isolate that fact.

# %%
run_dir = learn("recipes/vae/vae-000-kl-off.yaml", profile=PROFILE)
summary = load_run_summary(run_dir)
print(summary["best_validation_metrics"])
display(Image(filename=str(run_dir / "figures/reconstructions.png")))
display(Image(filename=str(run_dir / "figures/random-latent-samples.png")))

# %% [markdown]
# Compare reconstruction with decoding randomly invented standard-normal notes.
# Weighted KL is exactly zero; raw KL measures mismatch but is not optimized.
# Randomness alone does not make that convenient distribution match the encoded
# clouds. Evaluation reconstructs from the cloud center for a stable view;
# training samples from the cloud.

# %% [markdown]
# ## Advancement gate — transfer check
#
# A beta-zero VAE reconstructs well but samples poorly from the standard normal. Is randomness broken?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Not necessarily. Reconstruction can work while encoded clouds occupy locations the standard-normal sampler rarely visits. The missing constraint is prior compatibility.
# </details>

# %% [markdown]
# <details>
# <summary>Optional: go deeper</summary>
#
# Reparameterization writes a sample as
# $z=\mu+\exp(0.5\,\mathrm{logvar})\epsilon$, with independent standard-normal noise.
# For a fixed noise sample, this is differentiable in the center and spread;
# reconstruction gradients can train both. `logvar` stores log variance, not
# standard deviation. See the [gradient reference](../lessons/06-vae-control.md).
# </details>

# %% [markdown]
# **Next:** [Lesson 07](07-standard-vae.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/07-standard-vae.ipynb). No worksheet is required.
