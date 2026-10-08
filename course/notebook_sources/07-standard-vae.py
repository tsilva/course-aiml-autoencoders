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
# # Lesson 07 — Give the clouds a shared address system
#
# **Learning objective:** explain why prior pressure trades reconstruction detail for easier sampling.
#
# Imagine every input's cloud using a different, far-away address system.
# Random notes drawn near zero will miss many of them. KL pressure charges for
# moving or narrowing a cloud away from the shared standard-normal prior.
# That encourages overlap with places our sampler knows how to visit.

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
# Compared with beta zero, should prior pressure improve every reconstruction and every sample?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# No. It can improve prior compatibility while sacrificing reconstruction information. Too much pressure can make clouds lose their input-specific information.
# </details>

# %%
zero_dir = learn("recipes/vae/vae-000-kl-off.yaml", profile=PROFILE)
one_dir = learn("recipes/vae/vae-001-basic.yaml", profile=PROFILE)
for name, run_dir in (("beta zero", zero_dir), ("beta one", one_dir)):
    metrics = load_run_summary(run_dir)["best_validation_metrics"]
    print(name, "reconstruction BCE:", metrics["validation/reconstruction_loss"],
          "raw KL:", metrics["validation/kl_loss"])
    display(Image(filename=str(run_dir / "figures/random-latent-samples.png")))

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
# ## Advancement gate — transfer check
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
# **Next:** [Lesson 08](08-rate-distortion.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/08-rate-distortion.ipynb). No worksheet is required.
