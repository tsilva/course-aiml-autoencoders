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
# # Lesson 08 — Change the price of information
#
# **Learning objective:** interpret beta through a reconstruction/prior tradeoff.
#
# Beta is the price charged for moving a cloud away from the shared prior.
# Compare one familiar price with a higher one. Use the same architecture,
# split, optimizer, and training budget; change only beta.

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
# If beta rises from 1 to 4, which should usually decrease: reconstruction error or raw KL?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Raw KL should tend to decrease. Reconstruction error may rise because retaining input-specific detail costs more. Optimization can violate the trend.
# </details>

# %%
base = load_yaml(ROOT / "recipes/vae/vae-001-basic.yaml")
expensive = with_overrides(base, {"objective.beta": 4.0})
expensive["id"] = "vae/vae-beta-4"
observations = []
for beta, recipe in ((1.0, base), (4.0, expensive)):
    run_dir = learn(recipe, profile=PROFILE)
    summary = load_run_summary(run_dir)
    metrics = summary["best_validation_metrics"]
    rate, distortion = metrics["validation/kl_loss"], metrics["validation/reconstruction_loss"]
    observations.append((beta, rate, distortion))
    print("beta", beta, "raw KL", rate, "reconstruction BCE", distortion,
          "KL-threshold dimensions", summary["vae_latent"]["active_dimensions"])
    display(Image(filename=str(run_dir / "figures/random-latent-samples.png")))
for beta, rate, distortion in observations:
    plt.scatter(rate, distortion)
    plt.annotate(f"beta={beta:g}", (rate, distortion))
plt.xlabel("Raw KL, nats/image")
plt.ylabel("Center-decoded reconstruction BCE/image")
plt.title("Two observed tradeoffs; short runs, not an optimal frontier")
plt.show()

# %% [markdown]
# A point toward the left uses less posterior-to-prior KL; a point lower down
# reconstructs better under the chosen BCE convention. Neither axis defines a
# universally best model. Inspect samples and input dependence to decide whether
# the exchange helps the task. Total losses are unsuitable for ranking because
# beta changes what they mean.

# %% [markdown]
# ## Advancement gate — transfer check
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
# **Next:** [Lesson 09](09-collapse.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/09-collapse.ipynb). No worksheet is required.
