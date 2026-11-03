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
# # 🧪 Lesson 06 — Write a cloud of possible notes
#
# **Learning objective:** understand stochastic encoding before introducing a prior penalty.
#
# The VAE encoder writes a small cloud around an image's note instead of one
# fixed note. Its center says where the note belongs; its spread says how much
# it can vary. The decoder practices reading sampled notes from that cloud.
# Start with two dimensions so you can see randomness before seeing equations.

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
from course_aiml_autoencoders.models.vae import reparameterize
from course_aiml_autoencoders.course.notebook_views import show_run_figure
from course_aiml_autoencoders.course.notebook_views import plot_sample_clouds

torch.manual_seed(0)

display(Markdown("""
| Tool | What it does in this lesson |
| --- | --- |
| `torch` | Creates tensors and controls gradient tracking. |
| `learn` | Trains a recipe or reuses an exact matching run. |
| `load_run_summary` | Reads recorded validation results and run metadata. |
| `reparameterize` | Samples latent values from a center and log variance. |
| `show_run_figure` | Displays a diagnostic saved by the selected run. |
| `plot_sample_clouds` | Plots sampled latent values around a shared center. |
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
# If the center stays fixed and the spread grows, what changes when we encode the same input repeatedly?
#
# <details>
# <summary>Reveal the expected reasoning</summary>
#
# Samples spread farther from the center. The input still has the same predicted center, but each sampled note can differ.
# </details>

# %% [markdown]
# ## 1. 🎯 Fix the cloud center

# %%
# Choose the coordinates of the center.
center = [2.0, -1.0]

# Put the center in a tensor.
center_tensor = torch.tensor([center])

# Repeat the same center for 200 draws.
mu = center_tensor.expand(200, -1)

# Show the first shared center.
print(mu[0])

# %% [markdown]
# ## 2. ☁️ Change the spread
#
# 🧪 **Try** changing the second spread to `0.4` and rerunning.
# Watch how the cloud shrinks around the same center.

# %%
# Choose two standard deviations.
spreads = (0.1, 0.8)

# Start a collection of sample clouds.
clouds = []

# Sample at each spread.
for spread in spreads:

    # Represent the spread as a tensor.
    spread_tensor = torch.tensor(spread)

    # Take the log of the standard deviation.
    log_spread = torch.log(spread_tensor)

    # Convert to log variance.
    log_variance = 2 * log_spread

    # Use that variance in both dimensions.
    logvar = torch.full_like(mu, log_variance)

    # Draw samples from the cloud.
    samples = reparameterize(mu, logvar)

    # Name the cloud for the plot.
    label = f"spread {spread}"

    # Save the samples and their label.
    clouds.append((label, samples))

# Compare the sampled clouds.
plot_sample_clouds(clouds, center)

# %% [markdown]
# Sampling adds variation but does not tell the encoder where all clouds should
# live. Turn off the prior penalty (`beta=0`) to isolate that fact.

# %% [markdown]
# ## 3. 🏋️ Train without prior pressure

# %%
# Use the quick training budget.
PROFILE = "quick"

# Run this recipe or reuse matching evidence.
run_dir = learn("recipes/vae/vae-000-kl-off.yaml", profile=PROFILE)

# %% [markdown]
# ## 4. 📊 Inspect validation metrics

# %%
# Read the run summary.
summary = load_run_summary(run_dir)

# Select the validation metrics.
metrics = summary["best_validation_metrics"]

# Show the reconstruction and KL metrics.
print(metrics)

# %% [markdown]
# ## 5. 🖼️ Inspect reconstructions

# %%
# Display the diagnostic from this run.
show_run_figure(run_dir, "reconstructions.png")

# %% [markdown]
# ## 6. 🎲 Inspect invented samples

# %%
# Display the diagnostic from this run.
show_run_figure(run_dir, "random-latent-samples.png")

# %% [markdown]
# Compare reconstruction with decoding randomly invented standard-normal notes.
# Weighted KL is exactly zero; raw KL measures mismatch but is not optimized.
# Randomness alone does not make that convenient distribution match the encoded
# clouds. Evaluation reconstructs from the cloud center for a stable view;
# training samples from the cloud.

# %% [markdown]
# ## ✅ Advancement gate — transfer check
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
# ➡️ **Next:** [Lesson 07](07-standard-vae.ipynb) · [Open in Colab](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/07-standard-vae.ipynb). No worksheet is required.
