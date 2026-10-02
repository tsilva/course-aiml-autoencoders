<div align="center">
  <img src="./logo.png" alt="Autoencoders: From Bottlenecks to Discrete Latents" width="760" />

  **🧪 Learn autoencoders by predicting, running, and inspecting experiments. 🧪**
</div>

An executable Python course on autoencoders, VAEs, and VQ-VAEs for learners
and ML practitioners. Work through 14 notebooks: predict a result, inspect a
small experiment, and check your understanding. Math and larger experiments
are optional.

## Curriculum

Follow lessons 00–13 in order.

| Lesson / notebook | Colab | What you learn |
|---|---|---|
| [00 · See what an autoencoder does](course/notebooks/00-laboratory.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/00-laboratory.ipynb) | Trace the encoder, bottleneck, and decoder before training. |
| [01 · Give reconstruction a reference](course/notebooks/01-mean-baseline.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/01-mean-baseline.ipynb) | Measure reconstruction error against mean-image and black-image baselines. |
| [02 · Rebuild a garment from eight sliders](course/notebooks/02-linear-ae.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/02-linear-ae.ipynb) | Train a linear AE with an eight-dimensional bottleneck. |
| [03 · Let the reconstruction sheet bend](course/notebooks/03-capacity.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/03-capacity.ipynb) | Explore what nonlinearity adds to a fixed-size bottleneck. |
| [04 · Learn to remove noise](course/notebooks/04-regularized-ae.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/04-regularized-ae.ipynb) | Recover clean targets from noisy inputs. |
| [05 · Reconstruction does not choose new codes](course/notebooks/05-ae-geometry.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/05-ae-geometry.ipynb) | Understand why interpolation does not provide a sampling rule. |
| [06 · Write a cloud of possible notes](course/notebooks/06-vae-control.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/06-vae-control.ipynb) | Explore stochastic encoding and reparameterization without a prior penalty. |
| [07 · Give the clouds a shared address system](course/notebooks/07-standard-vae.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/07-standard-vae.ipynb) | See how KL pressure changes reconstruction and sampling. |
| [08 · Change the price of information](course/notebooks/08-rate-distortion.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/08-rate-distortion.ipynb) | Trade reconstruction quality against the KL information cost using beta. |
| [09 · Check whether the decoder uses the note](course/notebooks/09-collapse.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/09-collapse.ipynb) | Test latent dependence and diagnose posterior collapse. |
| [10 · Snap notes to a learned vocabulary](course/notebooks/10-vqvae.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/10-vqvae.ipynb) | Turn continuous features into discrete VQ-VAE tokens. |
| [11 · Count the symbols actually used](course/notebooks/11-codebook.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/11-codebook.ipynb) | Measure codebook usage, dead codes, and perplexity. |
| [12 · Learn which token arrangements belong together](course/notebooks/12-code-prior.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/12-code-prior.ipynb) | Learn a token prior; compare prediction baselines and sampling. |
| [13 · Choose a representation for a task](course/notebooks/13-synthesis.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsilva/course-aiml-autoencoders/blob/main/course/notebooks/13-synthesis.ipynb) | Choose an AE, VAE, or VQ-VAE for a task. |

<details>
<summary>Colab setup — run once in each new session before the lesson cells</summary>

Choose a CPU runtime with Python 3.11 or 3.12 under **Runtime → Change runtime
type**. Colab's [runtime version guide](https://research.google.com/colaboratory/runtime-version-faq.html)
lists compatible versions; `2026.07` provides Python 3.12. Copy this into a new
code cell above the lesson's first code cell and run it:

```python
import os
from pathlib import Path
import subprocess
import sys

if not (3, 11) <= sys.version_info[:2] < (3, 13):
    raise RuntimeError("Select a Colab runtime with Python 3.11 or 3.12, then rerun this cell.")

course_root = Path("/content/course-aiml-autoencoders")
if not course_root.exists():
    subprocess.run([
        "git", "clone", "--depth", "1", "--branch", "main",
        "https://github.com/tsilva/course-aiml-autoencoders.git", str(course_root),
    ], check=True)
os.chdir(course_root)
sys.path.insert(0, str(course_root / "src"))

import matplotlib, torch, torchvision, yaml
from course_aiml_autoencoders.course import repository_root
print("Ready:", repository_root())
```

Run the lesson cells in order, then open the next lesson from the table.
Colab uses its preinstalled libraries; downloads and experiments disappear
when the runtime resets. Local setup uses locked dependencies and keeps files
on your machine.

</details>

## Install

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/tsilva/course-aiml-autoencoders.git
cd course-aiml-autoencoders
uv sync --frozen
uv run --frozen python -m jupyterlab course/notebooks/00-laboratory.ipynb
```

Open Jupyter's printed URL in your browser.

## Notes

- Lessons default to short CPU runs and reuse completed experiments.
- Fashion-MNIST downloads to `data/` on first use; experiment results go in
  `runs/`. Both directories are ignored by Git.

See the [course guide](course/README.md) for experiment commands and authoring,
and [troubleshooting](course/TROUBLESHOOTING.md) for setup or training issues.
The [learning roadmap](docs/ROADMAP.md) and
[experiment protocol](docs/EXPERIMENT_PROTOCOL.md) cover deeper study.

## Architecture

![Course and experiment architecture](./architecture.png)
