# Lesson 06 — Write a cloud of possible notes

**Learning objective:** understand stochastic encoding before introducing a prior penalty.

The VAE encoder writes a small cloud around an image's note instead of one
fixed note. Its center says where the note belongs; its spread says how much
it can vary. The decoder practices reading sampled notes from that cloud.
Start with two dimensions so you can see randomness before seeing equations.

[Open the notebook](../notebooks/06-vae-control.ipynb). Run its cells in order for the experiment and evidence.

## Predict before running

If the center stays fixed and the spread grows, what changes when we encode the same input repeatedly?

<details>
<summary>Reveal the expected reasoning</summary>

Samples spread farther from the center. The input still has the same predicted center, but each sampled note can differ.
</details>

## Run the revealing experiment

The notebook uses short CPU runs and reuses exact matching completed evidence.
It prints the run path and measured duration. Keep `PROFILE = "quick"` for the
core path; use `"full"` for the original budget. No worksheet is required.

## Advancement gate — transfer check

A beta-zero VAE reconstructs well but samples poorly from the standard normal. Is randomness broken?

<details>
<summary>Reveal the expected reasoning</summary>

Not necessarily. Reconstruction can work while encoded clouds occupy locations the standard-normal sampler rarely visits. The missing constraint is prior compatibility.
</details>

<details>
<summary>Optional: go deeper</summary>

Reparameterization writes a sample as
$z=\mu+\exp(0.5\,\mathrm{logvar})\epsilon$, with independent standard-normal noise.
For a fixed noise sample, this is differentiable in the center and spread;
reconstruction gradients can train both. `logvar` stores log variance, not
standard deviation. See the [gradient reference](../lessons/06-vae-control.md).
</details>

**Next:** [Lesson 07](../notebooks/07-standard-vae.ipynb).
