# Lesson 04 — Learn to remove noise

**Learning objective:** distinguish reconstructing an input from recovering its clean target.

Give an ordinary AE a noisy image and copying some noise may help its task.
Give a denoising AE the same noisy image but score against the clean image,
and copying noise is penalized. Change only training corruption; hold the
architecture, clean targets, split, optimizer, and evaluation noise fixed.

[Open the notebook](../notebooks/04-regularized-ae.ipynb). Run its cells in order for the experiment and evidence.

## Predict before running

Which model should better recover a clean image from noise? Must it also win on clean inputs?

<details>
<summary>Reveal the expected reasoning</summary>

The denoising model is trained for noisy-input recovery. It can trade some clean-input fidelity for robustness, so clean MSE alone cannot answer the denoising question.
</details>

## Run the revealing experiment

The notebook uses short CPU runs and reuses exact matching completed evidence.
It prints the run path and measured duration. Keep `PROFILE = "quick"` for the
core path; use `"full"` for the original budget. No worksheet is required.

## Advancement gate — transfer check

A denoiser improves clean-input MSE but leaves noisy images untouched. Has it demonstrated denoising?

<details>
<summary>Reveal the expected reasoning</summary>

No. Score noisy-input reconstructions against the clean targets on the same corruptions used for the control. Clean-input improvement answers another question.
</details>

<details>
<summary>Optional: go deeper</summary>

Sparse AEs add a cost for latent activity rather than changing input noise:
`L = reconstruction + lambda * mean(abs(z))`. Smaller mean activity alone does
not prove more zero activations or useful sparsity; latent rescaling can also
reduce that number. Inspect activity patterns and decoder weights.

Full studies: `studies/ae/ae-004-denoising.yaml` and
`studies/ae/ae-005-sparsity.yaml`. Run them only if these extensions answer your
next question.
</details>

**Next:** [Lesson 05](../notebooks/05-ae-geometry.ipynb).
