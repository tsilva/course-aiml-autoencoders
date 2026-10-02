# Lesson 13 — Choose a representation for a task

**Learning objective:** select a model from required behavior and diagnose contradictory evidence.

Choose the mechanism the task needs: an AE for input-conditioned
reconstruction, a VAE for regularized continuous sampling, or a VQ-VAE for
visual tokens plus a learned prior for generation. Finish with one choice and
one unfamiliar failure case. A report, seed sweep, and extra application are
optional extensions.

[Open the notebook](../notebooks/13-synthesis.ipynb). Run its cells in order for the experiment and evidence.

## Predict before running

A product needs compact discrete image tokens for another model. Which family fits, and what evidence would you inspect?

<details>
<summary>Reveal the expected reasoning</summary>

VQ-VAE. Check reconstruction, token maps, and global vocabulary usage. If the product also generates images, evaluate a prior paired with that exact representation.
</details>

## Run the revealing experiment

The notebook uses short CPU runs and reuses exact matching completed evidence.
It prints the run path and measured duration. Keep `PROFILE = "quick"` for the
core path; use `"full"` for the original budget. No worksheet is required.

## Advancement gate — transfer check

An unfamiliar VAE has high KL, excellent reconstruction, and poor prior samples. What single next check would distinguish useful input information from prior mismatch?

<details>
<summary>Reveal the expected reasoning</summary>

Inspect the encoded aggregate distribution versus the sampling prior, or test a controlled change in beta. High KL alone does not mean useful representation capacity; it also includes aggregate mismatch. Avoid changing architecture, data, and beta together.
</details>

<details>
<summary>Optional: go deeper</summary>

For a durable claim, run only the relevant two variants over seeds 0, 1,
and 2, report mean and variation, and inspect seed-specific failures. Use
[WORKSHEET.md](../WORKSHEET.md) only when preserving a research conclusion.

For a downstream extension, freeze the encoder and compare a classifier or
retrieval probe on latent features with raw pixels and PCA. Keep identical
splits and assess the task rather than assuming reconstruction implies utility.

Use `build_test_dataloader` for final confirmation after choosing the model.
The official test split is reserved; routine training and selection use a fixed
holdout from the original training split. Do not tune again on test results.
</details>
