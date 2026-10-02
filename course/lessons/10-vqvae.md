# Lesson 10 — Snap notes to a learned vocabulary

**Learning objective:** understand vector quantization as choosing symbols for image regions.

Instead of any continuous note, use one of a limited set of learned symbols.
A VQ-VAE encoder describes each image region as a vector; quantization snaps
that vector to its nearest codebook entry. The decoder rebuilds the image from
a grid of those entries. First watch four points snap to three fixed symbols.

[Open the notebook](../notebooks/10-vqvae.ipynb). Run its cells in order for the experiment and evidence.

## Predict before running

If two nearby vectors choose the same symbol, will the decoder still see their small difference?

<details>
<summary>Reveal the expected reasoning</summary>

No. Quantization replaces both with the same embedding. That loses detail but creates a discrete vocabulary.
</details>

## Run the revealing experiment

The notebook uses short CPU runs and reuses exact matching completed evidence.
It prints the run path and measured duration. Keep `PROFILE = "quick"` for the
core path; use `"full"` for the original budget. No worksheet is required.

## Advancement gate — transfer check

The codebook and decoder are trained. Why can a grid of uniformly random tokens still look incoherent?

<details>
<summary>Reveal the expected reasoning</summary>

Knowing word meanings does not teach sentence probabilities. The model has learned symbols and decoding, not a distribution over coherent spatial arrangements.
</details>

<details>
<summary>Optional: go deeper</summary>

Nearest-neighbor selection is $k=\arg\min_j\|z_e-e_j\|^2$.
Straight-through uses the snapped value forward and an identity surrogate
gradient backward; this is biased. Reconstruction gradients reach the encoder,
codebook loss updates embeddings, and commitment loss updates the encoder.
For isolated gradient probes, see the [VQ reference](../lessons/10-vqvae.md).
</details>

**Next:** [Lesson 11](../notebooks/11-codebook.ipynb).
