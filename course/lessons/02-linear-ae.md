# Lesson 02 — Rebuild a garment from eight sliders

**Learning objective:** understand a linear encoder, decoder, and bottleneck through visible reconstruction.

The encoder turns 784 pixel values into eight slider settings. The decoder
mixes eight learned patterns to rebuild the image. These are broad patterns,
not eight selected pixels. Training teaches the encoder and decoder a shared
language; the bottleneck forces them to prioritize what that language can express.

[Open the notebook](../notebooks/02-linear-ae.ipynb). Run its cells in order for the experiment and evidence.

## Predict before running

Will eight input-dependent sliders beat one mean poster? What details should survive?

<details>
<summary>Reveal the expected reasoning</summary>

Usually the model beats the mean poster by tailoring the sliders to each input. Broad shape and brightness should survive; fine detail is expensive to express with eight linear directions.
</details>

## Run the revealing experiment

The notebook uses short CPU runs and reuses exact matching completed evidence.
It prints the run path and measured duration. Keep `PROFILE = "quick"` for the
core path; use `"full"` for the original budget. No worksheet is required.

## Advancement gate — transfer check

A smooth blend between a shirt and a shoe appears on screen. Does that establish a useful generator?

<details>
<summary>Reveal the expected reasoning</summary>

No. A linear decoder creates smooth blends automatically. We still need a way to draw plausible new latent codes; reconstruction does not teach a sampling distribution.
</details>

<details>
<summary>Optional: go deeper</summary>

The encoder computes `z = W_e x + b_e`; the decoder computes
`x_hat = W_d z + b_d`. Their composition is affine. With eight latent
coordinates, all reconstructions lie in an at-most-eight-dimensional affine
subspace. Under appropriate optimization, the reconstruction subspace matches
PCA, although individual latent axes may be rotated or rescaled.

Explore decoder directions, a one-slider traversal, and interpolation using the
[linear AE reference](../lessons/02-linear-ae.md).
</details>

**Next:** [Lesson 03](../notebooks/03-capacity.ipynb).
