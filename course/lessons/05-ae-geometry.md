# Lesson 05 — Reconstruction does not choose new codes

**Learning objective:** distinguish interpolation from sampling.

An AE learns to read notes written by its encoder. Inventing random notes
asks the decoder to read a distribution it was never trained to expect.
A smooth path between two valid notes is a different claim from a reliable
method for drawing new notes.

[Open the notebook](../notebooks/05-ae-geometry.ipynb). Run its cells in order for the experiment and evidence.

## Predict before running

Should a smooth interpolation guarantee that random standard-normal codes produce plausible garments?

<details>
<summary>Reveal the expected reasoning</summary>

No. Continuity gives smooth changes; it says nothing about how probable the visited codes are. The AE has not matched its encoded distribution to a standard normal.
</details>

## Run the revealing experiment

The notebook uses short CPU runs and reuses exact matching completed evidence.
It prints the run path and measured duration. Keep `PROFILE = "quick"` for the
core path; use `"full"` for the original budget. No worksheet is required.

## Advancement gate — transfer check

You rescale every AE latent coordinate to unit variance. Is generation now solved?

<details>
<summary>Reveal the expected reasoning</summary>

No. Matching coordinate scales does not match correlations, shapes, or low-density holes. Sampling requires a distribution that models the joint encoded data.
</details>

<details>
<summary>Optional: go deeper</summary>

Plot a PCA projection of encoded validation examples, colored by class.
A projection helps reveal structure but cannot map every hole in a
higher-dimensional space. See the [geometry reference](../lessons/05-ae-geometry.md).
</details>

**Next:** [Lesson 06](../notebooks/06-vae-control.ipynb).
