# Lesson 01 — Give reconstruction a reference

**Learning objective:** explain what MSE notices and why reconstruction needs a baseline.

MSE is a pixel accountant: it squares each brightness mismatch, then averages.
It knows locations and numbers, not garments. A shifted picture can score poorly
even when you recognize it. First use a tiny symbol; no training is needed.

[Open the notebook](../notebooks/01-mean-baseline.ipynb). Run its cells in order for the experiment and evidence.

## Predict before running

Will shifting a symbol one pixel cost more than deleting one bright pixel?

<details>
<summary>Reveal the expected reasoning</summary>

Often yes: a shift creates errors where the symbol used to be and where it moves. Pixel agreement differs from semantic similarity.
</details>

## Run the revealing experiment

The notebook uses short CPU runs and reuses exact matching completed evidence.
It prints the run path and measured duration. Keep `PROFILE = "quick"` for the
core path; use `"full"` for the original budget. No worksheet is required.

## Advancement gate — transfer check

A model predicts the same blurry image for every garment and earns a low MSE. What comparison would expose the problem?

<details>
<summary>Reveal the expected reasoning</summary>

Compare with the mean training image on the same held-out examples. Check whether changing the input changes the reconstruction.
</details>

<details>
<summary>Optional: go deeper</summary>

Under squared error, averaging minimizes the sum of deviations at each pixel.
MSE is useful for aligned pixel fidelity; it is sensitive to shifts and can
reward blur. Use a task-specific measure when asking a different question.
</details>

**Next:** [Lesson 02](../notebooks/02-linear-ae.ipynb).
