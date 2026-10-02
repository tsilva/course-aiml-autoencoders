# Lesson 03 — Let the reconstruction sheet bend

**Learning objective:** separate nonlinear geometry from bottleneck size.

A linear decoder builds everything from one flat set of directions. Nonlinear
layers let the mapping bend and respond differently in different regions.
Keep the eight-number bottleneck fixed and change only the hidden-layer setup.
The larger parameter count remains a confound; this tests two architectures,
not the isolated effect of an activation function.

[Open the notebook](../notebooks/03-capacity.ipynb). Run its cells in order for the experiment and evidence.

## Predict before running

With eight sliders in both models, can nonlinear layers still improve reconstruction?

<details>
<summary>Reveal the expected reasoning</summary>

Yes: the mapping can represent more complex structure at the same latent dimension. Improvement is empirical, and added parameters also contribute.
</details>

## Run the revealing experiment

The notebook uses short CPU runs and reuses exact matching completed evidence.
It prints the run path and measured duration. Keep `PROFILE = "quick"` for the
core path; use `"full"` for the original budget. No worksheet is required.

## Advancement gate — transfer check

A 128-dimensional model beats an eight-dimensional one. Is it the better representation?

<details>
<summary>Reveal the expected reasoning</summary>

It is better on the measured reconstruction task. Compression, robustness, generation, and downstream usefulness still require separate evidence.
</details>

<details>
<summary>Optional: go deeper</summary>

Run the full latent-size sweep:
`uv run course-aiml-autoencoders study studies/ae/ae-003-latent-capacity.yaml --seeds 0`.
Vary only latent size. Expect possible diminishing returns; inspect the result
before concluding. The original nonlinearity study is also available in
`studies/ae/ae-002-nonlinearity.yaml`.
</details>

**Next:** [Lesson 04](../notebooks/04-regularized-ae.ipynb).
