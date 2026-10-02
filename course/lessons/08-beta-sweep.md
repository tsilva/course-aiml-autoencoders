# Lesson 08 — Change the price of information

**Learning objective:** interpret beta through a reconstruction/prior tradeoff.

Beta is the price charged for moving a cloud away from the shared prior.
Compare one familiar price with a higher one. Use the same architecture,
split, optimizer, and training budget; change only beta.

[Open the notebook](../notebooks/08-rate-distortion.ipynb). Run its cells in order for the experiment and evidence.

## Predict before running

If beta rises from 1 to 4, which should usually decrease: reconstruction error or raw KL?

<details>
<summary>Reveal the expected reasoning</summary>

Raw KL should tend to decrease. Reconstruction error may rise because retaining input-specific detail costs more. Optimization can violate the trend.
</details>

## Run the revealing experiment

The notebook uses short CPU runs and reuses exact matching completed evidence.
It prints the run path and measured duration. Keep `PROFILE = "quick"` for the
core path; use `"full"` for the original budget. No worksheet is required.

## Advancement gate — transfer check

The beta-four model has a lower total loss but worse reconstruction. Is it the winner?

<details>
<summary>Reveal the expected reasoning</summary>

The totals optimize differently weighted objectives. Compare named components and the behavior required by the task; lower total loss does not settle the choice.
</details>

<details>
<summary>Optional: go deeper</summary>

For beta 0, 0.1, 1, and 4, run
`uv run course-aiml-autoencoders study studies/vae/vae-001-beta-sweep.yaml --seeds 0`.
Repeat only a meaningful finalist contrast across seeds if you want a durable
numerical conclusion. Predict sample diversity as well as prior compatibility.
</details>

**Next:** [Lesson 09](../notebooks/09-collapse.ipynb).
