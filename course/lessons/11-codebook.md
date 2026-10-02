# Lesson 11 — Count the symbols actually used

**Learning objective:** distinguish vocabulary size, coverage, and balanced usage.

A dictionary with 128 words is not a 128-word conversation if almost every
sentence repeats one word. Codebook size counts available symbols; codes used
counts observed symbols; perplexity summarizes how evenly they are used.
Inspect the run from Lesson 10 before spending time training larger codebooks.

[Open the notebook](../notebooks/11-codebook.ipynb). Run its cells in order for the experiment and evidence.

## Predict before running

Two encoders use all eight symbols. One uses them evenly; the other uses one symbol 99% of the time. Should their perplexities match?

<details>
<summary>Reveal the expected reasoning</summary>

No. Both have full coverage, but the concentrated encoder has much lower effective vocabulary size.
</details>

## Run the revealing experiment

The notebook uses short CPU runs and reuses exact matching completed evidence.
It prints the run path and measured duration. Keep `PROFILE = "quick"` for the
core path; use `"full"` for the original budget. No worksheet is required.

## Advancement gate — transfer check

A model offers 512 codes, uses 40, and has perplexity 9. What do those three numbers establish?

<details>
<summary>Reveal the expected reasoning</summary>

512 is nominal vocabulary, 40 received assignments in the evaluated data, and the uneven usage has effective diversity about 9. None alone establishes reconstruction or downstream quality.
</details>

<details>
<summary>Optional: go deeper</summary>

Full size and commitment sweeps live in
`studies/vqvae/vqvae-001-codebook-size.yaml` and
`studies/vqvae/vqvae-002-commitment.yaml`. Change one factor and compare raw and
weighted loss terms separately. Dead codes are unobserved in the evaluated
sample, not proof they can never be used. EMA updates and dead-code recovery
are further optional experiments.
</details>

**Next:** [Lesson 12](../notebooks/12-code-prior.ipynb).
