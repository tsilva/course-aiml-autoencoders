# Lesson 12 — Learn which token arrangements belong together

**Learning objective:** separate learning a visual vocabulary from learning a sampling distribution.

A vocabulary tells us what symbols mean. A prior learns which symbol should
come next, given the symbols already present. Scanning the token grid row by
row lets a small GRU learn these dependencies while the VQ encoder and decoder
stay frozen. Training supplies the true previous tokens; sampling supplies the
prior's own previous choices.

[Open the notebook](../notebooks/12-code-prior.ipynb). Run its cells in order for the experiment and evidence.

## Predict before running

Will beating a uniform token predictor prove that the prior learned spatial dependencies?

<details>
<summary>Reveal the expected reasoning</summary>

No. Frequent symbols alone can beat uniform prediction. Compare with a training-frequency unigram baseline as well as inspecting generated arrangements.
</details>

## Run the revealing experiment

The notebook uses short CPU runs and reuses exact matching completed evidence.
It prints the run path and measured duration. Keep `PROFILE = "quick"` for the
core path; use `"full"` for the original budget. No worksheet is required.

## Advancement gate — transfer check

The prior improves dramatically, but generated images still lack fine detail. Which stage might be the bottleneck?

<details>
<summary>Reveal the expected reasoning</summary>

The VQ encoder/codebook/decoder may have discarded that detail. A better token distribution cannot recover information absent from the representation.
</details>

<details>
<summary>Optional: go deeper</summary>

Raster order gives the factorization
$p(k_1,\ldots,k_N)=\prod_i p(k_i\mid k_{<i})$.
Teacher forcing uses true history during training; sampling uses sampled
history. The small GRU is chosen for clarity, not top-tier image quality.
Temperature and larger priors are optional investigations; preserve the exact
paired VQ checkpoint when comparing them.
</details>

**Next:** [Lesson 13](../notebooks/13-synthesis.ipynb).
