# Lesson 09 — Check whether the decoder uses the note

**Learning objective:** diagnose latent dependence before trying collapse remedies.

A decoder can appear to work while ignoring the note. To test dependence,
compare normal reconstruction with decoding one fixed zero note for every
input. The fixed-note output is a deliberately broken control: it illustrates
loss of information and does not prove the trained model collapsed.

[Open the notebook](../notebooks/09-collapse.ipynb). Run its cells in order for the experiment and evidence.

## Predict before running

If every input receives the same note, can the decoder reconstruct their different details?

<details>
<summary>Reveal the expected reasoning</summary>

A deterministic decoder receives identical inputs and produces identical outputs. Normal reconstructions should preserve more input-specific information if the learned latent is used.
</details>

## Run the revealing experiment

The notebook uses short CPU runs and reuses exact matching completed evidence.
It prints the run path and measured duration. Keep `PROFILE = "quick"` for the
core path; use `"full"` for the original budget. No worksheet is required.

## Advancement gate — transfer check

Run A has low KL and distinct, accurate reconstructions; Run B has low KL and constant reconstructions. Which deserves a collapse investigation?

<details>
<summary>Reveal the expected reasoning</summary>

Run B. In A, investigate whether little information is being used efficiently. Confirm B with latent interventions and posterior variation; the shared low KL number is not enough.
</details>

<details>
<summary>Optional: go deeper</summary>

Free bits optimizes a per-dimension floor $\max(\lambda, KL_j)$.
Below the floor, the compression term supplies no further gradient; it does not
force a dimension to encode information. Raw KL and optimized effective KL can
differ. Run `studies/vae/vae-002-collapse-remedies.yaml` to compare immediate KL,
warm-up, and free bits. The small decoder may not collapse: report that honestly.
</details>

**Next:** [Lesson 10](../notebooks/10-vqvae.ipynb).
