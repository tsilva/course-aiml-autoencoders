# Lesson 00 — See what an autoencoder does

**Learning objective:** follow an image through an encoder, a bottleneck, and a decoder.

Imagine sending a picture through a tiny note. The encoder writes the note;
the decoder rebuilds a picture using only that note. Training teaches both sides
the same shorthand. Today we inspect the system *before* it learns.

You need basic Python, tensors as arrays of numbers, and the idea that training
adjusts weights to reduce an error. Probability and gradient details are
introduced when needed. Run cells in order; make a mental prediction, then
expand its explanation. The course moves from reconstruction to sampling and
finally to discrete visual tokens.

[Open the notebook](../notebooks/00-laboratory.ipynb). Run its cells in order for the experiment and evidence.

## Predict before running

An untrained model already has an encoder and decoder. Will its reconstruction resemble the input?

<details>
<summary>Reveal the expected reasoning</summary>

Usually not: the shape of the system is in place, but the weights have not learned a shared shorthand.
</details>

## Run the revealing experiment

Lesson 00 inspects untrained reconstruction on synthetic shapes. No dataset
download or training is needed.

## Advancement gate — transfer check

A colleague shows you a model with perfect tensor shapes and no training history. What evidence is still missing?

<details>
<summary>Reveal the expected reasoning</summary>

Evidence that training produces input-dependent reconstructions on held-out images and improves on an input-ignoring baseline.
</details>

<details>
<summary>Optional: go deeper</summary>

For implementation trust checks, run `uv run --frozen python -m pytest`.
The model returns reconstruction, latent, and specialized extras so one generic
trainer can serve all three families. Models and objectives own their math;
notebooks configure experiments and inspect evidence.
</details>

**Next:** [Lesson 01](../notebooks/01-mean-baseline.ipynb).
