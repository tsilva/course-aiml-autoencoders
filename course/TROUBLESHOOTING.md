# Course Troubleshooting

## The first real-image cell is slow or cannot download Fashion-MNIST

The dataset downloads once into ignored `data/`. Check network access and
retry the cell. Lesson 00 uses synthetic shapes and needs no download.

## Training or diagnostics are taking time

The core defaults to eight CPU epochs on 2,048 training and 512 validation
images. The run prints measured elapsed time once training and figures finish.
Rerunning a completed configuration reuses it. For a smaller exploratory run,
pass a recipe with reduced `dataset.train_size` and `dataset.validation_size`;
treat its conclusions as provisional. Do not silently change budgets within a
comparison.

## A cell was interrupted

Rerun it. Only completed matching runs with a summary, metrics, configuration,
and best checkpoint are reused. Incomplete evidence remains under `runs/`;
the next attempt creates a fresh directory.

## I want the full experiment or a fresh run

Change the notebook setting to `PROFILE = "full"` and rerun its cells. Keep the
profile consistent across lessons. Use `learn(recipe, rerun=True)` for a fresh
run; setting `PROFILE` alone does not execute anything.

## I lost the run directory

The exact selected paths are recorded in `runs/course/session.json`. The cell
also prints the path whenever evidence is reused. Inspect it with:

```bash
uv run --frozen course-aiml-autoencoders inspect <RUN_DIR>
```

Runs made before the new training-holdout split are intentionally not reused
by course helpers. Do not combine their numbers with the new comparisons.

## AE, VAE, and VQ-VAE losses have different magnitudes

AE/VQ-VAE recipes use mean pixel MSE. VAE recipes use BCE summed per image,
plus KL. Compare named components with matching conventions. VAE validation
reconstructs from posterior centers, so its reconstruction number is not an
estimate of the expected stochastic reconstruction term in the ELBO.

## Denoising results look contradictory

Use `validation/corrupted_mse`, which compares noisy-input output with clean
targets. Both models must use the same evaluation noise. Compare also with
`validation/noisy_input_mse` and inspect the four-row clean-target figure.
Clean-input MSE answers a separate question. A short run may be undertrained.

## KL is near zero or codebook usage is low

Check reconstructions and whether changing the latent changes the output.
Low KL alone does not diagnose posterior collapse. Inspect global code usage
rather than one batch; a code unobserved in this sample is not necessarily
permanently dead. Useful reconstruction can coexist with a small vocabulary.

## A result contradicts the expected reasoning

Check resolved config, ranges, named loss terms, correct checkpoint, and whether
the control actually exhibited the claimed failure. Then choose one smallest
experiment distinguishing implementation error, insufficient training, and a
false hypothesis. Expected explanations are hypotheses, not required outcomes.

## A command cannot find a lesson or recipe

Run from the repository root. Notebook helpers locate that root from a project
subdirectory; the recipe's dataset cache path is resolved to an absolute path.
