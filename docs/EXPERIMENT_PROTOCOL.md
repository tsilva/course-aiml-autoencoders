# Experiment Protocol

For a core lesson, make one mental prediction, inspect one revealing contrast,
and answer one transfer check. A written report is optional. Short runs explain
mechanisms; they do not establish close numerical rankings.

For a study or durable report, state:

1. The question and falsifiable hypothesis.
2. The baseline recipe and one declared independent variable.
3. Controlled variables, including data split, evaluation inputs, and training budget.
4. The metric and figures that answer the question.
5. The observed result, its limitations, and the decision.

Use a fixed seeded holdout from the original training split for validation and
checkpoint selection. Reserve the official test set for confirmation after
selection. Do not tune on test results or mix runs from the older test-as-validation
protocol with the new training-holdout protocol.

Denoising must compare output with clean targets on identical corrupted
validation inputs. Report corrupted-input MSE, noise-only MSE, and clean-input
MSE separately. The four-row figure uses clean targets for its error map.

Do not rank AE, VAE, and VQ-VAE total objectives: observation losses, reductions,
and auxiliary terms differ. VAE validation decodes posterior centers for stable
inspection; its reconstruction term is not a Monte Carlo ELBO estimate.
Average raw KL also includes aggregate-posterior mismatch, so positive KL is
not proof of input-dependent information. Check latent interventions and images.

Use one seed for exploration. Confirm close important conclusions with three
or more seeds. Record exact run paths and resolved configs; pair a token prior
with its exact frozen VQ checkpoint. The course helpers validate configs, source
fingerprints, and checkpoint identity before reuse.

Reusable model mathematics stays in model/objective modules and training logic
stays in the trainer. Before trusting a new model, verify shapes, gradients,
tiny-overfit behavior, and fixed reconstructions. Store generated evidence in
ignored `runs/`; reports preserve conclusions, not raw checkpoints or datasets.
