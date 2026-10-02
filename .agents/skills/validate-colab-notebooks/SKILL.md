---
name: validate-colab-notebooks
description: Validate all course notebooks, or a specified lesson, in a real Google Colab runtime, checking complete execution and expected numerical and visual learning results. Use for Colab compatibility checks and notebook verification in this repository.
---

# Validate course notebooks in Google Colab

Default to every lesson in `course/curriculum.yaml`. Accept a lesson ID, notebook
filename, or notebook/source path when the user specifies one. Validation means
both complete execution and evidence supporting the lesson's learning objective.

## Prepare the exact version

From the repository root, rebuild and check the canonical lessons:

```sh
uv run --frozen python scripts/build_notebooks.py
uv run --frozen python scripts/build_notebooks.py --check
```

Generate a self-contained validation notebook and provenance manifest:

```sh
uv run --frozen python .agents/skills/validate-colab-notebooks/scripts/prepare_validation.py
# Single lesson; also accepts 02-linear-ae.ipynb or its repository path:
uv run --frozen python .agents/skills/validate-colab-notebooks/scripts/prepare_validation.py --notebook 02
```

The helper prints paths under ignored `runs/colab-validation/`. It embeds the
selected generated notebooks, current Python sources, recipes, and curriculum.
This tests unpublished working-tree changes without a push. Retain the manifest's
SHA-256 fingerprint; HEAD alone does not identify uncommitted changes.

The harness runs the notebooks' shared first setup cell in an empty Colab session,
then replaces the fetched course code/configuration with the embedded snapshot.
It executes every original code cell in order with fresh lesson globals, sharing
matching training runs only within this validation session. It records outputs,
cell counts, failures, environment versions, and source hashes. The helper refuses
different setup cells: validate those lessons separately so each bootstrap is tested.

## Run in actual Colab

Use the native Codex in-app Browser and its documented browser runtime. Read
[references/browser-workflow.md](references/browser-workflow.md) for upload,
runtime controls, and evidence collection. Local execution or a mocked
`google.colab` import is useful preflight, but does not establish a Colab pass.

1. Open Google Colab and upload the generated validation notebook as a new copy.
   Keep the user's existing notebook intact. Use a fresh runtime with the default
   CPU setup; lessons default to `PROFILE = "quick"` and need no GPU.
2. Run all cells. Complete any Colab trust confirmation for this generated copy.
   Use Colab's installed packages first; do not install a different Torch stack
   merely to make verification pass. Record any required workaround as a failure
   of the ordinary learner path until it is incorporated into canonical setup.
3. Wait for terminal results for every selected lesson. Inspect failed-cell
   tracebacks and the runtime state; a disconnected runtime or partial output is
   incomplete validation. Keep progress updates concise during long training.
4. Collect `report.json`, the output-bearing lesson copies, and relevant plots
   from the harness's runtime directory. Preserve evidence locally under the
   printed validation directory using browser-supported download/file APIs.
   When those APIs are unavailable, copy the visible report/output text and
   save screenshots there; record the runtime artifact paths and Colab URL.

## Verify the learning results

Read the selected lesson's canonical source, including predictions and collapsed
answers, and its row in [references/expected-results.md](references/expected-results.md).
Inspect the actual numerical results and displayed images. The harness provides
execution evidence; it deliberately leaves semantic review pending.

- Every code cell must finish without an exception; counts and notebook hashes
  must match the local manifest. Check that plots render and checkpoints,
  summaries, metric histories, and figures requested by the lesson exist.
- Losses and displayed tensors must be finite, shapes/indices valid, and
  quantitative toy examples must give their stated answers. Inspect learning
  curves and held-out baselines, not just training completion.
- Compare controlled experiments on the same split and metric. Short-run model
  rankings and sample quality are observations, not universal guarantees. If a
  contrast reverses, report the values and determine whether the lesson still
  explains the evidence. Never loosen a check or invent a favorable result.
- Review images for meaningful reconstruction, noise removal, interpolations,
  latent samples, and token maps where relevant. A PNG's existence alone is
  insufficient. Record blurry, identical, blank, or misleading outputs.

Write `review.md` beside the local manifest: snapshot fingerprint, Colab URL,
runtime versions, selected lessons, per-lesson execution counts and observed
outcomes, evidence paths, and **pass / fail / incomplete** with reasons. A lesson
passes only after both execution and expected-result review pass. Unreviewed
lessons remain incomplete even when `report.json` says `execution_pass`.

## Repair and finish

If authorized notebook fixes are needed, edit canonical sources or reusable
modules, rebuild, generate a new snapshot, and rerun affected lessons in a fresh
runtime. Broaden to all lessons when shared setup, dependencies, training helpers,
or data handling changed. Keep evidence out of committed notebooks. Do not push
unless requested. If publication itself is under test, also run the published
GitHub/Colab link directly and verify the fetched commit; a snapshot harness pass
does not certify that an older published notebook has been updated.

Finish with scope, Colab runtime, pass/fail counts, any unresolved expected-result
issue, and links to the review/evidence. If account access or runtime allocation
blocks execution, explain the blocker and mark validation incomplete.
