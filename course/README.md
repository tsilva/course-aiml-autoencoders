# From Bottlenecks to Discrete Latents

Learn the mechanisms with one question, one prediction, one small experiment,
one explanation, and one transfer check per lesson. Start with a reconstruction
baseline, learn continuous latents, then learn discrete tokens and their prior.

## Start here

You need basic Python, arrays/tensors, and the idea that training adjusts
weights to reduce an error. Probability and gradient details arrive when they
help explain behavior. Python 3.11 or 3.12 and uv are required.

From the repository root:

```bash
uv sync --frozen
uv run --frozen python -m jupyterlab course/notebooks/00-laboratory.ipynb
```

Open Jupyter's printed URL. Lesson 00 runs without a dataset download or
training. Follow each notebook's **Next** link and run cells in order.

1. Make the prediction mentally before revealing the expected reasoning.
2. Run the small experiment and inspect its images or named metrics.
3. Reconcile a contradictory result with the explanation.
4. Answer the brief transfer check and expand its reasoning.

No worksheets, code-reading assignments, full sweeps, or seed reports are
required for the core path. Optional collapsed sections contain the math and
larger investigations. If a check exposes a gap, revisit that example before
continuing; there is no grading ceremony.

## Short runs and immediate evidence

Small examples expose pixel error, stochastic clouds, quantization, and uneven
symbol use before expensive training. They are illustrative probes, not saved
training results. Distributed notebooks remain output-free.

The default `PROFILE = "quick"` uses the original architecture and objective,
up to eight CPU epochs, 2,048 training images, and 512 validation images. Full
studies are optional. Short runs teach the mechanism; their rankings, samples,
and negative results may change with more training.

`learn(...)` reuses a completed run only when the resolved configuration,
training source fingerprint, and artifacts match. `learn_prior(...)` also
checks the exact VQ checkpoint path and content hash. Both print the run path
and measured training-plus-diagnostics time on its recorded device. Repeated
lessons reuse that same evidence rather than selecting an unrelated latest run.
A local run index is saved under ignored `runs/course/session.json`.

- Set `PROFILE = "full"` for the original recipe budget.
- Use `learn(..., rerun=True)` to generate fresh evidence.
- Keep all lessons on the same profile for consistent comparisons.
- Quick training caps CPU threads at four and restores the previous setting.
- The first Fashion-MNIST use downloads the dataset once into ignored `data/`.
- A failed or incomplete run is retried, not reused as completed evidence.

Curriculum time estimates are approximate **reading and thinking time**, not
hardware benchmarks. Actual compute time is printed for your device. See
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) for first-run downloads, interruptions,
and ways to revisit exact evidence.

## Core course map

| Lessons | Main question | Extra work is optional |
|---|---|---|
| 00–01 | What does reconstruction mean, and what baseline should it beat? | Pipeline contracts and tests |
| 02–03 | What do eight sliders preserve, and how does nonlinearity help? | Decoder directions and latent-size sweep |
| 04–05 | What task does denoising optimize, and why does an AE lack a sampling rule? | Sparsity and latent projections |
| 06–09 | How do clouds and prior pressure enable sampling, and how can they lose information? | Gradient derivations, beta sweep, collapse remedies |
| 10–12 | How do symbols learn, get used, and form coherent arrangements? | Codebook/commitment sweeps and larger priors |
| 13 | Which mechanism fits the task, and what would falsify that choice? | Multi-seed capstone and downstream probe |

List lessons or open a concise reference:

```bash
uv run --frozen course-aiml-autoencoders course
uv run --frozen course-aiml-autoencoders course 06
```

For interactive guidance, ask: “Start lesson 06 with me. Let me predict before
you reveal the explanation.” Share the printed run directory when discussing
evidence; it contains the resolved config, metrics, best checkpoint, and figures.

## Evidence that answers the question

Training and selection use disjoint, seeded subsets of the original training
split. The official test split is reserved for optional final confirmation via
`build_test_dataloader`; never use it to choose checkpoints or tune variants.
Older runs used a different split and should not be mixed with new evidence.

Denoising compares models on identical seeded corruption and scores output
against the **clean target**. Its figure shows clean target, noisy input,
reconstruction, and clean-target error. Clean-input MSE remains a separate
measure. Code priors are compared with uniform and training-frequency unigram
baselines, because learning frequent tokens alone can beat uniform prediction.

Low KL, high codebook usage, attractive samples, and a small total loss are
individually incomplete evidence. Match the diagnostic to the task. Total
objectives across AE, VAE, and VQ-VAE use different terms and scales.

## Optional research path

Choose one extension that answers your next question:

- Run one declared-variable study from `studies/`.
- Confirm an important close comparison with seeds 0, 1, and 2.
- Freeze an encoder and compare a classification or retrieval probe with raw
  pixels and PCA, using identical splits.
- Preserve a conclusion using [WORKSHEET.md](WORKSHEET.md) in `reports/`.

<details>
<summary>Commands for standalone experiments</summary>

Run these from the repository root:

```bash
uv run --frozen course-aiml-autoencoders train recipes/smoke/fake-ae.yaml --device cpu
uv run --frozen course-aiml-autoencoders train recipes/ae/ae-001-linear.yaml
uv run --frozen course-aiml-autoencoders study studies/ae/ae-003-latent-capacity.yaml --seeds 0
uv run --frozen course-aiml-autoencoders inspect <RUN_DIR>
```

</details>

Keep studies (questions), recipes (configurations), reports (conclusions), and
ignored runs (evidence) distinct. See [GLOSSARY.md](GLOSSARY.md) and
[the experiment protocol](../docs/EXPERIMENT_PROTOCOL.md) when needed.

## Authoring and execution

Edit canonical `course/notebook_sources/*.py`; never hand-edit generated
`course/notebooks/*.ipynb`. Keep reusable computation under
`src/course_aiml_autoencoders`. Use concise questions followed by collapsed
expected reasoning, with no blank answer fields.

```bash
uv run --frozen python scripts/build_notebooks.py
uv run --frozen python scripts/build_notebooks.py --check
uv run --frozen python -m pytest
```

To execute output-bearing copies under ignored `runs/notebook-executions/`:

```bash
uv run --frozen python scripts/execute_notebooks.py --profile smoke
uv run --frozen python scripts/execute_notebooks.py --profile foundations
uv run --frozen python scripts/execute_notebooks.py --profile all
```

Notebook conversion is deterministic and output-free. Execution creates your
own evidence; it never overwrites the distributed notebooks. Explicit lesson
sets work independently: prerequisite experiments are created or reused by
configuration when required.
