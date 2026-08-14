# Canonical notebook sources

These Jupytext `py:percent` files are the source of truth for the executable
course. Edit them instead of the generated `.ipynb` files.

Build and verify:

```bash
uv run python scripts/build_notebooks.py
uv run python scripts/build_notebooks.py --check
```

Every lesson must preserve the learning loop:

1. State one learning objective.
2. Give the learner an intuitive mental model or metaphor.
3. Build understanding through small, incremental, visible examples.
4. Ask for a mental prediction before revealing evidence.
5. Expose relevant tensors, metrics, images, and failure behavior.
6. Explain what the model is for, how to use it, and its pros and cons.
7. Use math when it resolves a real question; keep nonessential derivations in
   optional collapsed sections rather than making them the main path.
8. Change one declared experimental variable when comparing experiments.
9. End with a collapsed advancement gate and expected reasoning.

Keep reusable model, objective, training, and plotting logic in
`src/course_aiml_autoencoders`.

Do not add blank “Your answer” or “Prediction” fields. Put the expected
reasoning in a collapsed `<details>` block directly beneath the question.
