---
name: autoencoders-course_author-notebooks
description: Author or refactor this autoencoder course's Jupytext lessons into small runnable steps with visible results, editable experiments, and a collapsed setup cell.
---

# Author course notebooks

Use [Lesson 01](../../../course/notebook_sources/01-mean-baseline.py) as the reference for pacing and presentation. Read the target lesson and its guide first. Preserve its learning objective, experiment settings, data split, and expected evidence unless the user asks to change them.

1. Edit the canonical `course/notebook_sources/*.py` file. Start with one code cell tagged `course-setup` that uses the Lesson 01 checkout loader and runs the imports. The builder collapses the cell in Colab and JupyterLab. Have it display a short table of the lesson's learner-facing imports and what each does.
2. Give each cell one small job and a visible result, such as a tensor, a number, or a plot. Put dataset, configuration, and checkpoint plumbing in reusable course helpers where it obscures the lesson. Keep the operations the learner is meant to understand in view.
3. Use one operation per line and name intermediate values. Put a brief comment above each step with one blank line before every comment after the first in the cell. Show learners how to inspect a value by putting its name on the last line of a cell.
4. Use a few meaningful emojis in headings and mark editable probes with **🧪 Try**. State what to change, what to look for, and which cells to rerun. Keep comparisons on the same data and vary one quantity at a time. Use a collapsed answer for prediction and transfer questions.
5. Rebuild with `uv run --frozen python scripts/build_notebooks.py` and check with `uv run --frozen python scripts/build_notebooks.py --check`. Execute the target with `uv run --frozen python scripts/execute_notebooks.py --lessons NN`. Review the outputs and exercise an editable probe when applicable. Keep executed notebooks and training evidence in ignored `runs/`. For a real Colab runtime check, use the project's Colab validation skill.
