"""Notebook-facing helpers for inspecting experiments without duplicating logic."""

from course_aiml_autoencoders.course.probes import (
    balanced_class_batch,
    latest_run_dir,
    load_metrics,
    load_run_summary,
    load_trained_model,
    plot_metric_history,
    repository_root,
)
from course_aiml_autoencoders.course.learning import learn, learn_prior, lesson_config

__all__ = [
    "balanced_class_batch",
    "latest_run_dir",
    "load_metrics",
    "load_run_summary",
    "load_trained_model",
    "plot_metric_history",
    "repository_root",
    "learn",
    "learn_prior",
    "lesson_config",
]
