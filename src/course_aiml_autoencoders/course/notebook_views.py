"""Data preparation and presentation helpers for the course notebooks."""

from pathlib import Path

import matplotlib.pyplot as plt

from course_aiml_autoencoders.config import load_yaml
from course_aiml_autoencoders.course.probes import balanced_class_batch, repository_root
from course_aiml_autoencoders.data import build_dataloaders
from course_aiml_autoencoders.data.datasets import dataset_spec
from course_aiml_autoencoders.models import build_model


def untrained_example_model():
    """Build the small untrained model used to introduce the course pipeline."""
    recipe = repository_root() / "recipes/smoke/fake-ae.yaml"
    config = load_yaml(recipe)
    spec = dataset_spec(config["dataset"])
    return build_model(config["model"], spec).eval(), spec.input_shape


def validation_examples(config):
    """Select a balanced set from the run's held-out split."""
    _, validation, spec = build_dataloaders(config["dataset"], config["training"])
    images, _ = balanced_class_batch(validation, spec.num_classes)
    return images


def show_run_figure(run_dir: Path, name: str) -> None:
    """Display one saved diagnostic from the selected run."""
    from IPython.display import Image, display

    display(Image(filename=str(run_dir / "figures" / name)))


def plot_sample_clouds(clouds, center):
    """Compare two-dimensional samples around a shared center."""
    _, axis = plt.subplots()
    for label, samples in clouds:
        axis.scatter(samples[:, 0], samples[:, 1], s=8, alpha=0.4, label=label)
    axis.scatter(*center, marker="x", c="black", label="center")
    axis.set_aspect("equal")
    axis.legend()
    axis.set_title("Same center, two different clouds")
    plt.show()


def plot_codebook_snap(points, embeddings, chosen):
    """Draw a line from each encoder vector to its selected codebook entry."""
    _, axis = plt.subplots()
    axis.scatter(points[:, 0], points[:, 1], label="encoder vectors")
    axis.scatter(embeddings[:, 0], embeddings[:, 1], marker="s", s=100, label="codebook")
    for point, symbol in zip(points, chosen, strict=True):
        axis.plot([point[0], symbol[0]], [point[1], symbol[1]], color="gray")
    axis.legend()
    axis.set_aspect("equal")
    axis.set_title("Snapping to symbols")
    plt.show()


def plot_rate_distortion(observations):
    """Label observed rate–distortion pairs with their beta values."""
    _, axis = plt.subplots()
    for beta, rate, distortion in observations:
        axis.scatter(rate, distortion)
        axis.annotate(f"beta={beta:g}", (rate, distortion))
    axis.set_xlabel("Raw KL, nats/image")
    axis.set_ylabel("Center-decoded reconstruction BCE/image")
    axis.set_title("Observed tradeoffs from short runs")
    plt.show()
