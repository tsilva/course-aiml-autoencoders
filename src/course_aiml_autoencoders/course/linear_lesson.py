"""Data preparation for the linear autoencoder notebook."""

from course_aiml_autoencoders.course.probes import balanced_class_batch
from course_aiml_autoencoders.data import build_dataloaders, class_names


def prepare_linear_lesson_data(config):
    """Rebuild the run's splits and pick a balanced set of validation images."""
    dataset = config["dataset"]
    train, validation, spec = build_dataloaders(dataset, config["training"])
    images, labels = balanced_class_batch(validation, spec.num_classes)
    return train, validation, images, labels, class_names(dataset["name"])
