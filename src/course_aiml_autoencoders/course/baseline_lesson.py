"""Dataset plumbing for the notebook's mean-image experiment."""

import torch

from course_aiml_autoencoders.course.learning import lesson_config
from course_aiml_autoencoders.data import build_dataloaders


def load_fashion_image_splits(*, profile: str = "quick") -> tuple[torch.Tensor, torch.Tensor]:
    """Load the same training and validation images used by the linear AE lesson."""
    config = lesson_config("recipes/ae/ae-001-linear.yaml", profile=profile)
    train_loader, validation_loader, _ = build_dataloaders(
        config["dataset"], config["training"]
    )
    train_images = torch.cat([images for images, _ in train_loader])
    validation_images = torch.cat([images for images, _ in validation_loader])
    return train_images, validation_images
