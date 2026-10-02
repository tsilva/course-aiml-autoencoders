from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as pyplot
import torch
from matplotlib.figure import Figure
from torch import Tensor


def per_example_mse(inputs: Tensor, reconstructions: Tensor) -> Tensor:
    """Return one channel-and-pixel-averaged MSE value per example."""

    if inputs.shape != reconstructions.shape:
        raise ValueError(
            "Inputs and reconstructions must have the same shape, "
            f"got {tuple(inputs.shape)} and {tuple(reconstructions.shape)}"
        )
    return (inputs - reconstructions).square().flatten(start_dim=1).mean(dim=1)


def _show_tensor_image(axis, image: Tensor, *, cmap: str = "gray") -> None:
    rendered = image.detach().cpu().clamp(0, 1)
    if rendered.shape[0] == 1:
        axis.imshow(
            rendered.squeeze(0).numpy(), cmap=cmap, vmin=0, vmax=1
        )
    else:
        axis.imshow(rendered.permute(1, 2, 0).numpy())
    axis.axis("off")


def plot_reconstruction_grid(
    inputs: Tensor,
    reconstructions: Tensor,
    *,
    labels: Tensor | None = None,
    class_names: Sequence[str] | None = None,
    max_items: int = 8,
    include_error: bool = False,
    targets: Tensor | None = None,
) -> Figure:
    """Create an input/reconstruction comparison for files or notebooks."""

    targets = inputs if targets is None else targets
    if inputs.shape != targets.shape:
        raise ValueError("Inputs and targets must have the same shape")
    per_item_error = per_example_mse(targets, reconstructions)
    count = min(max_items, inputs.shape[0])
    show_targets = targets is not inputs
    rows = 2 + int(show_targets) + int(include_error)
    figure, axes = pyplot.subplots(
        rows,
        count,
        figsize=(1.8 * count, 2.1 * rows),
        squeeze=False,
    )
    for index in range(count):
        if show_targets:
            _show_tensor_image(axes[0, index], targets[index])
        input_row = int(show_targets)
        reconstruction_row = input_row + 1
        _show_tensor_image(axes[input_row, index], inputs[index])
        _show_tensor_image(axes[reconstruction_row, index], reconstructions[index])
        title = f"MSE {per_item_error[index]:.4f}"
        if labels is not None:
            label = int(labels[index])
            name = (
                class_names[label]
                if class_names is not None and label < len(class_names)
                else str(label)
            )
            title = f"{name}\n{title}"
        axes[0, index].set_title(title, fontsize=9)
        if include_error:
            error = (targets[index] - reconstructions[index]).square()
            if error.shape[0] == 1:
                axes[-1, index].imshow(
                    error.squeeze(0).detach().cpu().numpy(),
                    cmap="magma",
                    vmin=0,
                    vmax=1,
                )
            else:
                axes[-1, index].imshow(
                    error.mean(dim=0).detach().cpu().numpy(),
                    cmap="magma",
                    vmin=0,
                    vmax=1,
                )
            axes[-1, index].axis("off")

    if show_targets:
        axes[0, 0].set_ylabel("Clean target")
    axes[int(show_targets), 0].set_ylabel("Noisy input" if show_targets else "Input")
    axes[int(show_targets) + 1, 0].set_ylabel("Reconstruction")
    if include_error:
        axes[-1, 0].set_ylabel("Clean-target error" if show_targets else "Squared error")
    # axis('off') hides axis labels, so render the row labels as visible text too.
    for row in range(rows):
        axes[row, 0].text(
            -0.08, 0.5, axes[row, 0].get_ylabel(),
            transform=axes[row, 0].transAxes, rotation=90,
            va="center", ha="right", fontsize=10,
        )
    figure.suptitle("Clean targets, noisy inputs, and reconstructions" if show_targets
                   else "Inputs and reconstructions")
    figure.tight_layout()
    return figure


def plot_image_grid(
    images: Tensor,
    *,
    labels: Tensor | None = None,
    class_names: Sequence[str] | None = None,
    title: str | None = None,
    max_items: int = 10,
) -> Figure:
    """Create a labelled single-row image grid."""

    count = min(max_items, images.shape[0])
    figure, axes = pyplot.subplots(
        1, count, figsize=(1.8 * count, 2.2), squeeze=False
    )
    for index in range(count):
        _show_tensor_image(axes[0, index], images[index])
        if labels is not None:
            label = int(labels[index])
            name = (
                class_names[label]
                if class_names is not None and label < len(class_names)
                else str(label)
            )
            axes[0, index].set_title(name, fontsize=9)
    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def save_reconstruction_grid(
    inputs: Tensor,
    reconstructions: Tensor,
    path: Path,
    max_items: int = 8,
    *,
    labels: Tensor | None = None,
    class_names: Sequence[str] | None = None,
    include_error: bool = False,
    targets: Tensor | None = None,
) -> None:
    figure = plot_reconstruction_grid(
        inputs,
        reconstructions,
        labels=labels,
        class_names=class_names,
        max_items=max_items,
        include_error=include_error,
        targets=targets,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    pyplot.close(figure)
