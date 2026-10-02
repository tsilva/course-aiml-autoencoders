from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn

from course_aiml_autoencoders.objectives import Objective
from course_aiml_autoencoders.diagnostics.reconstructions import per_example_mse


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    objective: Objective,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    examples = 0
    for inputs, _labels in batches:
        inputs = inputs.to(device)
        losses = objective(model(inputs), inputs)
        batch_size = inputs.shape[0]
        examples += batch_size
        for name, value in losses.items():
            totals[name] += float(value.detach().cpu()) * batch_size
    return {name: value / examples for name, value in totals.items()}


@torch.no_grad()
def evaluate_reconstruction_mse(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    *,
    transform: Callable[[Tensor], Tensor] | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Score against clean targets, with identical CPU corruption for each model."""

    model.eval()
    total = input_total = 0.0
    examples = 0
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for clean, _labels in batches:
            clean = clean.cpu()
            inputs = transform(clean) if transform else clean
            reconstruction = model(inputs.to(device)).reconstruction.cpu()
            total += float(per_example_mse(clean, reconstruction).sum())
            input_total += float(per_example_mse(clean, inputs).sum())
            examples += len(clean)
    if not examples:
        raise ValueError("Cannot evaluate an empty loader")
    return {"mse": total / examples, "input_mse": input_total / examples}
