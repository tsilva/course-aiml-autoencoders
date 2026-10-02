from functools import partial

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from course_aiml_autoencoders.diagnostics import plot_reconstruction_grid
from course_aiml_autoencoders.models.outputs import LatentModelOutput
from course_aiml_autoencoders.training.evaluator import evaluate_reconstruction_mse
from course_aiml_autoencoders.training.trainer import corrupt_inputs


class CopyModel(nn.Module):
    def forward(self, images):
        return LatentModelOutput(images, images)


class CleanModel(nn.Module):
    def forward(self, images):
        clean = torch.zeros_like(images)
        return LatentModelOutput(clean, clean)


def test_denoising_score_rewards_clean_recovery_and_matches_noise():
    loader = DataLoader(TensorDataset(torch.zeros(5, 1, 4, 4), torch.zeros(5)), batch_size=2)
    corruption = partial(corrupt_inputs, config={"kind": "gaussian", "standard_deviation": 0.3})
    state = torch.get_rng_state().clone()
    recovered = evaluate_reconstruction_mse(CleanModel(), loader, torch.device("cpu"), transform=corruption)
    copied = evaluate_reconstruction_mse(CopyModel(), loader, torch.device("cpu"), transform=corruption)
    assert recovered["mse"] == 0
    assert recovered["input_mse"] > 0
    assert copied["mse"] == recovered["input_mse"]
    assert copied["input_mse"] == recovered["input_mse"]
    assert torch.equal(torch.get_rng_state(), state)


def test_denoising_figure_scores_and_displays_the_clean_target():
    clean = torch.zeros(1, 1, 4, 4)
    noisy = torch.ones_like(clean)
    figure = plot_reconstruction_grid(noisy, clean, targets=clean, include_error=True)
    assert len(figure.axes) == 4
    assert figure.axes[0].get_ylabel() == "Clean target"
    assert figure.axes[0].get_title() == "MSE 0.0000"
    assert figure.axes[-1].images[0].get_array().sum() == 0
    plt.close(figure)
