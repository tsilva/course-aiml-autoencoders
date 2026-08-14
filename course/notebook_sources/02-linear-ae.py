# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lesson 02 — Rebuilding a garment from eight sliders
#
# **Learning objective:** understand a linear autoencoder as a learned
# compression system: what the encoder and decoder do, what an eight-number
# bottleneck forces the model to preserve, how to judge whether it worked, and
# why reconstruction ability does not automatically make it a generator.
#
# By the end, you should be able to explain:
#
# - what information flows through an autoencoder;
# - why a bottleneck can force useful compression;
# - what a linear autoencoder preserves and loses;
# - why it beats the mean-image baseline;
# - why smooth interpolation and random generation are different abilities.
#
# Lesson 01 gave every input the same ghostly poster. This lesson lets the model
# inspect each input and write an eight-number reconstruction instruction.

# %%
import math

import matplotlib.pyplot as plt
import torch

from latent_lab.config import load_yaml
from latent_lab.course import (
    balanced_class_batch,
    load_metrics,
    load_trained_model,
    plot_metric_history,
    repository_root,
)
from latent_lab.data import build_dataloaders, class_names
from latent_lab.diagnostics import (
    per_example_mse,
    plot_image_grid,
    plot_reconstruction_grid,
)
from latent_lab.diagnostics.interpolations import linear_interpolation
from latent_lab.models import build_model
from latent_lab.training import (
    compute_mean_image,
    constant_reconstruction_errors,
    run_training,
)
from latent_lab.training.seeding import resolve_device

ROOT = repository_root()
recipe_path = ROOT / "recipes/ae/ae-001-linear.yaml"
config = load_yaml(recipe_path)
device = resolve_device(config["training"]["device"])
train_loader, validation_loader, spec = build_dataloaders(
    config["dataset"], config["training"]
)
names = class_names(config["dataset"]["name"])
config

# %% [markdown]
# ## 1. The coat-check ticket metaphor
#
# Imagine checking a 784-piece garment at a coat desk. The attendant cannot pass
# all 784 pixel values through the tiny ticket window. They must describe the
# garment using only **eight numbers**.
#
# Two workers cooperate:
#
# - The **encoder** looks at the full image and writes the eight-number ticket.
# - The **decoder** sees only the ticket and tries to rebuild the image.
#
# ```
# garment image ──> encoder ──> 8-number ticket ──> decoder ──> reconstruction
# ```
#
# The ticket is called the **latent code** or **bottleneck**. If the rebuilt
# garment is good, those eight numbers must carry useful information about the
# input. Unlike Lesson 01's mean poster, a different input can now receive a
# different ticket.
#
# This is lossy compression: the model must decide what is worth preserving when
# almost 99% of the original coordinates cannot pass through directly.

# %%
pixels_per_image = math.prod(spec.input_shape)
latent_dimensions = int(config["model"]["latent_dim"])
compression_fraction = latent_dimensions / pixels_per_image

print("input values:", pixels_per_image)
print("latent values:", latent_dimensions)
print(f"coordinates retained: {compression_fraction:.2%}")
print(
    "shape path:",
    f"B×1×28×28 → B×{pixels_per_image} → "
    f"B×{latent_dimensions} → B×{pixels_per_image} → B×1×28×28",
)

# %% [markdown]
# The image is flattened from a 28-by-28 grid into 784 values, compressed to
# eight, expanded back to 784, then reshaped into an image.
#
# “Eight values instead of 784” does **not** mean the model stores eight original
# pixels. Each latent value can summarize a broad pattern spread across the
# image.
#
# <details>
# <summary>Question: why not use a 784-number ticket?</summary>
#
# With enough capacity, the model could learn something close to copying and
# would face little pressure to discover compact regularities. The eight-number
# bottleneck makes exact copying impossible for arbitrary images, forcing the
# encoder to prioritize recurring structure such as silhouette, position, and
# broad brightness.
# </details>

# %% [markdown]
# ## 2. Before learning, the ticket is meaningless
#
# The architecture alone does not know anything about garments. Its connections
# begin with random values. Let us pass three images through an untrained
# autoencoder so we have a true “before” picture.

# %%
example_images, example_labels = balanced_class_batch(
    validation_loader, spec.num_classes
)
untrained_model = build_model(config["model"], spec)
untrained_model.eval()
with torch.no_grad():
    untrained_output = untrained_model(example_images[:3])

print("untrained tickets:")
print(untrained_output.latent.round(decimals=2))
_ = plot_reconstruction_grid(
    example_images[:3],
    untrained_output.reconstruction,
    labels=example_labels[:3],
    class_names=names,
    max_items=3,
    include_error=True,
)

# %% [markdown]
# The encoder already emits eight numbers, but they do not yet form a useful
# language between encoder and decoder. A bottleneck creates **pressure** to
# compress; training is what teaches the two sides a shared code.
#
# During training, the model repeatedly:
#
# 1. encodes a garment into eight values;
# 2. decodes those values into an image;
# 3. receives a pixel-accountant MSE score;
# 4. adjusts both workers so the next reconstruction is a little closer.
#
# No class labels are used. “T-shirt,” “shoe,” and “bag” never appear in the
# training objective. The only task is to reconstruct the input.

# %% [markdown]
# ## 3. What makes this autoencoder linear?
#
# This model has one encoder layer and one decoder layer with no nonlinear
# hidden activation between them. Each output is made by adding together scaled
# versions of learned patterns.
#
# A helpful picture is an image-editing panel with eight sliders:
#
# - the encoder chooses the eight slider positions for an input;
# - the decoder combines the eight slider effects into a reconstruction.
#
# The sliders are learned. We do not name one “sleeve length” or another “shoe
# height,” and the model is not required to organize them so neatly.
#
# <details>
# <summary>Optional math: the complete linear autoencoder</summary>
#
# After flattening the image, the encoder and decoder compute:
#
# $$
# z=W_e x+b_e,\qquad \hat{x}=W_dz+b_d.
# $$
#
# Substituting the encoder into the decoder shows that the entire reconstruction
# is one affine transformation of the input. Because $z$ has only eight
# coordinates, every reconstruction lies on an at-most-eight-dimensional flat
# sheet inside the 784-dimensional pixel space.
# </details>
#
# <details>
# <summary>Optional connection: why people compare this with PCA</summary>
#
# PCA finds the best low-dimensional **flat** description of centered data under
# squared reconstruction error. A suitably optimized linear autoencoder spans
# the same best flat subspace. Its individual slider axes need not equal PCA's
# axes: the model may rotate or rescale the coordinate system while preserving
# the same reconstructable sheet.
# </details>

# %% [markdown]
# ## 4. Predict what training can and cannot accomplish
#
# Before running training, form a mental picture:
#
# - Will eight input-dependent numbers beat Lesson 01's one-size-fits-all poster?
# - Which details will survive?
# - Which details will be sacrificed?
# - Will the result look sharp or like a simplified sketch?
#
# <details>
# <summary>Reveal the expected intuition</summary>
#
# The model should beat the mean image because it can tailor the eight slider
# positions to each input. Broad silhouette, location, and brightness should
# survive because they reduce error across many pixels. Fine edges, texture,
# logos, and unusual details are expensive to describe with only eight linear
# coordinates, so reconstructions should look smooth or blurry.
# </details>

# %% [markdown]
# ## 5. Let the encoder and decoder invent their shared language
#
# The recipe trains for 20 passes through the training set. The reusable trainer
# writes the checkpoint, metrics, figures, and resolved recipe under `runs/`.

# %%
result = run_training(config, run_root=ROOT / "runs")
run_dir = result.run_dir
print("run directory:", run_dir)
print("best epoch:", result.best_epoch)
print("best validation MSE:", result.best_validation_loss)

# %% [markdown]
# ## 6. Watch learning happen, not just finish
#
# A final score is like seeing only the last frame of a race. The curves show
# whether learning was steady and whether the model behaved similarly on images
# it practiced on and held-out validation images.
#
# - A falling training curve means the model is improving on its practice set.
# - A falling validation curve means that improvement transfers to unseen data.
# - A widening gap would warn that the model is specializing to training images.

# %%
records = load_metrics(run_dir)
_ = plot_metric_history(
    records,
    ["train/reconstruction_loss", "validation/reconstruction_loss"],
    title="The eight-number language improves over training",
)

# %% [markdown]
# <details>
# <summary>Question: what curve shape would make you distrust the final score?</summary>
#
# Examples include a validation curve that rises while training loss falls,
# violent instability, no improvement over the initial epochs, or a large gap
# between training and validation. A single best number could hide all of these
# behaviors.
# </details>

# %% [markdown]
# ## 7. Inspect what eight numbers preserve
#
# We now load the best checkpoint and give it one unseen example from every
# class. Read the grid vertically: input, reconstruction, then squared error.

# %%
model, resolved_config = load_trained_model(run_dir, device=device)
train_loader, validation_loader, spec = build_dataloaders(
    resolved_config["dataset"], resolved_config["training"]
)
inputs, labels = balanced_class_batch(validation_loader, spec.num_classes)
with torch.no_grad():
    output = model(inputs.to(device))

reconstructions = output.reconstruction.cpu()
latents = output.latent.cpu()

print("image shape:", tuple(inputs.shape))
print("ticket shape:", tuple(latents.shape))
print(
    "raw reconstruction range:",
    (float(reconstructions.min()), float(reconstructions.max())),
)
_ = plot_reconstruction_grid(
    inputs,
    reconstructions,
    labels=labels,
    class_names=names,
    max_items=spec.num_classes,
    include_error=True,
)

# %% [markdown]
# Look for a consistent tradeoff:
#
# - broad garment category and pose are often recognizable;
# - large light and dark regions are placed approximately correctly;
# - edges are softened;
# - small logos, thin straps, soles, and unusual contours are lost.
#
# The model is not trying to produce a sharp plausible garment at any cost. MSE
# rewards matching the particular input pixel by pixel, and the bottleneck gives
# it only eight linear controls. A blurry compromise is often the safest answer.
#
# The raw output range may extend slightly below zero or above one. This decoder
# has no final sigmoid to fence its values into the dataset range.
#
# <details>
# <summary>Question: is every lost detail a training failure?</summary>
#
# No. Some loss is the intended consequence of severe compression. A failure
# would be learning nothing input-specific, failing to beat the mean baseline,
# or producing unstable nonsense. Losing fine detail while preserving dominant
# structure reveals what the limited bottleneck prioritizes.
# </details>

# %% [markdown]
# ## 8. Did it actually beat the lazy poster?
#
# Lesson 01 made the baseline meaningful. We now compare both systems on the
# entire validation set with the same pixel range and MSE convention.

# %%
mean_image, _train_examples = compute_mean_image(
    train_loader, torch.device("cpu")
)
baseline_errors = constant_reconstruction_errors(
    validation_loader, mean_image, torch.device("cpu")
)

model_errors = []
validation_latents = []
validation_labels = []
with torch.no_grad():
    for batch_inputs, batch_labels in validation_loader:
        batch_output = model(batch_inputs.to(device))
        model_errors.append(
            per_example_mse(
                batch_inputs,
                batch_output.reconstruction.cpu(),
            )
        )
        validation_latents.append(batch_output.latent.cpu())
        validation_labels.append(batch_labels.cpu())

model_errors = torch.cat(model_errors)
validation_latents = torch.cat(validation_latents)
validation_labels = torch.cat(validation_labels)

baseline_mse = baseline_errors.mean()
model_mse = model_errors.mean()
print(f"mean-poster validation MSE: {float(baseline_mse):.4f}")
print(f"linear-AE validation MSE:    {float(model_mse):.4f}")
print(
    "error removed beyond baseline: "
    f"{float(1 - model_mse / baseline_mse):.1%}"
)

# %% [markdown]
# The model clears the baseline because it no longer hands everyone the same
# poster. It uses the input to select a location on its learned eight-dimensional
# sheet.
#
# This is stronger evidence than “the loss is small”:
#
# 1. it beats an input-ignoring competitor;
# 2. different inputs receive visibly different reconstructions;
# 3. the preserved and lost features match the expected bottleneck limitation.
#
# <details>
# <summary>Question: does beating the baseline prove that the latent code is useful for every downstream task?</summary>
#
# No. It proves that the code carries information useful for reconstruction.
# Classification, retrieval, anomaly detection, or generation may need different
# information and require their own probes.
# </details>

# %% [markdown]
# ## 9. Open the eight-number tickets
#
# Each row below is the ticket for one garment class; each column is one learned
# slider. The colors show negative and positive values.

# %%
color_limit = float(latents.abs().max())
figure, axis = plt.subplots(figsize=(9, 5))
image = axis.imshow(
    latents.numpy(),
    cmap="coolwarm",
    aspect="auto",
    vmin=-color_limit,
    vmax=color_limit,
)
axis.set(
    title="Different inputs receive different eight-number tickets",
    xlabel="Latent slider",
    ylabel="Input class",
)
axis.set_xticks(range(latent_dimensions))
axis.set_yticks(range(len(names)), labels=names)
figure.colorbar(image, ax=axis, label="Latent value")
figure.tight_layout()

# %% [markdown]
# The rows differ, confirming that the encoder reacts to the input. But avoid
# reading a clean human concept into a single column. Linear autoencoder axes are
# free to rotate: several sliders can cooperate to encode a sleeve, and one
# slider can affect several visual properties.
#
# The decoder reveals what each slider *does*. Starting from its learned base
# canvas, moving one slider upward adds red regions and subtracts blue regions.

# %%
decoder_layer = model.decoder[0]
decoder_bias = decoder_layer.bias.detach().cpu().reshape(
    1, *spec.input_shape
)
decoder_directions = (
    decoder_layer.weight.detach()
    .cpu()
    .T.reshape(latent_dimensions, *spec.input_shape)
)

_ = plot_image_grid(
    decoder_bias,
    title="The decoder's learned starting canvas",
    max_items=1,
)

# %%
direction_limit = float(decoder_directions.abs().max())
figure, axes = plt.subplots(2, 4, figsize=(10, 5))
for index, axis in enumerate(axes.flat):
    axis.imshow(
        decoder_directions[index, 0].numpy(),
        cmap="coolwarm",
        vmin=-direction_limit,
        vmax=direction_limit,
    )
    axis.set_title(f"slider {index}")
    axis.axis("off")
figure.suptitle(
    "Decoder directions: red adds brightness, blue removes it"
)
figure.tight_layout()

# %% [markdown]
# The reconstruction is the starting canvas plus a mixture of these eight
# direction images. This is what “linear” looks like operationally: every
# garment must be assembled from the same eight global paint patterns.
#
# That is efficient and interpretable, but restrictive. Curved or conditional
# structure—such as “this feature matters only when the input is a shoe”—cannot
# be represented naturally by one flat set of directions.

# %% [markdown]
# ## 10. Move one slider while holding the others still
#
# We choose the latent dimension that varies most across validation inputs.
# Then we keep one T-shirt ticket fixed and sweep only that slider from low to
# high.

# %%
latent_stds = validation_latents.std(dim=0)
slider_index = int(latent_stds.argmax())
slider_mean = validation_latents[:, slider_index].mean()
slider_std = latent_stds[slider_index]
slider_values = torch.linspace(
    slider_mean - 2 * slider_std,
    slider_mean + 2 * slider_std,
    steps=9,
)

slider_codes = output.latent[0:1].repeat(9, 1)
slider_codes[:, slider_index] = slider_values.to(device)
with torch.no_grad():
    slider_images = model.decode(slider_codes).cpu()

slider_names = tuple(f"{float(value):.1f}" for value in slider_values)
_ = plot_image_grid(
    slider_images,
    labels=torch.arange(len(slider_names)),
    class_names=slider_names,
    title=f"Only slider {slider_index} changes from left to right",
    max_items=len(slider_names),
)

# %% [markdown]
# The change should be gradual and spread across many pixels. The slider may
# alter several apparent properties at once because it is a learned direction,
# not a named semantic control.
#
# <details>
# <summary>Question: why does one slider affect the entire image instead of one pixel?</summary>
#
# Each decoder slider has a learned weight connecting it to every output pixel.
# Raising that slider adds its whole direction image. The bottleneck compresses
# by controlling broad reusable patterns, not by forwarding eight selected
# pixels.
# </details>

# %% [markdown]
# ## 11. A smooth path is not the same as a valid generator
#
# Take the ticket for a T-shirt and the ticket for an ankle boot. Walking in a
# straight line between them makes the decoder blend their slider settings.

# %%
with torch.no_grad():
    path = linear_interpolation(
        output.latent[0],
        output.latent[-1],
        steps=11,
    )
    interpolated = model.decode(path).cpu()
_ = plot_image_grid(
    interpolated,
    title="A straight path from an encoded T-shirt to an encoded ankle boot",
    max_items=11,
)

# %% [markdown]
# The sequence is smooth because the decoder itself is linear: equal steps in
# ticket space cause equal-sized pixel changes. Smoothness is therefore expected,
# not proof that every midpoint is a realistic garment.
#
# Now try writing 16 random tickets using a convenient standard normal
# distribution.

# %%
torch.manual_seed(0)
with torch.no_grad():
    random_latents = torch.randn(
        16, latent_dimensions, device=device
    )
    random_decoded = model.decode(random_latents).cpu()

print(
    "encoded ticket means:",
    validation_latents.mean(dim=0).round(decimals=2).tolist(),
)
print(
    "encoded ticket stds: ",
    validation_latents.std(dim=0).round(decimals=2).tolist(),
)
print(
    "raw random-decoding range:",
    (float(random_decoded.min()), float(random_decoded.max())),
)
_ = plot_image_grid(
    random_decoded,
    title="Random N(0, I) tickets: nobody taught the AE this ticket distribution",
    max_items=16,
)

# %% [markdown]
# The decoder learned to read tickets produced by its encoder. Nothing trained
# those tickets to follow a standard normal distribution, fill space smoothly,
# or avoid empty regions.
#
# It is like inventing random coat-check numbers and expecting each one to refer
# to a real coat. The decoder can always produce *an image*, but that does not
# mean the image is a plausible sample from the data.
#
# This is the boundary between an ordinary autoencoder and a generative model:
#
# - an **autoencoder** learns to reconstruct after seeing an input;
# - a **generative model** also gives us a principled way to obtain new valid
#   latent codes without first supplying an input.
#
# <details>
# <summary>Question: why not declare N(0, I) to be the latent distribution after training?</summary>
#
# A declaration does not change what the encoder learned. Its eight coordinates
# may have different means, scales, correlations, holes, and class-dependent
# regions. Sampling works only when training or a separately learned model makes
# the sampling distribution match the encoded data.
# </details>

# %% [markdown]
# ## 12. When is a linear autoencoder useful?
#
# | Strength | Limitation |
# |---|---|
# | Fast, simple reconstruction benchmark | Can only build from one flat set of directions |
# | Forces a compact input-dependent summary | Fine and nonlinear details become blurred |
# | Useful for learning bottleneck mechanics | Individual latent axes are not uniquely meaningful |
# | Smooth, easy-to-probe decoder behavior | No learned distribution for random sampling |
# | Closely connected to PCA | Linear outputs may leave the valid pixel range |
#
# Use one when you want:
#
# - a stronger baseline than the mean image;
# - compact features for data that is approximately linear;
# - a transparent sanity check before adding nonlinear capacity;
# - a controlled laboratory for understanding encoders, decoders, and latents.
#
# Do not expect it to model complex image structure, preserve tiny details under
# a severe bottleneck, or generate convincing samples from an arbitrary prior.

# %% [markdown]
# ## Takeaway
#
# A linear autoencoder is a pair of collaborators:
#
# - the encoder turns 784 pixels into an input-specific eight-number ticket;
# - the decoder rebuilds the image from a base canvas and eight learned slider
#   directions.
#
# The bottleneck makes the model preserve broad, recurring structure and discard
# details that do not fit its flat eight-dimensional description. It beats the
# mean poster because it responds to the input, but reconstruction alone does not
# teach it where valid new latent tickets come from.
#
# <details>
# <summary>Advancement gate: can you explain Lesson 02 without leaning on equations?</summary>
#
# A strong explanation includes:
#
# 1. The encoder compresses each image into its own eight-number ticket.
# 2. The decoder uses the ticket as eight slider settings to rebuild the image.
# 3. Training makes the two sides invent a shared reconstruction language.
# 4. The bottleneck keeps broad structure but loses fine detail.
# 5. The model must beat the mean-image baseline and visibly react to inputs.
# 6. Linear decoding creates smooth blends, but the AE never learned a
#    distribution from which we can safely draw new tickets.
#
# If those six ideas feel natural, Lesson 03 can ask what changes when the model
# gains nonlinear layers and different bottleneck sizes.
# </details>
