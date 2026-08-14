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
# # Lesson 01 — A reconstruction score needs a reference
#
# **Learning objective:** build an intuitive picture of what reconstruction MSE
# notices, why a predictor that ignores its input can still earn a respectable
# score, and how to use the mean-image baseline when judging an autoencoder.
#
# By the end, you should be able to explain:
#
# - what MSE does in ordinary language;
# - why the mean image is the best one-size-fits-all prediction under MSE;
# - why a decent loss does not prove that a model understands its input;
# - what the baseline is useful for, and what it cannot tell us.
#
# **No model is trained in this lesson.** We first construct a deliberately lazy
# competitor. In Lesson 02, an autoencoder will have to beat it.

# %%
import matplotlib.pyplot as plt
import torch

from latent_lab.config import load_yaml
from latent_lab.course import balanced_class_batch, repository_root
from latent_lab.data import build_dataloaders, class_names
from latent_lab.diagnostics import plot_image_grid, plot_reconstruction_grid
from latent_lab.training import (
    compute_mean_image,
    constant_reconstruction_errors,
)

ROOT = repository_root()
config = load_yaml(ROOT / "recipes/ae/ae-001-linear.yaml")
train_loader, validation_loader, spec = build_dataloaders(
    config["dataset"], config["training"]
)
names = class_names(config["dataset"]["name"])
spec

# %% [markdown]
# ## 1. Think of MSE as a pixel accountant
#
# Imagine placing two images on top of each other. A **pixel accountant** visits
# each matching square and records how different its brightness is.
#
# The accountant:
#
# 1. compares pixels at the same location;
# 2. turns every miss into a positive penalty by squaring it;
# 3. averages the penalties into one score.
#
# It does **not** know what a shirt is. It does not know that an object shifted
# one pixel is still the same object. It only checks whether matching squares
# contain matching numbers.
#
# Let us give this accountant a tiny five-by-five symbol. The first candidate is
# exact, the second loses one pixel, the third shifts the symbol, and the fourth
# predicts black everywhere.

# %%
toy_target = torch.zeros(1, 1, 5, 5)
toy_target[0, 0, 1, 1:4] = 1
toy_target[0, 0, 1:4, 2] = 1

perfect = toy_target.clone()
one_pixel_missing = toy_target.clone()
one_pixel_missing[0, 0, 3, 2] = 0
shifted_right = torch.zeros_like(toy_target)
shifted_right[:, :, :, 1:] = toy_target[:, :, :, :-1]
all_black = torch.zeros_like(toy_target)

toy_predictions = torch.cat(
    [perfect, one_pixel_missing, shifted_right, all_black]
)
toy_targets = toy_target.expand(4, -1, -1, -1)
toy_names = (
    "perfect copy",
    "one pixel missing",
    "shifted right",
    "all black",
)
toy_labels = torch.arange(len(toy_names))

_ = plot_reconstruction_grid(
    toy_targets,
    toy_predictions,
    labels=toy_labels,
    class_names=toy_names,
    max_items=4,
    include_error=True,
)

# %%
toy_mse = (
    (toy_targets - toy_predictions)
    .square()
    .flatten(start_dim=1)
    .mean(dim=1)
)
for name, error in zip(toy_names, toy_mse, strict=True):
    print(f"{name:>18}: MSE {float(error):.3f}")

# %% [markdown]
# Read the picture by rows:
#
# - **Input:** the answer we wanted.
# - **Reconstruction:** the candidate answer.
# - **Squared error:** the squares where the pixel accountant charged us.
#
# Missing one of 25 pixels causes one charge. Shifting the symbol changes
# several locations, even though a person still recognizes the same symbol.
# This is MSE's central strength and limitation: it is a precise measure of
# pixel agreement, not a measure of human-perceived meaning.
#
# <details>
# <summary>Optional math: what number is the accountant computing?</summary>
#
# For an image $x$ and reconstruction $\hat{x}$:
#
# $$
# \operatorname{MSE}(x,\hat{x})
# = \frac{1}{CHW}\sum_{c,h,w}(x_{chw}-\hat{x}_{chw})^2
# $$
#
# $C$, $H$, and $W$ are the channel, height, and width counts. The formula says:
# square every matching-pixel difference, then take their mean. For images whose
# values lie in $[0,1]$, MSE also lies in $[0,1]$.
# </details>
#
# <details>
# <summary>Question: why can a small shift receive a surprisingly large penalty?</summary>
#
# The accountant has no concept of “the same object moved slightly.” Pixels that
# used to contain the object become wrong, and pixels at the new location also
# become wrong. One semantic change creates errors along multiple pixel edges.
# </details>

# %% [markdown]
# ## 2. Meet the actual images
#
# Before trusting any score, look at what is being scored. Fashion-MNIST contains
# centered grayscale garments on a mostly black 28-by-28 canvas.
#
# We deliberately select one validation example from every class. Otherwise, one
# convenient batch could give us a misleading picture of the dataset.

# %%
class_images, class_labels = balanced_class_batch(
    validation_loader, spec.num_classes
)
print(
    "shape:",
    tuple(class_images.shape),
    "range:",
    (float(class_images.min()), float(class_images.max())),
)
_ = plot_image_grid(
    class_images,
    labels=class_labels,
    class_names=names,
    title="One validation example from each Fashion-MNIST class",
)

# %% [markdown]
# The shape `[10, 1, 28, 28]` means:
#
# - `10`: one selected image for each class;
# - `1`: one grayscale channel;
# - `28, 28`: image height and width.
#
# Notice the large black regions. Those pixels are easy points for a predictor:
# outputting zero is often correct even if the predictor has no idea which
# garment is present.
#
# <details>
# <summary>Question: why is “predict black” less absurd here than for a photograph that fills the frame?</summary>
#
# Every Fashion-MNIST object sits on a black canvas, so many target pixels are
# already zero or almost zero. A black prediction gets all those locations right
# for free. A photograph usually contains information across most of the frame,
# so black would be wrong in many more places.
# </details>

# %% [markdown]
# ## 3. The lazy artist challenge
#
# Suppose an artist must reconstruct every customer’s garment—but refuses to
# look at the garment first. The artist is allowed to paint **one reusable
# poster** and hand that same poster to everyone.
#
# This sounds useless, and it is useless for recognizing individual inputs. But
# it gives us a vital reference:
#
# > How well can someone score without using the input at all?
#
# If an autoencoder cannot beat this lazy artist, its impressive-looking loss
# number tells us very little.
#
# **Predict before revealing the poster:**
#
# - What will remain after averaging many shirts, shoes, bags, and trousers?
# - Will a particular class remain recognizable?
# - Why might the result still achieve a nonterrible MSE?
#
# <details>
# <summary>Reveal the expected intuition</summary>
#
# Shared structure should survive: a dark background and a pale, centered
# garment-like region. Incompatible details should blur together, so no one
# class should be cleanly recognizable. The poster can still score reasonably
# because it gets common background and common object locations approximately
# right.
# </details>

# %% [markdown]
# ## 4. Watch an average turn into a ghost
#
# Averaging images resembles taking a long-exposure group photograph. Features
# that repeatedly occupy the same place remain visible. Features that disagree
# from image to image fade into a ghost.
#
# We will average 1, 10, 100, 1,000, and finally all 60,000 training images.

# %%
preview_batches = []
preview_count = 0
for inputs, _labels in train_loader:
    preview_batches.append(inputs)
    preview_count += inputs.shape[0]
    if preview_count >= 1_000:
        break

preview_images = torch.cat(preview_batches)[:1_000]
snapshot_counts = (1, 10, 100, 1_000)
average_snapshots = [
    preview_images[:count].mean(dim=0, keepdim=True)
    for count in snapshot_counts
]

device = torch.device("cpu")
mean_image, train_examples = compute_mean_image(train_loader, device)
average_snapshots.append(mean_image.cpu())
average_snapshots = torch.cat(average_snapshots)

snapshot_names = (
    "1 image",
    "10 images",
    "100 images",
    "1,000 images",
    f"{train_examples:,} images",
)
_ = plot_image_grid(
    average_snapshots,
    labels=torch.arange(len(snapshot_names)),
    class_names=snapshot_names,
    title="Individual details fade; shared locations survive",
    max_items=len(snapshot_names),
)

# %% [markdown]
# At one image, we see an individual garment. As the crowd grows:
#
# - the black background stays black because most images agree there;
# - the center remains bright because garments usually occupy it;
# - sleeves, shoes, bags, and trouser legs disagree and blur together.
#
# The final result is the **mean training image**: a statistical summary, not a
# remembered example and not a newly imagined garment.
#
# We compute it from training data only, then judge it on validation data. Using
# validation images to design the poster would let the test influence the answer.
#
# <details>
# <summary>Optional math: why is the mean the best reusable poster under MSE?</summary>
#
# Focus on one pixel location. If its training values are
# $x_1,\ldots,x_N$, a constant prediction $a$ has loss
#
# $$
# L(a)=\frac{1}{N}\sum_i(x_i-a)^2.
# $$
#
# If $a$ is below the mean, increasing it reduces more squared error than it
# adds. If it is above the mean, decreasing it helps. At the mean, the upward
# and downward pulls balance. The derivative confirms the same result:
#
# $$
# \frac{dL}{da}=\frac{2}{N}\sum_i(a-x_i)=0
# \quad\Longrightarrow\quad
# a=\frac{1}{N}\sum_i x_i.
# $$
#
# Applying that choice independently at every pixel creates the mean image.
# </details>

# %% [markdown]
# ## 5. Hand the same poster to every customer
#
# Now comes the crucial probe. Each input below is different, but every
# “reconstruction” is exactly the same mean image.

# %%
constant_predictions = mean_image.cpu().expand_as(class_images)
_ = plot_reconstruction_grid(
    class_images,
    constant_predictions,
    labels=class_labels,
    class_names=names,
    max_items=spec.num_classes,
    include_error=True,
)

# %%
largest_prediction_difference = (
    constant_predictions - constant_predictions[0:1]
).abs().max()
print(
    "largest difference between any two predictions:",
    float(largest_prediction_difference),
)

# %% [markdown]
# The input row contains ten classes. The reconstruction row contains one ghost
# repeated ten times. The printed difference is zero.
#
# This is the fastest conceptual test for input dependence:
#
# > If I swap the input, can the output change?
#
# Here the answer is no. Therefore this predictor has learned no representation
# of the particular garment, regardless of its eventual score.
#
# <details>
# <summary>Question: could you identify the input class from these reconstructions?</summary>
#
# No. Every input maps to the same output. The poster contains dataset-level
# regularities—dark borders and a bright center—but no information about which
# individual image was supplied.
# </details>

# %% [markdown]
# ## 6. Why the lazy artist can earn a respectable score
#
# Let us compare two input-ignoring predictors on every validation image:
#
# - **all black:** exploits the common background;
# - **mean image:** also exploits where garments commonly appear.

# %%
black_image = torch.zeros_like(mean_image)
black_errors = constant_reconstruction_errors(
    validation_loader, black_image, device
)
mean_errors = constant_reconstruction_errors(
    validation_loader, mean_image, device
)

dark_pixels = 0
total_pixels = 0
for inputs, _labels in validation_loader:
    dark_pixels += int((inputs <= 0.05).sum())
    total_pixels += inputs.numel()

print(f"nearly black validation pixels: {dark_pixels / total_pixels:.1%}")
print(f"all-black validation MSE:       {float(black_errors.mean()):.4f}")
print(f"mean-image validation MSE:      {float(mean_errors.mean()):.4f}")
print(
    "mean-image improvement:        "
    f"{1 - float(mean_errors.mean() / black_errors.mean()):.1%}"
)

# %% [markdown]
# The black predictor earns many free successes because the canvas dominates the
# image. The mean poster improves further by placing brightness where garments
# usually occur.
#
# Neither predictor recognizes the input. They exploit the dataset's **common
# layout**. That is exactly why a loss number needs a baseline: the dataset may
# make part of the task easy before a model learns anything interesting.

# %%
comparison_target = class_images[0:1].expand(2, -1, -1, -1)
comparison_predictions = torch.cat([black_image.cpu(), mean_image.cpu()])
comparison_names = ("all black", "mean image")
_ = plot_reconstruction_grid(
    comparison_target,
    comparison_predictions,
    labels=torch.arange(len(comparison_names)),
    class_names=comparison_names,
    max_items=2,
    include_error=True,
)

# %% [markdown]
# Look at the error row. The mean image reduces error across common garment
# regions, but it cannot place the particular edges and details of this input.
#
# <details>
# <summary>Question: if the mean image scores better, has it learned a better representation?</summary>
#
# No. It is a better **constant guess**, not an input-dependent representation.
# It summarizes the training dataset more effectively than black, but still
# throws away all information about the current input.
# </details>

# %% [markdown]
# ## 7. One average score can hide many experiences
#
# A class average at school can hide students who found the exam easy and others
# who found it hard. A dataset-wide MSE does the same.
#
# We should inspect both the distribution of per-image errors and the average for
# each garment class.

# %%
validation_labels = torch.cat(
    [labels.cpu() for _inputs, labels in validation_loader]
)
class_error_means = torch.stack(
    [mean_errors[validation_labels == label].mean() for label in range(10)]
)

figure, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(mean_errors.numpy(), bins=30, color="slateblue", alpha=0.85)
axes[0].axvline(
    float(mean_errors.mean()),
    color="black",
    linestyle="--",
    label="overall mean",
)
axes[0].set(
    title="Some images are much harder than others",
    xlabel="Per-image MSE",
    ylabel="Number of validation images",
)
axes[0].legend()

axes[1].bar(names, class_error_means.numpy(), color="darkorange")
axes[1].set(
    title="The same poster fits some classes better",
    ylabel="Mean-image MSE",
)
axes[1].tick_params(axis="x", rotation=70)
figure.tight_layout()

# %% [markdown]
# The mean image is closer to classes whose shapes overlap the dataset's central
# ghost, and farther from classes with distinctive geometry. The overall mean
# hides this variation.
#
# This habit will matter throughout the course:
#
# > Treat a scalar metric as a summary of evidence, not as the evidence itself.
#
# Pair it with aligned images, error maps, distributions, and class-level views.

# %% [markdown]
# ## 8. What this baseline is—and is not—for
#
# Think of the mean baseline as the bar in a high-jump competition. Clearing it
# matters, but clearing it does not prove Olympic ability.
#
# | Tool | Useful because... | Misleading if... |
# |---|---|---|
# | **MSE** | it is simple, stable, and precisely measures matching-pixel fidelity | we treat it as a measure of semantic understanding or visual quality |
# | **Mean-image baseline** | it is cheap, reproducible, and exposes how much score comes from dataset regularity | we mistake a good constant guess for an input-dependent model |
#
# In practice, use the baseline to:
#
# 1. compute a reference from the **training** split;
# 2. evaluate it on validation data with the same preprocessing and MSE reduction
#    as the model;
# 3. require the model to beat it;
# 4. also verify visually that different inputs produce appropriately different
#    reconstructions.
#
# The baseline cannot recognize classes, preserve individual details, learn a
# useful latent code, or generate diverse samples.
#
# <details>
# <summary>Question: an autoencoder reports validation MSE 0.05. Is that good?</summary>
#
# The number alone is incomplete. Compare it with the all-black and mean-image
# scores under exactly the same data range and averaging convention. Then inspect
# whether reconstructions preserve input-specific shapes and whether outputs
# change when inputs change. Beating the baseline is necessary evidence, but not
# sufficient evidence of a useful representation.
# </details>

# %% [markdown]
# ## 9. The bridge to an autoencoder
#
# The lazy baseline follows this rule:
#
# ```
# any input  ───────────────> the same mean poster
# ```
#
# The autoencoder in Lesson 02 will follow:
#
# ```
# input  ──> encoder ──> compact code ──> decoder ──> tailored reconstruction
# ```
#
# A successful autoencoder should:
#
# - beat the mean-image validation MSE;
# - change its reconstruction when the input changes;
# - preserve recognizable, input-specific structure;
# - fail in ways that make sense for its limited bottleneck.
#
# That comparison turns “the loss went down” into a meaningful claim.

# %% [markdown]
# ## Takeaway
#
# MSE is a **pixel-agreement score**. It is useful, but it is not an understanding
# meter.
#
# Fashion-MNIST's shared black background and centered layout let a lazy,
# input-ignoring predictor score surprisingly well. Under MSE, the best such
# predictor is the mean training image: one blurred poster handed to every input.
#
# Therefore:
#
# > A reconstruction loss becomes meaningful only relative to what could be
# > achieved without learning the input.
#
# <details>
# <summary>Advancement gate: can you explain Lesson 01 without using a formula?</summary>
#
# A strong explanation includes these ideas:
#
# 1. MSE acts like a pixel accountant: it checks matching locations, not meaning.
# 2. Fashion-MNIST contains many easy black-background pixels.
# 3. The mean image is the best one-size-fits-all poster under squared error.
# 4. Its nonterrible score comes from common layout, not input understanding.
# 5. A real autoencoder must beat that score **and** produce reconstructions that
#    visibly depend on the input.
#
# If those five statements feel natural, proceed to Lesson 02.
# </details>
