from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetSpec:
    channels: int
    image_size: int
    num_classes: int

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.channels, self.image_size, self.image_size)


SPECS = {
    "fake": DatasetSpec(channels=1, image_size=28, num_classes=10),
    "mnist": DatasetSpec(channels=1, image_size=28, num_classes=10),
    "fashion_mnist": DatasetSpec(channels=1, image_size=28, num_classes=10),
    "cifar10": DatasetSpec(channels=3, image_size=32, num_classes=10),
}

CLASS_NAMES = {
    "fashion_mnist": (
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ),
    "mnist": tuple(str(index) for index in range(10)),
    "cifar10": (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ),
}


def dataset_spec(config: dict[str, Any]) -> DatasetSpec:
    name = str(config["name"]).lower()
    if name not in SPECS:
        raise ValueError(f"Unsupported dataset: {name}")
    base = SPECS[name]
    return DatasetSpec(
        channels=int(config.get("channels", base.channels)),
        image_size=int(config.get("image_size", base.image_size)),
        num_classes=int(config.get("num_classes", base.num_classes)),
    )


def _datasets(config: dict[str, Any], root: Path):
    name = str(config["name"]).lower()
    spec = dataset_spec(config)
    transform = transforms.ToTensor()

    if name == "fake":
        total = int(config.get("train_size", 256))
        validation_size = int(config.get("validation_size", 64))
        full = datasets.FakeData(
            size=total + validation_size,
            image_size=spec.input_shape,
            num_classes=spec.num_classes,
            transform=transform,
            random_offset=0,
        )
        generator_seed = int(config.get("split_seed", 0))
        import torch

        generator = torch.Generator().manual_seed(generator_seed)
        return random_split(full, [total, validation_size], generator=generator)

    dataset_types = {
        "mnist": datasets.MNIST,
        "fashion_mnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
    }
    dataset_type = dataset_types[name]
    train = dataset_type(root=root, train=True, download=True, transform=transform)
    validation = dataset_type(
        root=root, train=False, download=True, transform=transform
    )
    return train, validation


def build_dataloaders(
    dataset_config: dict[str, Any],
    training_config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DatasetSpec]:
    root = Path(dataset_config.get("root", "data"))
    train, validation = _datasets(dataset_config, root)
    batch_size = int(training_config["batch_size"])
    workers = int(training_config.get("num_workers", 0))
    seed = int(training_config.get("seed", 0))

    import torch

    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": bool(training_config.get("pin_memory", False)),
    }
    train_loader = DataLoader(
        train, shuffle=True, generator=generator, drop_last=False, **common
    )
    validation_loader = DataLoader(
        validation, shuffle=False, drop_last=False, **common
    )
    return train_loader, validation_loader, dataset_spec(dataset_config)


def class_names(dataset_name: str) -> tuple[str, ...]:
    """Return human-readable labels when the dataset defines them."""

    name = dataset_name.lower()
    spec = SPECS.get(name)
    if spec is None:
        raise ValueError(f"Unsupported dataset: {name}")
    return CLASS_NAMES.get(
        name, tuple(str(index) for index in range(spec.num_classes))
    )
