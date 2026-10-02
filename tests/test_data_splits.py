import pytest
import torch
from torch.utils.data import Dataset

from course_aiml_autoencoders.data import build_dataloaders, build_test_dataloader
from course_aiml_autoencoders.data import datasets as module


class IndexedImages(Dataset):
    calls = []

    def __init__(self, *, root, train, download, transform):
        self.train = train
        self.calls.append(train)

    def __len__(self):
        return 100 if self.train else 15

    def __getitem__(self, index):
        return torch.full((1, 28, 28), float(index)), index % 10


def test_training_holdout_is_disjoint_repeatable_and_never_opens_test(monkeypatch):
    IndexedImages.calls = []
    monkeypatch.setattr(module.datasets, "FashionMNIST", IndexedImages)
    config = {"name": "fashion_mnist", "train_size": 60, "validation_size": 20, "split_seed": 7}
    training = {"batch_size": 8, "seed": 0}
    train, validation, _ = build_dataloaders(config, training)
    repeated_train, repeated_validation, _ = build_dataloaders(config, {**training, "seed": 5})
    assert len(train.dataset) == 60
    assert len(validation.dataset) == 20
    assert set(train.dataset.indices).isdisjoint(validation.dataset.indices)
    assert train.dataset.indices == repeated_train.dataset.indices
    assert validation.dataset.indices == repeated_validation.dataset.indices
    assert IndexedImages.calls == [True, True]
    test = build_test_dataloader(config, training)
    assert len(test.dataset) == 15
    assert IndexedImages.calls[-1] is False


def test_default_holdout_and_invalid_subset_sizes(monkeypatch):
    monkeypatch.setattr(module.datasets, "FashionMNIST", IndexedImages)
    config = {"name": "fashion_mnist"}
    train, validation, _ = build_dataloaders(config, {"batch_size": 8})
    assert (len(train.dataset), len(validation.dataset)) == (90, 10)
    for sizes in ({"train_size": 0}, {"validation_size": 100}, {"train_size": 95, "validation_size": 10}):
        with pytest.raises(ValueError, match="training split"):
            build_dataloaders({**config, **sizes}, {"batch_size": 8})
