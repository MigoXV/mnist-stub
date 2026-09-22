from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from mnist_stub.configs.training import TrainConfig
from mnist_stub.models.cnn import PREPROCESSING


def prepare_data(root: Path) -> None:
    from torchvision.datasets import MNIST

    for train in (True, False):
        MNIST(str(root), train=train, download=True)


def datasets(config: TrainConfig) -> tuple[dict[str, TensorDataset], dict]:
    if config.synthetic:
        generator = torch.Generator().manual_seed(2024)
        train_x = torch.randint(0, 256, (200, 28, 28), dtype=torch.uint8, generator=generator)
        train_y = torch.arange(200) % 10
        test_x = torch.randint(0, 256, (40, 28, 28), dtype=torch.uint8, generator=generator)
        test_y = torch.arange(40) % 10
        validation_size = 40
    else:
        from torchvision.datasets import MNIST

        training = MNIST(config.data_dir, train=True, download=False)
        testing = MNIST(config.data_dir, train=False, download=False)
        train_x, train_y, test_x, test_y = (
            training.data,
            training.targets,
            testing.data,
            testing.targets,
        )
        validation_size = 5000
    for name, images, labels in (("train", train_x, train_y), ("test", test_x, test_y)):
        if images.ndim != 3 or tuple(images.shape[1:]) != (28, 28):
            raise ValueError(f"{name}: 非法图像 shape")
        if len(images) != len(labels) or not len(labels) or labels.min() < 0 or labels.max() > 9:
            raise ValueError(f"{name}: 样本数或标签非法")
    generator = torch.Generator().manual_seed(config.seed)
    # Stratification keeps every class represented in the pretraining/fine-tuning subsets.
    train_indices, val_indices, a_indices, b_indices = [], [], [], []
    for label in range(10):
        indices = (train_y == label).nonzero().flatten()
        indices = indices[torch.randperm(len(indices), generator=generator)]
        val = indices[: validation_size // 10].tolist()
        remaining = indices[validation_size // 10 :].tolist()
        val_indices.extend(val)
        train_indices.extend(remaining)
        a_indices.extend(remaining[::2])
        b_indices.extend(remaining[1::2])
    for indices in (train_indices, val_indices, a_indices, b_indices):
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices[:] = [indices[i] for i in order]
    selected = {"full": train_indices, "a": a_indices, "b": b_indices}[config.subset]
    splits = {
        "train": selected[: config.train_limit],
        "validation": val_indices[: config.validation_limit],
        "test": list(range(len(test_y)))[: config.test_limit],
    }
    result = {}
    for name, indices in splits.items():
        if not indices:
            raise ValueError(f"{name}: 数据划分为空")
        source_x, source_y = (test_x, test_y) if name == "test" else (train_x, train_y)
        x = source_x[indices].float().unsqueeze(1).div(255)
        x = (x - PREPROCESSING["mean"]) / PREPROCESSING["std"]
        result[name] = TensorDataset(x, source_y[indices].long())
    digest = hashlib.sha256()
    for tensor in (train_x, train_y, test_x, test_y):
        digest.update(tensor.numpy().tobytes())
    return result, {
        "dataset": "synthetic" if config.synthetic else "MNIST",
        "fingerprint": digest.hexdigest(),
        "indices": splits,
    }
