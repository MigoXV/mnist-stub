from __future__ import annotations

import numpy as np
import pytest
import torch
from datasets import ClassLabel, Dataset, Features, Value
from datasets import Image as DatasetImage
from PIL import Image

from mnist_stub.configs.training import DataConfig
from mnist_stub.models.processing import MnistProcessor
from mnist_stub.tasks.data import MnistDataModule


def parquet_fixture(root, *, missing=False, invalid_label=False, invalid_image=False):
    directory = root / "data"
    directory.mkdir(parents=True)
    for split, count in (("train", 60), ("test", 20)):
        labels = list(np.arange(count).astype(int) % 10)
        if invalid_label:
            labels[0] = 12
        content = {"label": labels}
        features = {"label": Value("int64") if invalid_label else ClassLabel(num_classes=10)}
        if not missing:
            size = (20, 20) if invalid_image else (28, 28)
            content["image"] = [Image.new("L", size, i % 256) for i in range(count)]
            features["image"] = DatasetImage()
        Dataset.from_dict(content, features=Features(features)).to_parquet(
            directory / f"{split}-00000.parquet"
        )


def test_local_parquet_limits_and_cache(tmp_path):
    parquet_fixture(tmp_path)
    cfg = DataConfig(
        path=str(tmp_path),
        validation_size=20,
        train_limit=12,
        validation_limit=5,
        test_limit=7,
        num_workers=0,
    )
    data = MnistDataModule(cfg, 42)
    data.setup()
    assert {k: len(v) for k, v in data.splits.items()} == {"train": 12, "validation": 5, "test": 7}
    again = MnistDataModule(cfg, 42)
    again.setup()
    assert data.metadata == again.metadata
    torch.testing.assert_close(
        data.splits["train"][0]["pixel_values"], again.splits["train"][0]["pixel_values"]
    )
    changed = MnistDataModule(cfg, 43)
    changed.setup()
    with pytest.raises(ValueError, match="数据内容或划分"):
        changed.load_state_dict(data.state_dict())


@pytest.mark.parametrize(
    "option,match",
    [("missing", "缺少字段"), ("invalid_label", "标签"), ("invalid_image", "图像预处理")],
)
def test_invalid_parquet(tmp_path, option, match):
    parquet_fixture(tmp_path, **{option: True})
    data = MnistDataModule(DataConfig(path=str(tmp_path), validation_size=20), 42)
    with pytest.raises(ValueError, match=match):
        data.setup()


def test_missing_split(tmp_path):
    with pytest.raises(ValueError, match="train split"):
        MnistDataModule(DataConfig(path=str(tmp_path)), 42).setup()


def test_processor_normalization():
    pixels = np.zeros((28, 28), dtype=np.uint8)
    result = MnistProcessor()(pixels)["pixel_values"]
    assert result.shape == (1, 1, 28, 28)
    torch.testing.assert_close(result, torch.full_like(result, -0.1307 / 0.3081))
