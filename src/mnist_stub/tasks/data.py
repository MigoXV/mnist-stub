from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import datasets
import numpy as np
import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Sampler

from mnist_stub.common import sha256
from mnist_stub.configs.training import DataConfig
from mnist_stub.models.processing import MnistProcessor

logger = logging.getLogger(__name__)


def preprocess_batch(batch: dict) -> dict:
    return {"pixel_values": MnistProcessor()(batch["image"])["pixel_values"].numpy()}


class EpochSampler(Sampler):
    def __init__(self, size: int, seed: int) -> None:
        self.size, self.seed, self.epoch = size, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        return iter(
            torch.randperm(
                self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)
            ).tolist()
        )

    def __len__(self) -> int:
        return self.size


class MnistDataModule(LightningDataModule):
    def __init__(self, config: DataConfig, seed: int) -> None:
        super().__init__()
        self.config, self.seed = config, seed
        self.splits: dict = {}
        self.metadata: dict = {}
        self.generator = torch.Generator().manual_seed(seed)

    def setup(self, stage: str | None = None) -> None:
        if self.splits:
            return
        cfg = self.config
        if cfg.synthetic:
            rng = np.random.default_rng(2024)
            source = datasets.DatasetDict(
                {
                    name: datasets.Dataset.from_dict(
                        {
                            "image": [
                                Image.fromarray(x)
                                for x in rng.integers(0, 256, (count, 28, 28), dtype=np.uint8)
                            ],
                            "label": np.arange(count) % 10,
                        },
                        features=datasets.Features(
                            {
                                "image": datasets.Image(),
                                "label": datasets.ClassLabel(num_classes=10),
                            }
                        ),
                    )
                    for name, count in (("train", 200), ("test", 40))
                }
            )
            fingerprint = "synthetic-v2"
            validation_size = 40
        else:
            root = Path(cfg.path)
            files = {
                name: sorted((root / "data").glob(f"{name}-*.parquet"))
                for name in ("train", "test")
            }
            for name, paths in files.items():
                if not paths:
                    raise ValueError(f"缺少 {name} split: {root / 'data'}")
            digest = hashlib.sha256()
            for name, paths in files.items():
                digest.update(name.encode())
                for path in paths:
                    digest.update(sha256(path).encode())
            fingerprint = digest.hexdigest()
            cache = cfg.cache_dir or str(root / ".mnist-cache")
            source = datasets.load_dataset(
                "parquet",
                data_files={k: [str(p) for p in v] for k, v in files.items()},
                cache_dir=cache,
            )
            validation_size = cfg.validation_size
        for name, dataset in source.items():
            missing = {"image", "label"} - set(dataset.column_names)
            if missing:
                raise ValueError(f"{name}: 缺少字段 {sorted(missing)}")
            labels = np.asarray(dataset["label"])
            if (
                not len(labels)
                or labels.dtype.kind not in "iu"
                or np.any(labels < 0)
                or np.any(labels > 9)
            ):
                raise ValueError(f"{name}: 标签必须为 0–9 整数且样本非空")
        labels = np.asarray(source["train"]["label"])
        generator = torch.Generator().manual_seed(self.seed)
        train, validation, a, b = [], [], [], []
        if validation_size % 10:
            raise ValueError("data.validation_size 必须为 10 的倍数")
        for label in range(10):
            indices = np.flatnonzero(labels == label).tolist()
            indices = [indices[i] for i in torch.randperm(len(indices), generator=generator)]
            count = validation_size // 10
            if len(indices) < count + 2:
                raise ValueError(f"train: 标签 {label} 样本不足以划分 validation 和 A/B")
            validation.extend(indices[:count])
            remaining = indices[count:]
            train.extend(remaining)
            a.extend(remaining[::2])
            b.extend(remaining[1::2])
        for indices in (train, validation, a, b):
            indices[:] = [indices[i] for i in torch.randperm(len(indices), generator=generator)]
        selected = {
            "train": {"full": train, "a": a, "b": b}[cfg.subset][: cfg.train_limit],
            "validation": validation[: cfg.validation_limit],
            "test": list(range(len(source["test"])))[: cfg.test_limit],
        }
        for name, indices in selected.items():
            if not indices:
                raise ValueError(f"{name}: 样本限制后数据为空")
            subset = source["test" if name == "test" else "train"].select(indices)
            try:
                subset = subset.map(
                    preprocess_batch,
                    batched=True,
                    batch_size=512,
                    remove_columns=[c for c in subset.column_names if c != "label"],
                    desc=f"预处理 {name}",
                )
            except (ValueError, TypeError, OSError) as exc:
                raise ValueError(f"{name}: 图像预处理失败: {exc}") from exc
            self.splits[name] = subset.with_format("torch")
        self.metadata = {
            "fingerprint": fingerprint,
            "indices": selected,
            "preprocessing_version": 1,
        }
        logger.info(
            "data=%s split_sizes=%s fingerprint=%s",
            cfg.path,
            {k: len(v) for k, v in self.splits.items()},
            fingerprint,
        )

    def loader(self, split: str) -> DataLoader:
        cfg = self.config
        return DataLoader(
            self.splits[split],
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            generator=self.generator,
            sampler=EpochSampler(len(self.splits[split]), self.seed) if split == "train" else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self.loader("train")

    def val_dataloader(self) -> DataLoader:
        return self.loader("validation")

    def test_dataloader(self) -> DataLoader:
        return self.loader("test")

    def state_dict(self) -> dict:
        return {"generator": self.generator.get_state(), "data": self.metadata}

    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict["data"] != self.metadata:
            raise ValueError("恢复训练的数据内容或划分与 Checkpoint 不一致")
        self.generator.set_state(state_dict["generator"])
