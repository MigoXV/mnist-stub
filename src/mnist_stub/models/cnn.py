from __future__ import annotations

import torch
from torch import nn

ARCHITECTURE = "mnist-cnn-v1"
FEATURES = {
    "conv1": [16, 28, 28],
    "pool1": [16, 14, 14],
    "conv2": [32, 14, 14],
    "pool2": [32, 7, 7],
    "embedding": [64],
}
PREPROCESSING = {
    "version": 1,
    "shape": [1, 28, 28],
    "mean": 0.1307,
    "std": 0.3081,
    "image_adapter": "foreground-fit20-center-v1",
}


class MnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)[0]

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        conv1 = torch.relu(self.conv1(x))
        pool1 = torch.nn.functional.max_pool2d(conv1, 2)
        conv2 = torch.relu(self.conv2(pool1))
        pool2 = torch.nn.functional.max_pool2d(conv2, 2)
        embedding = torch.relu(self.fc1(pool2.flatten(1)))
        return self.fc2(embedding), {
            "conv1": conv1,
            "pool1": pool1,
            "conv2": conv2,
            "pool2": pool2,
            "embedding": embedding,
        }

    def freeze_backbone(self, freeze: bool) -> None:
        for module in (self.conv1, self.conv2):
            for parameter in module.parameters():
                parameter.requires_grad_(not freeze)


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device 仅支持 cpu 或 cuda[:index]")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("请求了 CUDA，但当前环境不可用")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA 设备不存在: {name}")
    return device
