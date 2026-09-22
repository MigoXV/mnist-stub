from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from mnist_stub.common import sha256, write_json
from mnist_stub.models.cnn import ARCHITECTURE, FEATURES, PREPROCESSING, MnistCNN


def load_asset(path: Path) -> tuple[MnistCNN, dict[str, Any]]:
    manifest = json.loads((path / "manifest.json").read_text())
    expected = {
        "format_version": 1,
        "architecture": ARCHITECTURE,
        "labels": list(range(10)),
        "preprocessing": PREPROCESSING,
        "features": FEATURES,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"模型资产字段不兼容: {key}")
    if manifest.get("weights_sha256") != sha256(path / "weights.pt"):
        raise ValueError("模型权重 SHA-256 校验失败")
    model = MnistCNN()
    state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    validate_weights(state, model)
    model.load_state_dict(state, strict=True)
    return model.eval(), manifest


def validate_weights(state: dict, model: MnistCNN) -> None:
    expected = model.state_dict()
    if state.keys() != expected.keys():
        raise ValueError("模型权重 key 不匹配")
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[key].shape:
            raise ValueError(f"模型权重 shape 不匹配: {key}")
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError(f"模型权重 dtype 或数值非法: {key}")


def save_asset(state: dict, destination: Path, metadata: dict[str, Any]) -> Path:
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖模型资产: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".export-", dir=destination.parent))
    try:
        model = MnistCNN().eval()
        cpu_state = {key: tensor.detach().cpu().clone() for key, tensor in state.items()}
        validate_weights(cpu_state, model)
        model.load_state_dict(cpu_state)
        torch.save(cpu_state, temporary / "weights.pt")
        digest = sha256(temporary / "weights.pt")
        manifest = {
            "format_version": 1,
            "architecture": ARCHITECTURE,
            "model_id": digest,
            "weights_sha256": digest,
            "labels": list(range(10)),
            "preprocessing": PREPROCESSING,
            "features": FEATURES,
            "metadata": metadata,
        }
        write_json(temporary / "manifest.json", manifest)
        loaded, _ = load_asset(temporary)
        sample = torch.linspace(-1, 1, 784).reshape(1, 1, 28, 28)
        with torch.inference_mode():
            torch.testing.assert_close(model(sample), loaded(sample), rtol=0, atol=0)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination
