from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from mnist_stub.assets import save_asset
from mnist_stub.models.cnn import MnistCNN


@pytest.fixture(scope="session", autouse=True)
def small_threads():
    torch.set_num_threads(1)


@pytest.fixture
def asset(tmp_path: Path) -> Path:
    torch.manual_seed(7)
    return save_asset(MnistCNN().state_dict(), tmp_path / "asset", {"test": True})


@pytest.fixture
def image_bytes() -> bytes:
    image = Image.new("L", (96, 96), 0)
    ImageDraw.Draw(image).line([(20, 20), (75, 20), (42, 80)], fill=255, width=9)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()
