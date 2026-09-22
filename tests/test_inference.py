from __future__ import annotations

import io
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageOps

from mnist_stub.assets import load_asset, save_asset
from mnist_stub.inference.images import MAX_BYTES, prepare_image
from mnist_stub.inference.runtime import Runtime
from mnist_stub.models.cnn import FEATURES
from mnist_stub.web.app import create_app


def png(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_asset_and_feature_alignment(asset, image_bytes):
    runtime = Runtime(asset)
    ordinary = runtime.predict(image_bytes)
    explained = runtime.predict(image_bytes, features=True)
    assert ordinary["probabilities"] == explained["probabilities"]
    assert abs(sum(ordinary["probabilities"]) - 1) < 1e-6
    assert ordinary["prediction"] == np.argmax(ordinary["probabilities"])
    assert ordinary["features"] == {}
    for name, shape in FEATURES.items():
        feature = explained["features"][name]
        assert feature["shape"] == shape
        assert list(np.asarray(feature["values"]).shape) == shape
    with pytest.raises(FileExistsError):
        save_asset(runtime.model.state_dict(), asset, {})


def test_bad_asset(asset):
    weights = asset / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="SHA-256"):
        load_asset(asset)


def test_unknown_manifest(asset):
    manifest = asset / "manifest.json"
    value = json.loads(manifest.read_text())
    value["format_version"] = 999
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="format_version"):
        load_asset(asset)


def test_images(image_bytes):
    source = Image.open(io.BytesIO(image_bytes))
    light = prepare_image(image_bytes)
    dark = prepare_image(png(ImageOps.invert(source)))
    np.testing.assert_array_equal(light, dark)
    assert light.shape == (28, 28)
    with pytest.raises(ValueError, match="空白"):
        prepare_image(png(Image.new("L", (28, 28), 255)))
    with pytest.raises(ValueError, match="解码"):
        prepare_image(b"broken")
    with pytest.raises(ValueError, match="2 MiB"):
        prepare_image(b"a" * (MAX_BYTES + 1))
    with pytest.raises(ValueError, match="400"):
        prepare_image(png(Image.new("L", (2001, 2000))))


def test_transparent_images():
    for color in ("white", "black"):
        image = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
        ImageDraw.Draw(image).line([(20, 20), (75, 20), (42, 80)], fill=color, width=9)
        result = prepare_image(png(image))
        assert result.max() == 255
        assert result.min() == 0


def test_http_errors_and_output(asset, image_bytes):
    runtime = Runtime(asset)
    with TestClient(create_app(asset, web_dir=None)) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").json() == {"ready": True}
        response = client.post(
            "/api/predict?features=true", content=image_bytes, headers={"content-type": "image/png"}
        )
        assert response.status_code == 200
        assert response.json()["probabilities"] == runtime.predict(image_bytes)["probabilities"]
        assert (
            client.post(
                "/api/predict", content=b"broken", headers={"content-type": "image/png"}
            ).status_code
            == 422
        )
        assert client.post("/api/predict", content=image_bytes).status_code == 415
        assert (
            client.post(
                "/api/predict",
                content=b"a" * (MAX_BYTES + 1),
                headers={"content-type": "image/png"},
            ).status_code
            == 413
        )
        assert client.get("/api/missing", headers={"Accept": "text/html"}).status_code == 404
        assert client.get("/api/metrics").json()["counts"]["success"] == 1


def test_static_routing(asset, tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<html>fixture</html>")
    (web / "asset.js").write_text("export default 1")
    with TestClient(create_app(asset, web_dir=web)) as client:
        assert client.get("/").text == "<html>fixture</html>"
        assert client.get("/nested/route", headers={"Accept": "text/html"}).status_code == 200
        assert client.get("/asset.js").text == "export default 1"
        assert client.get("/openapi.json").json()["info"]["title"] == "MNIST Fixture"
        for accept in ("text/html", "application/json"):
            assert client.get("/api/missing", headers={"Accept": accept}).status_code == 404
    with pytest.raises(RuntimeError, match="前端构建缺失"):
        create_app(asset, web_dir=tmp_path / "missing")


def test_independent_import_and_load(asset, tmp_path):
    code = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('mnist_stub.tasks', 'lightning', 'datasets')):
            raise AssertionError('training imported by inference')
sys.meta_path.insert(0, Guard())
from pathlib import Path
from mnist_stub.inference.runtime import Runtime
runtime = Runtime(Path(sys.argv[1]))
assert runtime.manifest['format_version'] == 2
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(asset)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 不可用")
def test_cpu_gpu_asset_alignment(asset, image_bytes):
    cpu = Runtime(asset, "cpu").predict(image_bytes, True)
    gpu = Runtime(asset, "cuda").predict(image_bytes, True)
    np.testing.assert_allclose(cpu["probabilities"], gpu["probabilities"], rtol=1e-4, atol=1e-5)
    for name in FEATURES:
        np.testing.assert_allclose(
            cpu["features"][name]["values"], gpu["features"][name]["values"], rtol=1e-4, atol=1e-5
        )
