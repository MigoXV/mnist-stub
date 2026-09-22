from __future__ import annotations

import io
import warnings
from typing import Literal

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 2 * 1024 * 1024
MAX_PIXELS = 4_000_000
Polarity = Literal["auto", "dark", "light"]


def prepare_image(payload: bytes, polarity: Polarity = "auto") -> np.ndarray:
    if len(payload) > MAX_BYTES:
        raise ValueError("图片超过 2 MiB")
    if polarity not in {"auto", "dark", "light"}:
        raise ValueError("不支持的笔迹模式")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as source:
                if source.format not in {"PNG", "JPEG"}:
                    raise ValueError("仅支持 PNG 或 JPEG")
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError("图片超过 400 万像素")
                source.load()
                rgba = ImageOps.exif_transpose(source).convert("RGBA")
                background_color = "black" if polarity == "light" else "white"
                if polarity == "auto":
                    alpha = np.asarray(rgba.getchannel("A"))
                    visible = alpha > 127
                    if np.any(alpha < 255) and visible.any():
                        luminance = np.asarray(rgba.convert("L"))[visible]
                        background_color = "black" if np.median(luminance) > 127 else "white"
                background = Image.new("RGBA", rgba.size, background_color)
                image = Image.alpha_composite(background, rgba).convert("L")
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("无法解码图片") from exc
    pixels = np.asarray(image)
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    if polarity == "dark" or (polarity == "auto" and np.median(border) > 127):
        pixels = 255 - pixels
    pixels = np.maximum(
        pixels.astype(np.float32)
        - float(np.median(np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]]))),
        0,
    )
    if float(pixels.max()) < 20:
        raise ValueError("图片为空白或没有可识别前景")
    mask = pixels > max(20, float(pixels.max()) * 0.15)
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        raise ValueError("图片没有足够的有效笔迹")
    crop = Image.fromarray(pixels.clip(0, 255).astype(np.uint8)).crop(
        (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    )
    crop.thumbnail((20, 20), Image.Resampling.LANCZOS)
    # thumbnail never enlarges: resize explicitly so small uploads have the same contract.
    scale = 20 / max(crop.size)
    crop = crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("L", (28, 28))
    canvas.paste(crop, ((28 - crop.width) // 2, (28 - crop.height) // 2))
    return np.asarray(canvas).copy()
