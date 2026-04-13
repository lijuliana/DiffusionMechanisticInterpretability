from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image


def pil_images_to_grid(
    images: Sequence[Image.Image] | Image.Image,
    *,
    grid_size: tuple[int, int] | None = None,
    padding: int = 2,
    bg: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Tile PIL images into a single RGB image."""
    if isinstance(images, Image.Image):
        return images.convert("RGB")

    imgs = [im.convert("RGB") for im in images]
    if not imgs:
        raise ValueError("No images to grid.")

    if grid_size is None:
        n = len(imgs)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_size

    w, h = imgs[0].size
    out_w = cols * w + padding * (cols + 1)
    out_h = rows * h + padding * (rows + 1)
    canvas = Image.new("RGB", (out_w, out_h), bg)

    for idx, im in enumerate(imgs):
        if idx >= rows * cols:
            break
        r, c = divmod(idx, cols)
        canvas.paste(im, (padding + c * (w + padding), padding + r * (h + padding)))
    return canvas
