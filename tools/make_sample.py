"""生成一张用于试玩的示例图片（色轮 + 灰阶 + 几个图形）。

    python tools/make_sample.py samples/sample.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W, H = 960, 640


def build() -> Image.Image:
    yy, xx = np.indices((H, W))
    u = xx / (W - 1)
    v = yy / (H - 1)

    # 上半：横向色相渐变 + 纵向明度渐变
    hue = u
    sat = np.where(v < 0.5, 1.0 - v * 1.2, 1.0)
    val = np.where(v < 0.5, 1.0, 1.0 - (v - 0.5) * 1.8)
    hue = np.where(v < 0.5, hue, (u * 3.0) % 1.0)
    sat = np.clip(sat, 0.0, 1.0)
    val = np.clip(val, 0.0, 1.0)

    h6 = hue * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p, q, t = val * (1 - sat), val * (1 - f * sat), val * (1 - (1 - f) * sat)
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4], [val, q, p, p, t], val)
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4], [t, val, val, q, p], p)
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4], [p, p, t, val, val], q)
    arr = np.stack([r, g, b], axis=-1)

    # 底部一条灰阶，用来检验字符密度分档
    ramp = np.linspace(0.0, 1.0, W)
    arr[int(H * 0.82):, :, :] = ramp[None, :, None]

    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse((W * 0.06, H * 0.10, W * 0.30, H * 0.46),
                 outline=(255, 255, 255), width=5)
    draw.rectangle((W * 0.68, H * 0.12, W * 0.94, H * 0.44),
                   outline=(0, 0, 0), width=5)
    draw.polygon([(W * 0.44, H * 0.46), (W * 0.58, H * 0.46), (W * 0.51, H * 0.16)],
                 fill=(255, 255, 255))
    draw.text((W * 0.05, H * 0.52), "ASCII COLOR", fill=(255, 255, 255))
    return img


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("samples/sample.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    build().save(out)
    print(f"已生成 {out}  ({W}x{H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
