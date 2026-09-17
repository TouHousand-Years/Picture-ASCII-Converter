"""在真实插画集上对比字符集的实测工具。

用法: python tools/bench_charsets.py [图集目录] [每图总字符数]

对每张图算出能让「列数 x 行数」接近目标总字符数的列数，然后比较不同字符集在
分级细度、实际用到的级数、单元平均色误差、2x2 子块空间误差、耗时等方面的差别。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ascii_art.core import (  # noqa: E402
    ASCII_CHARS,
    BLOCK_CHARS,
    DEFAULT_CHARS,
    AsciiOptions,
    GlyphSet,
    _downsample,
    convert,
)

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
BOX = Image.Resampling.BOX


def cols_for_cells(width: int, height: int, target_cells: int, aspect: float) -> int:
    """解出使 cols x rows 最接近 target_cells 的列数。

    rows ≈ cols * (height/width) / aspect，于是 cols ≈ sqrt(target * aspect * width / height)。
    """
    return max(1, int(round((target_cells * aspect * width / height) ** 0.5)))


def to_quadrants(array: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """把 (rows*H, cols*W, 3) 的图按每个单元 2x2 重新求平均 -> (rows, cols, 2, 2, 3)。"""
    small = np.asarray(
        Image.fromarray(array, "RGB").resize((cols * 2, rows * 2), BOX), np.float32
    ) / 255.0
    return small.reshape(rows, 2, cols, 2, 3).transpose(0, 2, 1, 3, 4)


def measure(path: Path, chars: str, target_cells: int, font_size: int = 20) -> dict:
    with Image.open(path) as img:
        img.load()
        rgba = img.convert("RGBA")

    glyphs = GlyphSet(chars, None, font_size)
    cols = cols_for_cells(rgba.width, rgba.height, target_cells,
                          glyphs.cell_h / glyphs.cell_w)
    started = time.perf_counter()
    result = convert(rgba, AsciiOptions(cols=cols, font_size=font_size, chars=chars))
    elapsed = time.perf_counter() - started

    rows, cell_h, cell_w = result.rows, result.cell_h, result.cell_w
    out = np.asarray(result.image.convert("RGB"), np.uint8)

    # 色差拆成两栏看：
    #   绝对色差 —— 用户目标函数直接对应的量（渲染区域平均色 vs 源区域平均色）
    #   全局增益 —— 最小二乘拟合出的整体压暗倍数
    #   结构残差 —— 除掉那个常数之后剩下的部分，才是真正的"结构对不上"
    # 只报绝对色差会被整体压暗主导（约八成），从而误判字符集优劣。
    source_cell = np.clip(_downsample(rgba, result.cols, rows), 0, 1)
    rendered_cell = out.reshape(rows, cell_h, result.cols, cell_w, 3).mean(axis=(1, 3)) / 255.0
    abs_err = float(np.abs(rendered_cell - source_cell).mean())
    gain = float((rendered_cell * source_cell).sum()
                 / max((rendered_cell * rendered_cell).sum(), 1e-12))
    struct_err = float(np.abs(gain * rendered_cell - source_cell).mean())

    used = set()
    for line in result.lines:
        used.update(line)

    return {
        "cols": result.cols,
        "rows": rows,
        "cells": result.cols * rows,
        "levels": len(glyphs.chars),
        "used": len(used),
        "max_gap": float(np.diff(glyphs.ramp.coverage).max()),
        "max_ink": glyphs.ramp.max_ink,
        "abs_err": abs_err,
        "gain": gain,
        "struct_err": struct_err,
        "mean_brightness": float(result.brightness.mean()),
        "seconds": elapsed,
    }


def fmt(m: dict) -> str:
    return (f"{m['levels']:>5}{m['used']:>5}{m['cells']:>8}"
            f"{str(m['cols']) + 'x' + str(m['rows']):>12}"
            f"{m['max_gap']:>9.3f}{m['max_ink']:>9.3f}"
            f"{m['abs_err']:>9.4f}{m['gain']:>9.3f}{m['struct_err']:>10.4f}"
            f"{m['mean_brightness']:>9.3f}{m['seconds'] * 1000:>8.0f}ms")


HEADER = (f"{'字符集':<14}{'级数':>5}{'用到':>5}{'总字符':>8}{'列x行':>12}"
          f"{'最大空档':>9}{'墨量上限':>9}{'绝对色差':>9}{'全局增益':>9}{'结构残差':>10}"
          f"{'平均亮度':>9}{'耗时':>10}")

CHARSETS = {
    "默认 16": DEFAULT_CHARS,
    "全 ASCII 95": ASCII_CHARS,
    "块元素 5": BLOCK_CHARS,
}


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else Path("test_pic")
    target = int(argv[1]) if len(argv) > 1 else 10000
    images = sorted(p for p in root.iterdir() if p.suffix.lower() in EXTS)
    if not images:
        print(f"{root} 里没找到图片")
        return 1

    print(f"图集 {root}  共 {len(images)} 张  目标总字符数 ≈ {target}\n")
    totals: dict[str, list[dict]] = {name: [] for name in CHARSETS}

    for path in images:
        with Image.open(path) as img:
            size = img.size
        print(f"### {path.name}  {size[0]}x{size[1]}")
        print(HEADER)
        for name, chars in CHARSETS.items():
            m = measure(path, chars, target)
            print(f"{name:<14}{fmt(m)}")
            totals[name].append(m)
        print()

    keys = list(totals[next(iter(totals))][0])
    print("### 汇总（所有图的平均）")
    print(HEADER)
    for name, rows in totals.items():
        avg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        for key in ("levels", "used", "cells", "cols", "rows", "max_gap"):
            avg[key] = int(round(avg[key]))
        avg["max_gap"] = float(np.mean([r["max_gap"] for r in rows]))
        print(f"{name:<14}{fmt(avg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
