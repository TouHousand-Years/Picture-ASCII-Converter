"""探测：怎么才能保住高亮部分的颜色。

用法: python tools/probe_highlight.py [图集目录] [每图总字符数]

背景：填充色规则 `K = clip(C/ink, 0, 1)` 的实际效果是
`渲染单元平均色 = min(C, ink)`（逐通道封顶）。高亮区域里高于墨量的通道会被压成
同一个值 —— 色度被抹平。比如 C=(0.95,0.6,0.2)、墨量 0.228 时渲染成
(0.228,0.228,0.20)，几乎全灰。

本脚本比较几种"封顶"方式（都按解析式算，不需要真渲染）：
  V0 逐通道截断（现状）
  V1 逐通道截断 + 颜色上限 g，归一化最大亮度同步改为 max_ink x g
  V2 等比缩放（保色度，上限 1）
  V3 等比缩放 + 颜色上限 g，归一化最大亮度同步改为 max_ink x g
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ascii_art.core import (  # noqa: E402
    ASCII_CHARS,
    BLOCK_CHARS,
    DEFAULT_CHARS,
    GlyphSet,
    _brightness,
    _downsample,
)

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def solve(ideal: np.ndarray, g: float, proportional: bool) -> np.ndarray:
    """理想填充色 ideal = C/ink -> 受限后的填充色。"""
    if proportional:
        peak = np.maximum(ideal.max(axis=-1, keepdims=True), 1e-9)
        return np.clip(ideal * np.minimum(g / peak, 1.0), 0.0, 1.0)
    return np.clip(ideal, 0.0, g)


def run(image: Path, chars: str, g: float, proportional: bool,
        target: int = 10000, font_size: int = 20) -> dict:
    with Image.open(image) as img:
        img.load()
        rgba = img.convert("RGBA")
    glyphs = GlyphSet(chars, None, font_size)
    max_ink = glyphs.ramp.max_ink
    cols = max(1, int(round((target * (glyphs.cell_h / glyphs.cell_w)
                             * rgba.width / rgba.height) ** 0.5)))
    aspect = glyphs.cell_h / glyphs.cell_w
    rows = max(1, int(round((rgba.height / (rgba.width / cols)) / aspect)))

    C = np.clip(_downsample(rgba, cols, rows), 0, 1)
    b = np.clip(_brightness(C, "luminance"), 0, 1)

    # 归一化的"最大亮度"——按设定同步改成 max_ink x g
    lookup = glyphs.ramp.ink / (max_ink * g)
    idx = np.clip(np.searchsorted(lookup, b, side="left"), 0, len(chars) - 1)
    ink = glyphs.ramp.ink[idx]

    ideal = C / np.maximum(ink[..., None], 1e-9)
    K = solve(ideal, g, proportional)
    rendered = ink[..., None] * K                      # 黑底：ink x K

    # 高亮单元 = 源亮度前 25%
    hot = b >= np.quantile(b, 0.75)
    chroma = lambda a: float((a.max(axis=-1) - a.min(axis=-1))[hot].mean())
    src_chroma = chroma(C)
    out_chroma = chroma(rendered)
    lum = lambda a: float(a.mean(axis=-1).mean())
    gain = float((rendered * C).sum() / max((rendered * rendered).sum(), 1e-12))
    return {
        "abs": float(np.abs(rendered - C).mean()),
        "gain": gain,
        "struct": float(np.abs(gain * rendered - C).mean()),
        "hot_chroma": out_chroma / max(src_chroma, 1e-9),
        "hot_err": float(np.abs(rendered[hot] - C[hot]).mean()),
        "overall_lum": lum(rendered),
    }


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else Path("test_pic")
    target = int(argv[1]) if len(argv) > 1 else 10000
    images = sorted(p for p in root.iterdir() if p.suffix.lower() in EXTS)

    print(f"图集 {root}（{len(images)} 张），每图约 {target} 字符，字符集 = 默认 16\n")
    print(f"{'方案':<30}{'绝对色差':>9}{'全局增益':>9}{'结构残差':>9}"
          f"{'高亮保色':>9}{'高亮色差':>9}{'输出亮度':>9}")
    for label, g, prop in (
        ("V0 逐通道截断（现状）", 1.0, False),
        ("V1 逐通道截断 g=0.8", 0.8, False),
        ("V1 逐通道截断 g=0.6", 0.6, False),
        ("V1 逐通道截断 g=0.4", 0.4, False),
        ("V2 等比缩放（上限1）", 1.0, True),
        ("V3 等比缩放 g=0.8", 0.8, True),
        ("V3 等比缩放 g=0.6", 0.6, True),
    ):
        acc = [run(p, DEFAULT_CHARS, g, prop, target) for p in images]
        avg = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
        print(f"{label:<30}{avg['abs']:>9.4f}{avg['gain']:>9.3f}{avg['struct']:>9.4f}"
              f"{avg['hot_chroma']:>8.0%}{avg['hot_err']:>9.4f}{avg['overall_lum']:>9.3f}")

    print(f"\n参考：块元素字符集（墨量 0.95）")
    for chars, name in ((BLOCK_CHARS, "块元素 V0"), (ASCII_CHARS, "全ASCII V0")):
        acc = [run(p, chars, 1.0, False, target) for p in images]
        avg = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
        print(f"{name:<30}{avg['abs']:>9.4f}{avg['gain']:>9.3f}{avg['struct']:>9.4f}"
              f"{avg['hot_chroma']:>8.0%}{avg['hot_err']:>9.4f}{avg['overall_lum']:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
