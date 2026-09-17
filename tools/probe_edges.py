"""探测：为什么边界检测偏向"细节丰富"而不是"长直边缘"。

用法: python tools/probe_edges.py [图集目录]

合成复现：平坦背景 + 一条长直边 + 一块同对比度的细密纹理。
再对每个候选打分函数，统计被选中的单元的**条带集中度**：
把单元内的梯度能量投影到法线方向，看它在法线上有多集中 ——
真正的直边是一条窄带（σ 小），纹理则是弥散的（σ 大）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ascii_art.core import GlyphSet, DEFAULT_CHARS, _EDGE_SUB  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def cell_terms(rgba: Image.Image, cols: int, rows: int) -> dict:
    """逐单元算出候选打分函数需要的各项。"""
    sub = _EDGE_SUB
    small = rgba.convert("L").resize((cols * sub, rows * sub), Image.Resampling.BOX)
    gray = np.asarray(small, np.float32) / 255.0
    gy, gx = np.gradient(gray)
    mag2 = gx * gx + gy * gy

    def reshape(values):
        return values.reshape(rows, sub, cols, sub)

    jxx = reshape(gx * gx).sum(axis=(1, 3))
    jyy = reshape(gy * gy).sum(axis=(1, 3))
    jxy = reshape(gx * gy).sum(axis=(1, 3))
    energy = jxx + jyy
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / np.maximum(energy, 1e-12)

    # 法线方向（= 梯度方向）上的偏移量与权重
    angle = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    nx, ny = np.cos(angle), np.sin(angle)
    u = (np.arange(sub) + 0.5) / sub
    du = (u - 0.5)[None, :, None, None]              # 子采样点的归一化纵向偏移
    dv = (u - 0.5)[None, None, None, :]
    proj = du * ny[:, None, :, None] + dv * nx[:, None, :, None]
    weight = reshape(mag2)
    total = np.maximum(weight.sum(axis=(1, 3)), 1e-12)
    mean = (weight * proj).sum(axis=(1, 3)) / total
    spread = np.sqrt(np.maximum(
        (weight * (proj - mean[:, None, :, None]) ** 2).sum(axis=(1, 3)) / total, 0.0))
    # 均匀分布的 σ ≈ 0.289，用它归一化：1 = 完全弥散，0 = 极窄的一条带
    band = np.clip(spread / 0.2887, 0.0, 2.0)

    # 单元内最强的 1/4 子采样点的梯度幅值均值：反映"边缘有多强"，
    # 又不像单点最大值那样容易被噪声带偏
    mag = np.sqrt(weight)                                   # (rows, sub, cols, sub)
    per_cell = mag.transpose(0, 2, 1, 3).reshape(rows, cols, sub * sub)
    keep = max(1, sub * sub // 4)
    peak = np.sort(per_cell, axis=-1)[..., -keep:].mean(axis=-1)

    # 集中度：峰值能量 / 单元平均能量。窄线条的梯度只落在少数子采样点上，
    # 这个比值高；均匀纹理的能量摊在所有子采样点上，比值接近 1。
    concentration = peak ** 2 / np.maximum(energy / (sub * sub), 1e-12)
    return {"energy": energy, "coherence": coherence, "band": band,
            "peak": peak, "concentration": concentration, "mag": mag}


def _gate(score, terms, band_max):
    picked = score >= np.quantile(score.ravel(), 1.0 - 0.05)
    return picked & (terms["coherence"] >= 0.5) & (terms["band"] <= band_max)


SCORES = {
    "S0 梯度能量总和 x 相干性（旧实现，有偏向）": (
        lambda t: t["energy"] * t["coherence"], lambda t, s: _gate(s, t, 2.0)),
    "S1 peak^2 x coherence": (
        lambda t: t["peak"] ** 2 * t["coherence"], lambda t, s: _gate(s, t, 2.0)),
    "S2 S1 x 集中度": (
        lambda t: t["peak"] ** 2 * t["coherence"] * t["concentration"],
        lambda t, s: _gate(s, t, 2.0)),
    "S3 S1 + 条带集中度 <= 0.3（现实现的门限思路）": (
        lambda t: t["peak"] ** 2 * t["coherence"], lambda t, s: _gate(s, t, 0.30)),
    "S4 S2 + 条带集中度 <= 0.3": (
        lambda t: t["peak"] ** 2 * t["coherence"] * t["concentration"],
        lambda t, s: _gate(s, t, 0.30)),
}


def synthetic() -> Image.Image:
    """平坦背景 + 一条长直边（下半部）+ 一块同对比度的细密纹理（右上）。"""
    rng = np.random.default_rng(0)
    px = np.full((480, 640), 90.0)
    px[300:, :] = 170.0                                # 长直水平边
    # 纹理做到 8 像素一块，才不会被 4x BOX 降采样低通掉
    blocks = rng.choice([-60.0, 60.0], size=(15, 20))
    texture = np.repeat(np.repeat(90.0 + blocks, 8, axis=0), 8, axis=1)
    px[40:160, 440:600] = texture[:120, :160]           # 细密纹理块
    return Image.fromarray(np.clip(px, 0, 255).astype(np.uint8), "L").convert("RGB")


def report_selection(terms: dict, score: np.ndarray, frac: float, label: str) -> None:
    flat = score.ravel()
    threshold = np.quantile(flat, 1.0 - frac)
    picked = score >= threshold
    picked &= terms["coherence"] >= 0.5
    band = terms["band"][picked]
    print(f"    {label:<34} 选中 {picked.mean():>5.1%}  "
          f"条带集中度中位数 {np.median(band) if band.size else float('nan'):.2f}  "
          f"（越小越像窄线条）")


def main(argv: list[str]) -> int:
    frac = 0.05
    print(f"选取前 {frac:.0%} 的单元，比较各打分函数选中的东西是不是「细线条」\n")

    print("[1] 合成图：平坦背景 + 一条长直边（下半部）+ 一块细密纹理（右上，同对比度）")
    img = synthetic()
    terms = cell_terms(img, cols=64, rows=48)
    for name, (fn, pick_fn) in SCORES.items():
        score = fn(terms)
        picked = pick_fn(terms, score)
        # 分别统计落在"长直边所在行"与"纹理块"里的选中比例
        edge_zone = np.zeros_like(picked); edge_zone[28:32, :] = True      # 直边附近
        tex_zone = np.zeros_like(picked); tex_zone[4:16, 44:60] = True     # 纹理块
        print(f"  {name:<30} 选中总数 {picked.sum():>4}  "
              f"其中落在直边上 {picked[edge_zone].sum():>3}  落在纹理上 {picked[tex_zone].sum():>3}")

    root = Path(argv[0]) if argv else Path("test_pic")
    images = sorted(p for p in root.iterdir() if p.suffix.lower() in EXTS)
    if not images:
        return 0
    print(f"\n[2] 真实图集（{len(images)} 张）上被选中单元的条带集中度")
    for name, (fn, pick_fn) in SCORES.items():
        bands, covered = [], []
        for path in images:
            with Image.open(path) as im:
                im.load()
                rgba = im.convert("RGBA")
            g = GlyphSet(DEFAULT_CHARS, None, 20)
            cols = max(1, int(round((10000 * (g.cell_h / g.cell_w)
                                     * rgba.width / rgba.height) ** 0.5)))
            rows = max(1, int(round((rgba.height / (rgba.width / cols)) / (g.cell_h / g.cell_w))))
            t = cell_terms(rgba, cols, rows)
            picked = pick_fn(t, fn(t))
            if picked.any():
                bands.append(float(np.median(t["band"][picked])))
                covered.append(float(picked.mean()))
        print(f"  {name:<30} 条带集中度中位数 {np.mean(bands):.2f}"
              f"   覆盖率 {np.mean(covered):.1%}")
    print("\n  参考：随机单元的条带集中度中位数 "
          f"{np.median(cell_terms(synthetic(), 64, 48)['band']):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
