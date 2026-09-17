"""探测：用字符的不对称结构做"次像素"渲染，到底有多少空间、代价是什么。

用法: python tools/probe_subpixel.py [图片]

思路：把每个字符单元切成 2x2 子块，比较三种渲染方式在**子块分辨率**上的误差：

  A 现状       —— 按亮度选字，填充色只解一个（整格同色）
  B 形状感知   —— 仍是一字一色，但在亮度允许的字形里挑"子块结构最贴合原图"的那个
  C 子块着色   —— 上限：同一个字形，但允许 2x2 四个子块各自解一个颜色

同时报告字符集在 2x2 粒度上的"不对称度"，以及单元内部 vs 单元之间的信息量之比。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ascii_art.core import (  # noqa: E402
    DEFAULT_CHARS,
    GlyphSet,
    _brightness,
    _downsample,
    _pick_candidates,
)

#: 定向（不对称）字符候选：墨落点有明确方位，才可能承载"次像素"信息
DIRECTIONAL = "'`,.-_\u00b4^\\/v<>\"!;[](){}LJT"

BINS = 2          # 子块切分：2x2


def sub_coverages(glyphs: GlyphSet) -> np.ndarray:
    """每个字形在 2x2 子块里的覆盖率 -> (n_chars, 2, 2)。"""
    out = []
    for mask in glyphs.atlas:
        small = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (BINS, BINS), Image.Resampling.BOX
        )
        out.append(np.asarray(small, np.float64) / 255.0)
    return np.stack(out)


def fit_glyph(coverage: np.ndarray, target: np.ndarray, bg: np.ndarray):
    """给定字形，解出让它 2x2 子块平均色最接近 target 的**单色** K，并返回误差。

    coverage 形状 (..., 2, 2)，target 形状 (..., 2, 2, 3)。
    逐通道最小二乘：K = Σ c_i (S_i - (1-c_i) bg) / Σ c_i²。
    """
    c = coverage[..., None]                                  # (..., 2, 2, 1)
    num = (c * (target - (1.0 - c) * bg)).sum(axis=(-3, -2))   # (..., 3)
    den = (c * c).sum(axis=(-3, -2))                           # (..., 1)
    k = np.clip(np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0), 0.0, 1.0)
    rendered = c * k[..., None, None, :] + (1.0 - c) * bg
    return k, np.abs(rendered - target).mean(axis=(-3, -2, -1))


def spatial_error(coverage: np.ndarray, color, target: np.ndarray, bg: np.ndarray):
    """渲染出来的子块平均色与目标的平均绝对差 -> (...)"""
    rendered = coverage[..., None] * color + (1.0 - coverage[..., None]) * bg
    return np.abs(rendered - target).mean(axis=(-3, -2, -1))


def per_subblock_error(coverage, target, bg):
    """上限：每个子块各解一个颜色，能贴多近。"""
    c = coverage[..., None]
    k = np.where(c > 1e-6, (target - (1.0 - c) * bg) / np.maximum(c, 1e-6), 0.0)
    k = np.clip(k, 0.0, 1.0)
    rendered = c * k + (1.0 - c) * bg
    return np.abs(rendered - target).mean(axis=(-2, -1))


def report(image_path: str, cols: int = 120, font_size: int = 20) -> None:
    src = Image.open(image_path).convert("RGBA")
    bg = np.zeros(3)
    glyphs = GlyphSet(DEFAULT_CHARS, None, font_size)
    ramp = glyphs.ramp
    cover = sub_coverages(glyphs)                 # (n, 2, 2)

    # --- 网格 ---------------------------------------------------------- #
    aspect = glyphs.cell_h / glyphs.cell_w
    rows = max(1, round((src.height / (src.width / cols)) / aspect))
    cell_mean = np.clip(_downsample(src, cols, rows), 0, 1)               # (rows,cols,3)
    sub = np.clip(_downsample(src, cols * BINS, rows * BINS), 0, 1)
    sub = sub.reshape(rows, BINS, cols, BINS, 3).transpose(0, 2, 1, 3, 4)

    print(f"源图 {src.size}  网格 {cols}x{rows}  单元 = {src.width/cols:.1f}x{src.height/rows:.1f} 源像素")
    print(f"字符集 {len(glyphs.chars)} 个：{''.join(glyphs.chars)!r}\n")

    # --- 1. 单元内部 vs 单元之间 --------------------------------------- #
    inter = float(cell_mean.std(axis=(0, 1)).mean())
    intra = float(sub.std(axis=(2, 3)).mean())
    print(f"[1] 信息量      单元之间 std {inter:.4f}   单元内部(2x2) std {intra:.4f}"
          f"   内部/之间 = {intra/inter:.1%}")

    # --- 2. 字符自身的不对称度 ----------------------------------------- #
    flat = cover.reshape(len(glyphs.chars), -1)
    asym = flat.max(axis=1) - flat.min(axis=1)
    order = np.argsort(-asym)
    print(f"[2] 字形不对称度（2x2 子块覆盖率极差，越大越「有方位」）")
    print("    现有字符集:" + "  ".join(
        f"{glyphs.chars[i]!r}:{asym[i]:.2f}" for i in order[:6]) + f"   …最大 {asym.max():.2f}")
    uni = float(np.abs(sub - sub.mean(axis=(2, 3), keepdims=True)).mean())
    print(f"    现字符集的平均子块不均匀度 {np.abs(flat - flat.mean(1, keepdims=True)).mean():.3f}"
          f"  （原图对应 {uni / max(cell_mean.mean(), 1e-6):.3f}）")

    # --- 3. 三种渲染方式的子块误差 -------------------------------------- #
    brightness = np.clip(_brightness(cell_mean, "luminance"), 0, 1)
    base = ramp.indices_for(brightness)

    # A 现状：按亮度选字 + 单色（用现有的候选机制，保持一致性）
    a_idx, a_ink, a_color = _pick_candidates(ramp, base, cell_mean, bg, 1)
    err_a = np.array([
        spatial_error(cover[a_idx[y, x]], a_color[y, x], sub[y, x], bg)
        for y in range(rows) for x in range(cols)
    ]).reshape(rows, cols)

    # B 形状感知：在亮度允许的窗口内挑子块误差最小的字形（仍是一字一色）
    window = 3
    best_idx = base.copy()
    best_err = err_a.copy()
    best_color = a_color.copy()
    for offset in range(window):
        idx = np.minimum(base + offset, len(glyphs.chars) - 1)
        k, err = fit_glyph(cover[idx], sub, bg)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_idx = np.where(better, idx, best_idx)
        best_color = np.where(better[..., None], k, best_color)
    err_b = best_err

    # C 子块着色上限：同一字形，四个子块各一色
    err_c = np.array([
        per_subblock_error(cover[a_idx[y, x]], sub[y, x], bg).mean()
        for y in range(rows) for x in range(cols)
    ]).reshape(rows, cols)

    print(f"\n[3] 子块（2x2）分辨率上的平均绝对色差")
    print(f"    A 现状（按亮度选字 + 单色）      {err_a.mean():.4f}")
    print(f"    B 形状感知（窗口 ±{window} 内挑方位） {err_b.mean():.4f}"
          f"   改善 {1 - err_b.mean()/err_a.mean():.1%}，换字比例 {(best_idx != a_idx).mean():.0%}")
    print(f"    C 子块各自着色（上限）          {err_c.mean():.4f}"
          f"   改善 {1 - err_c.mean()/err_a.mean():.1%}")

    # --- 4. 加入定向字符后形状感知能到哪 -------------------------------- #
    oriented = GlyphSet(DEFAULT_CHARS + "".join(c for c in DIRECTIONAL if c not in DEFAULT_CHARS),
                        None, font_size)
    cover_o = sub_coverages(oriented)
    ramp_o = oriented.ramp
    base_o = ramp_o.indices_for(brightness)
    swap = {c: i for i, c in enumerate(oriented.chars)}
    best_o, err_o = base_o.copy(), None
    for offset in range(window):
        idx = np.minimum(base_o + offset, len(oriented.chars) - 1)
        k, err = fit_glyph(cover_o[idx], sub, bg)
        if err_o is None:
            err_o = err
        else:
            better = err < err_o
            err_o = np.where(better, err, err_o)
            best_o = np.where(better, idx, best_o)
    # 把 A 现状在扩表之后的对应字形找出来，保证可比
    a_o = np.array([[swap[glyphs.chars[a_idx[y, x]]] for x in range(cols)] for y in range(rows)])
    err_a_o = np.array([
        spatial_error(cover_o[a_o[y, x]], a_color[y, x], sub[y, x], bg)
        for y in range(rows) for x in range(cols)
    ]).reshape(rows, cols)
    fav = np.argsort(-asym)[:5]
    print(f"    加入 {len(oriented.chars) - len(glyphs.chars)} 个定向字符后：")
    print(f"      A 现状（同样按亮度选字）        {err_a_o.mean():.4f}")
    print(f"      B 形状感知                      {err_o.mean():.4f}"
          f"   改善 {1 - err_o.mean()/err_a_o.mean():.1%}")
    print(f"      用到的定向字符占比 {(best_o >= len(glyphs.chars)).mean():.0%}"
          f"，最常被选中的：" + "".join(
              oriented.chars[i] for i in np.argsort(-np.bincount(best_o.ravel(),
                  minlength=len(oriented.chars)))[:6]))

    # --- 5. 代价：色调分级被瓜分 --------------------------------------- #
    print(f"\n[4] 代价：字形槽位是唯一的，给了方位就没法给色调")
    print(f"    现状每级 1 个字形 -> 亮度分级 {len(glyphs.chars)} 档，"
          f"最大空档 {float(np.diff(ramp.coverage).max()):.3f}")
    per_level = len(oriented.chars) / len(glyphs.chars)
    print(f"    若每档要 {per_level:.1f} 个方位变体，同尺寸字符集的色调档数会降到约 "
          f"{len(glyphs.chars)/per_level:.1f} 档")
    print(f"    另外：填充色已按 K = C/ink 补偿，所以换字形**不改变**单元平均色"
          f"（只要不撞墨量上限），代价只落在纹理语义上")


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "samples/sample.png")
