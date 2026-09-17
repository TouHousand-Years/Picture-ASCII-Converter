"""贪心搜索：为字符集补位，使亮度分级空档最小（跨字号稳健）。

用法: python tools/tune_charset.py

候选池分三档：
  1. 用户点名的"中心对称/近似中心对称"字符（旋转 180° 或镜像后仍是自己）
  2. 其它对称/准对称符号
  3. 不对称字符（对照组，用来说明是否值得为此牺牲风格）
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ascii_art.core import GlyphSet  # noqa: E402

CORE = " .:~|=+%$#@"
SIZES = (12, 20, 40)

# 用户点名可用：中心对称或近似中心对称
SYMMETRIC = "OXHS0/osxz8*+=|~:.%#@&/\\_-<>()[]"
# 其它符号（含不对称的）
OTHER_SYMBOLS = "!?^\"';,{}"
# 不对称字母（对照）
ASYMMETRIC_LETTERS = "MknwEAFjkqb"

POOLS = {
    "① 对称字符 + 数字": SYMMETRIC,
    "② 对称字符 + 其它符号": SYMMETRIC + OTHER_SYMBOLS,
    "③ 全部候选（含不对称字母）": SYMMETRIC + OTHER_SYMBOLS + ASYMMETRIC_LETTERS,
}


def levels(chars, size):
    glyphs = GlyphSet(list(chars), None, size)
    cov = glyphs.atlas.reshape(len(chars), -1).mean(axis=1)
    order = np.argsort(cov, kind="stable")
    return [chars[i] for i in order], cov[order] / cov.max()


def score(chars):
    """跨字号的最差空档 —— 只在某个字号上最优的字符集不能用。"""
    return max(float(np.diff(levels(chars, s)[1]).max()) for s in SIZES)


def search(pool, label, max_add=5, verbose=True):
    cur = CORE
    if verbose:
        print(f"--- {label} ---")
        print(f"起点（用户指定的 11 个）最差空档 = {score(cur):.3f}")
    while len(cur) < len(CORE) + max_add:
        best, best_score = None, score(cur)
        for ch in pool:
            if ch in cur:
                continue
            s = score(cur + ch)
            if s < best_score - 1e-9:
                best, best_score = ch, s
        if best is None:
            break
        cur += best
        if verbose:
            print(f"  加入 {best!r} -> 最差空档 {best_score:.3f}  ({len(cur)} 个字符)")
    if verbose:
        print(f"  结果: {cur!r}")
        for size in SIZES:
            cs, c = levels(cur, size)
            print(f"    size={size:<3}" + " ".join(f"{ch!r}:{v:.2f}" for ch, v in zip(cs, c)))
    return cur, score(cur)


if __name__ == "__main__":
    results = {}
    for label, pool in POOLS.items():
        results[label] = search(pool, label)
        print()

    print("=== 汇总 ===")
    for label, (chars, sc) in results.items():
        print(f"{label}: 最差空档 {sc:.3f}  {chars!r}")
