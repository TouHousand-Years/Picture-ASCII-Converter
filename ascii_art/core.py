"""彩色图片 -> 彩色 ASCII 字符阵列 的转换引擎。

亮度分级（查找表）
------------------
不再使用手写的亮度节点，而是**实测**每个字符的覆盖率：

1. 把字符集里每个字符用白色渲染到黑底上，取整块字符区域的平均亮度，
   这就是该字符的"墨量"(ink) —— 也就是它的覆盖比例。
2. 把 ink 线性归一化：墨量最大的字符定义为纯白 (1.0)，空格的墨量 0.0
   仍然是纯黑 (0.0)。于是每个字符得到自己的归一化亮度。
3. 查表时取"最小的平均亮度不小于目标亮度的字符"，也就是在归一化亮度
   上做上界（ceiling）查找。

这样分级自动跟随字体的真实字形，换字体、换字号都不用重新手调节点。

填色
----
填充色由"让字符区域的平均颜色尽量等于原图该区域的平均颜色"解出来：

    字符区域平均色 = ink * K + (1 - ink) * 背景色

令它等于原图区域的平均色 C，解出 K = (C - (1 - ink) * 背景色) / ink，
再夹到 [0, 1]。对黑底这就是 K = C / ink —— 覆盖率低的字符用更亮的颜色补偿。
K 被夹住时说明该颜色超出了"这个字符能表现的亮度上限"，此时已经是最优解。

注意这个模型下纯黑底的最大墨量决定了输出亮度上限（多数等宽字体最密的
ASCII 字符墨量约 0.35），想要接近原图的亮度就把全角块字符（如 U+2588）
加进字符集，它的墨量约为 1.0。

明暗之外
--------
* 亮度度量方式（metric）与 gamma 依旧可调。
* 新增图像饱和度（image_saturation），作用于每个单元的平均色。
* invert 让暗处用密字符，配合浅色背景使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

__all__ = [
    "CharRamp",
    "GlyphSet",
    "AsciiOptions",
    "AsciiResult",
    "convert",
    "convert_file",
    "find_default_font",
    "as_tuple_rgb",
    "DEFAULT_CHARS",
    "ASCII_CHARS",
    "BLOCK_CHARS",
    "BRIGHTNESS_METRICS",
]

#: 默认字符集。
#: 前 11 个是题目给定的分级（" " 到 "@"），后 5 个是为填满亮度空档实测挑出来的：
#: 只用这 16 个字符，跨字号的最大亮度空档从 0.403 降到 0.165。
#: 顺序无关紧要 —— 实际分级由运行时实测的墨量决定。
DEFAULT_CHARS = " .:~|=+%$#@S&oOs"

#: 高保真字符集：加上块元素后墨量能覆盖到 1.0，配色可以精确还原原图亮度。
#: （不是 ASCII，需要 TrueType 字体里有这些字形，按需用 --chars 指定。）
BLOCK_CHARS = " ░▒▓█"

#: 全部可打印 ASCII（0x20~0x7E），95 个字符。字形全部保留，分级仍由实测墨量决定。
ASCII_CHARS = " " + "".join(chr(code) for code in range(0x21, 0x7F))

#: 边界/线条检测是否启用。
#:
#: **暂时为 False（已屏蔽）**：实测效果不达预期 —— 几何判定虽然能把「长直边界」和
#: 「细节丰富的纹理」分开，但在真实插画上标记偏稀疏、每条线两端各少一格，整体观感
#: 没有明显提升，而且边界本身稀疏的图会一格都不标。代码与测试都保留着，
#: 把这里改成 True 即可恢复（CLI/GUI 上的入口也要一并加回，见 README 对应小节）。
EDGE_DETECTION_ENABLED = False

#: 用于标记边界/线条方向的字符。``-`` 与 ``_`` 靠笔画在单元内的**纵向位置**区分：
#: ``-`` 在中间、``_`` 贴基线，谁离检测到的边界更近就用谁。
EDGE_GLYPHS: tuple[str, ...] = ("|", "\\", "/", "-", "_")

#: 求结构张量时的细分倍数：每个字符单元再切成 SUB x SUB 个采样点。
#: 实测：一条 1 像素宽的细线，梯度会落在它两侧相邻的两个采样点上，两者相隔约 2 个
#: 采样点；只有 SUB >= 12 时这个跨度才小到能被判为「一条窄带」（SUB=4 时会读到 0.87
#: 而被误判成弥散），所以这里取 12。
_EDGE_SUB = 12
#: 相干性门槛：低于它的单元被认为是各向同性的纹理，不强制替换。
_EDGE_COHERENCE = 0.5
#: 条带集中度默认容差：把单元内的幅度范围投影到法线方向，均匀摊开时为 1.0、
#: 极窄的一条线接近 0。只有低于容差的单元才算「线条/边界」。
#: 这一条是区分「长直边界」和「细节丰富的纹理」的关键：纹理的能量摊在整格上，
#: 干净边界的能量集中在一条窄带里。可以用 --edges 覆盖。
_EDGE_BAND = 0.3
#: 最小局部对比度：低于它的单元（近似平坦）不标记，免得在纯色区被噪声触发。
_EDGE_MIN_CONTRAST = 0.04
#: 邻域方向一致性的夹角容差。取 30 度：既要求邻居朝向一致，又容许跨一个
#: 45 度的方向档，免得带一点弧度的线被切断。
_EDGE_ANGLE_TOL = np.radians(30.0)

#: 亮度度量方式 -> 说明。取值也可作为 ``AsciiOptions.metric`` 使用。
BRIGHTNESS_METRICS: dict[str, str] = {
    "luminance": "Perceptual luminance Rec.709 (0.2126R + 0.7152G + 0.0722B), default",
    "luma": "Perceptual luma Rec.601 (0.299R + 0.587G + 0.114B)",
    "value": "HSV value max(R, G, B); saturated colors appear brighter",
    "lightness": "HSL lightness (max + min) / 2",
    "average": "Arithmetic mean (R + G + B) / 3",
}

COLOR_MODES: dict[str, str] = {
    "match": "Solve for a fill color closest to the source area's average color (default)",
    "pure": "Legacy mode: maximize hue purity (vivid colors, less accurate brightness)",
}

EQUALIZE_MODES: dict[str, str] = {
    "none": "No equalization (default)",
    "global": "Global histogram equalization: stretch the overall brightness distribution",
    "local": "Local adaptive equalization: equalize tiles and blend them for local contrast",
}

# 首选等宽字体。ImageFont.truetype 会在系统字体目录里查找这些文件名。
_FONT_CANDIDATES: tuple[str, ...] = (
    "consola.ttf",            # Windows: Consolas
    "CascadiaMono.ttf",       # Windows Terminal 自带
    "CascadiaCode.ttf",
    "cour.ttf",               # Windows: Courier New
    "lucon.ttf",              # Windows: Lucida Console
    "DejaVuSansMono.ttf",     # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "Menlo.ttc",              # macOS
    "SFMono-Regular.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "NotoSansMono-Regular.ttf",
)


# --------------------------------------------------------------------------- #
# 字符分级表
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CharRamp:
    """亮度 -> 字符 的查找表，由实测字形覆盖率构建。

    ``chars`` 按实测墨量升序；``coverage`` 是归一化后的亮度（末项恒为 1.0）；
    ``ink`` 是未归一化的实测覆盖率，配色解算要用它。
    """

    chars: tuple[str, ...]
    coverage: np.ndarray
    ink: np.ndarray

    def __post_init__(self) -> None:
        if len(self.chars) != len(self.coverage) or len(self.chars) != len(self.ink):
            raise ValueError("Character count, normalized brightness, and ink levels must have the same length")
        if not self.chars:
            raise ValueError("Character lookup table cannot be empty")

    # -- 查询 -------------------------------------------------------------- #
    def index_for(self, brightness: float) -> int:
        """最小的平均亮度不小于 ``brightness`` 的那一级（上界查找）。"""
        idx = int(np.searchsorted(self.coverage, float(brightness), side="left"))
        return min(max(idx, 0), len(self.chars) - 1)

    def indices_for(self, brightness: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.coverage, brightness, side="left")
        return np.clip(idx, 0, len(self.chars) - 1).astype(np.int32)

    def char_for(self, brightness: float) -> str:
        return self.chars[self.index_for(brightness)]

    def ink_for(self, index: np.ndarray | int) -> np.ndarray:
        return self.ink[index]

    def table(self) -> list[tuple[int, str, float, float]]:
        """(级次, 字符, 归一化亮度, 实测墨量)，方便打印或展示。"""
        return [
            (i, ch, float(cov), float(ink))
            for i, (ch, cov, ink) in enumerate(zip(self.chars, self.coverage, self.ink))
        ]

    @property
    def max_ink(self) -> float:
        """字符集能达到的最大覆盖率 —— 也就是黑底下输出亮度的天花板。"""
        return float(self.ink.max())


# --------------------------------------------------------------------------- #
# 字形蒙版
# --------------------------------------------------------------------------- #
def find_default_font() -> str | None:
    """返回系统里第一个可用的等宽字体名/路径，找不到返回 None。"""
    for candidate in _FONT_CANDIDATES:
        try:
            ImageFont.truetype(candidate, 12)
        except OSError:
            continue
        return candidate
    return None


class GlyphSet:
    """渲染字符蒙版、实测墨量，并生成 :class:`CharRamp`。

    ``cell_w`` / ``cell_h`` 是渲染用的字符单元尺寸（像素），
    二者之比就是采样网格必须遵守的宽高比。
    """

    def __init__(
        self,
        chars: str | Sequence[str],
        font_path: str | None = None,
        font_size: int = 20,
    ) -> None:
        seq = list(chars) if not isinstance(chars, str) else list(chars)
        # 去重但保持用户给定顺序，重复的字符对分级没有意义
        seen: set[str] = set()
        self.requested = [c for c in seq if not (c in seen or seen.add(c))]
        if not self.requested:
            raise ValueError("Character set cannot be empty")
        bad = [c for c in self.requested if len(c) != 1]
        if bad:
            raise ValueError(f"Each character-set item must be one character; received {bad!r}")

        self.font_size = int(font_size)
        self.font_path = font_path or find_default_font()
        if self.font_path is None:
            raise RuntimeError(
                "No usable monospace font found; specify a .ttf/.ttc file with font_path/--font"
            )
        try:
            self.font = ImageFont.truetype(self.font_path, self.font_size)
        except OSError as exc:
            raise RuntimeError(f"Could not load font {self.font_path!r}: {exc}") from exc

        self.ascent, self.descent = self.font.getmetrics()
        self.cell_h = max(1, self.ascent + self.descent)
        # 等宽字体所有字符步进相同，用 "M" 量一次即可
        self.cell_w = max(1, int(round(self.font.getlength("M"))))
        self.aspect = self.cell_h / self.cell_w

        raw = np.stack([self._render(ch) for ch in self.requested])
        ink = raw.reshape(len(self.requested), -1).mean(axis=1)
        if ink.max() <= 0.0:
            raise ValueError("Character set contains only whitespace; cannot build brightness levels")

        # 按实测墨量升序排；同一墨量时保持用户给定顺序（稳定）
        order = np.lexsort((np.arange(len(self.requested)), ink))
        self.chars = tuple(self.requested[i] for i in order)
        self.atlas = raw[order]
        self.ink = ink[order].astype(np.float64)
        self.ramp = CharRamp(self.chars, self.ink / self.ink.max(), self.ink)

    def _render(self, ch: str) -> np.ndarray:
        """把一个字符渲染成 ``(cell_h, cell_w)`` 的 0.0~1.0 覆盖率蒙版。"""
        if ch.isspace():
            return np.zeros((self.cell_h, self.cell_w), np.float32)
        canvas = Image.new("L", (self.cell_w, self.cell_h), 0)
        # anchor="ls"：左对齐 + 基线对齐，保证所有字符共用同一条基线
        ImageDraw.Draw(canvas).text(
            (0, self.ascent), ch, font=self.font, fill=255, anchor="ls"
        )
        return np.asarray(canvas, dtype=np.float32) / 255.0

    def describe(self) -> str:
        """给用户看的分级表。"""
        font = self.font_path or "?"
        lines = [
            f"Character ramp ({font} @ {self.font_size}px, cell {self.cell_w}x{self.cell_h}px)",
            "  Id  Char   Normalized brightness   Measured ink",
        ]
        for i, ch, cov, ink in self.ramp.table():
            shown = "' '" if ch == " " else repr(ch)
            lines.append(f" {i:>3}  {shown:<6} {cov:>10.4f} {ink:>10.4f}")
        lines.append(f"  Maximum ink {self.ramp.max_ink:.4f} — brightness ceiling on black")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 色彩工具（numpy 向量化，不能用 colorsys：那是标量且很慢）
# --------------------------------------------------------------------------- #
def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(..., 3) 的 sRGB -> H, S, V，其中 H 在 [0,1)。"""
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    chroma = mx - mn
    colored = chroma > 1e-8

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    safe = np.where(colored, chroma, 1.0)
    hue = np.where(
        mx == r,
        ((g - b) / safe) % 6.0,
        np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0),
    )
    hue = np.where(colored, (hue / 6.0) % 1.0, 0.0)
    sat = np.where(mx > 1e-8, chroma / np.where(mx > 1e-8, mx, 1.0), 0.0)
    return hue, sat, mx


def _hsv_to_rgb(hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> np.ndarray:
    """H 在 [0,1)、S/V 在 [0,1] -> (..., 3) 的 sRGB。"""
    h6 = hue * 6.0
    sector = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p = val * (1.0 - sat)
    q = val * (1.0 - f * sat)
    t = val * (1.0 - (1.0 - f) * sat)

    r = np.select([sector == 0, sector == 1, sector == 2, sector == 3, sector == 4],
                  [val, q, p, p, t], default=val)
    g = np.select([sector == 0, sector == 1, sector == 2, sector == 3, sector == 4],
                  [t, val, val, q, p], default=p)
    b = np.select([sector == 0, sector == 1, sector == 2, sector == 3, sector == 4],
                  [p, p, t, val, val], default=q)
    return np.stack([r, g, b], axis=-1)


def as_tuple_rgb(color: object) -> tuple[int, int, int] | None:
    """把 ``"black"`` / ``"#rrggbb"`` / ``(r, g, b)`` 解析为 RGB 三元组。

    ``None`` / ``"transparent"`` / ``"none"`` 返回 ``None``，表示背景透明。
    """
    if color is None:
        return None
    if isinstance(color, str):
        name = color.strip().lower()
        if name in ("transparent", "none", "alpha"):
            return None
        if name == "black":
            return (0, 0, 0)
        if name == "white":
            return (255, 255, 255)
        text = name.lstrip("#")
        if len(text) == 3:
            text = "".join(c * 2 for c in text)
        if len(text) != 6:
            raise ValueError(f"Could not parse color {color!r}")
        try:
            return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
        except ValueError as exc:
            raise ValueError(f"Could not parse color {color!r}") from exc
    seq = tuple(int(c) for c in color)  # type: ignore[arg-type]
    if len(seq) == 4:
        seq = seq[:3]
    if len(seq) != 3:
        raise ValueError(f"Color must have 3 components; received {color!r}")
    return seq  # type: ignore[return-value]


def _brightness(rgb: np.ndarray, metric: str) -> np.ndarray:
    """(..., 3) 的 sRGB -> (...,) 的亮度，范围 0~1。"""
    if metric == "luminance":
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    if metric == "luma":
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if metric == "value":
        return rgb.max(axis=-1)
    if metric == "lightness":
        return (rgb.max(axis=-1) + rgb.min(axis=-1)) / 2.0
    if metric == "average":
        return rgb.mean(axis=-1)
    raise ValueError(
        f"Unknown brightness metric {metric!r}; choose from: {', '.join(BRIGHTNESS_METRICS)}"
    )


def _adjust_saturation(rgb: np.ndarray, factor: float) -> np.ndarray:
    """调整图像饱和度，1.0 为原样。

    做法是把颜色朝"同亮度的灰"推：``C' = gray + (C - gray) * factor``，
    与 Pillow 的 ``ImageEnhance.Color`` 一致。这样饱和度旋钮**只改颜色、
    不改明暗**，ASCII 的疏密结构不会被顺手改掉（HSV 的 S 缩放会保留 V 而不是
    亮度，鲜艳图一降饱和就洗成一片白，画面结构全丢了）。
    """
    if factor == 1.0:
        return rgb
    gray = (
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    )[..., None]
    return np.clip(gray + (rgb - gray) * factor, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 直方图均衡化
# --------------------------------------------------------------------------- #
_HIST_BINS = 256


def _tile_positions(length: int, tiles: int) -> np.ndarray:
    """把 ``length`` 个单元尽量均匀地分给 ``tiles`` 个块，返回每单元的块号。"""
    return np.minimum(np.arange(length) * tiles // length, tiles - 1)


def _tile_cdf(
    brightness: np.ndarray, tiles_y: int, tiles_x: int, clip_limit: float
) -> np.ndarray:
    """逐块统计亮度直方图，返回拉伸后的累积分布 ``(tiles_y, tiles_x, bins)``。

    ``cdf[t, k]`` 是块 t 里第 k 个 bin 的**包含自身**的累积概率，按 cdf 最小非零值
    拉伸到 [0, 1]（于是最暗处映射为 0、最亮处映射为 1）。整块只有一个取值的退化
    情况（大面积纯色）没有可拉伸的对比度，直接退回恒等映射，免得纯色区被推到黑或白。
    """
    rows, cols = brightness.shape
    bins = _HIST_BINS
    index = np.clip((brightness * bins).astype(np.int32), 0, bins - 1)
    tile_id = (
        _tile_positions(rows, tiles_y)[:, None] * tiles_x
        + _tile_positions(cols, tiles_x)[None, :]
    )
    counts = np.bincount(
        (tile_id * bins + index).ravel(),
        minlength=tiles_y * tiles_x * bins,
    ).reshape(tiles_y * tiles_x, bins).astype(np.float64)

    if clip_limit > 0:
        # CLAHE 的限幅。阈值取 clip × 总样本数 / bin 数，它等价于把映射曲线的
        # 斜率（也就是局部对比度的放大倍数）限制在 clip 上下。下限 1 个计数是为了
        # 不让稀疏直方图被削平：块只有几十个单元时每个 bin 平均还不到 1 个样本，
        # 没有这个下限就会把真实细节一起削掉。
        limit = np.maximum(
            clip_limit * counts.sum(axis=1, keepdims=True) / bins, 1.0
        )
        excess = np.maximum(counts - limit, 0.0).sum(axis=1, keepdims=True)
        counts = np.minimum(counts, limit) + excess / bins

    cdf = counts.cumsum(axis=1)
    nonzero = np.where(cdf > 0, cdf, np.inf)
    cdf_min = nonzero.min(axis=1, keepdims=True)
    cdf_min = np.where(np.isfinite(cdf_min), cdf_min, 0.0)
    total = counts.sum(axis=1, keepdims=True)
    span = total - cdf_min
    stretched = (cdf - cdf_min) / np.maximum(span, 1e-9)

    identity = (np.arange(bins, dtype=np.float64) + 0.5) / bins
    degenerate = span <= 1e-9                    # 整块只有一个取值
    stretched = np.where(degenerate, identity[None, :], stretched)
    return np.clip(stretched, 0.0, 1.0).reshape(tiles_y, tiles_x, bins)


def _apply_cdf(
    brightness: np.ndarray, cdf: np.ndarray, tiles_y: int, tiles_x: int
) -> np.ndarray:
    """按累积分布映射亮度，块之间用双线性插值过渡（避免方块痕迹）。

    bin 内不做插值：离散均衡化的映射本来就是阶梯函数，256 个 bin 的台阶在图上是
    看不见的，而插入插值反而会让正好落在 bin 下沿的取值（大面积纯色求平均很常见）
    取到不含自身的累积值，把纯色区压向黑端。
    """
    rows, cols = brightness.shape
    index = np.clip((np.clip(brightness, 0.0, 1.0) * _HIST_BINS).astype(np.int32),
                    0, _HIST_BINS - 1)

    # 块中心在"单元坐标"里的位置，算出每单元落在哪两个块之间
    fy = (np.arange(rows) + 0.5) * tiles_y / rows - 0.5
    fx = (np.arange(cols) + 0.5) * tiles_x / cols - 0.5
    y0 = np.clip(np.floor(fy).astype(np.int32), 0, tiles_y - 1)
    x0 = np.clip(np.floor(fx).astype(np.int32), 0, tiles_x - 1)
    y1, x1 = np.minimum(y0 + 1, tiles_y - 1), np.minimum(x0 + 1, tiles_x - 1)
    wy = np.clip(fy - y0, 0.0, 1.0)[:, None]
    wx = np.clip(fx - x0, 0.0, 1.0)[None, :]

    def sample(ty: np.ndarray, tx: np.ndarray) -> np.ndarray:
        return cdf[ty[:, None], tx[None, :], index]

    top = sample(y0, x0) * (1.0 - wx) + sample(y0, x1) * wx
    bottom = sample(y1, x0) * (1.0 - wx) + sample(y1, x1) * wx
    return top * (1.0 - wy) + bottom * wy


def _equalize_brightness(
    brightness: np.ndarray, mode: str, window: int, clip_limit: float
) -> np.ndarray:
    """直方图均衡化。``global`` 就是一个块，``local`` 才分块。"""
    if mode == "none":
        return brightness
    rows, cols = brightness.shape
    if mode == "global":
        tiles_y = tiles_x = 1
    else:
        span = max(2, int(window))
        tiles_y = max(1, int(np.ceil(rows / span)))
        tiles_x = max(1, int(np.ceil(cols / span)))
    cdf = _tile_cdf(brightness, tiles_y, tiles_x, clip_limit)
    return np.clip(_apply_cdf(brightness, cdf, tiles_y, tiles_x), 0.0, 1.0)


def _equalize_image(
    mean_rgb: np.ndarray, opts: "AsciiOptions"
) -> np.ndarray:
    """对图像做直方图均衡化：亮度走均衡曲线，颜色按同一比例缩放。

    也就是说均衡化后的图就是新的配色目标（色相与彩度比例保持不变，只有明暗被
    重新分布）。所有亮度度量都对颜色是一次的，所以缩放后再算一次亮度正好等于
    均衡化的目标值。
    """
    if opts.equalize == "none":
        return mean_rgb
    brightness = np.clip(_brightness(mean_rgb, opts.metric), 0.0, 1.0)
    target = _equalize_brightness(
        brightness, opts.equalize, opts.equalize_window, opts.equalize_clip
    )
    scale = np.where(brightness > 1e-6, target / np.maximum(brightness, 1e-6), 1.0)
    return np.clip(mean_rgb * scale[..., None], 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 边界 / 线条方向
# --------------------------------------------------------------------------- #
def _stroke_centroid(mask: np.ndarray) -> float:
    """字形笔画在单元内的归一化纵向重心（0 = 顶边，1 = 底边）。"""
    weights = mask.sum(axis=1)
    total = weights.sum()
    if total <= 1e-9:
        return 0.5
    ys = (np.arange(mask.shape[0]) + 0.5) / mask.shape[0]
    return float((weights * ys).sum() / total)


def _edge_scores(
    rgba: Image.Image, cols: int, rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """逐单元汇总结构张量，判断「这里有没有一条边界/线条」。

    在比单元网格更细的 ``_EDGE_SUB`` 倍网格上分析，所以单元内部的细线也能测到。
    返回 ``(强度, 相干性, 线条方向角, 纵向重心, 条带集中度)``，形状都是 ``(rows, cols)``。

    * **强度**用子采样点内的**局部幅度范围** (max-min)，不是梯度幅值：BOX 降采样会
      按落点稀释细线的梯度（同一条 1px 亮线，落在子采样中间时梯度是落在边界时的两倍），
      范围则与落点无关，而且亮线和暗线同样能测到。
    * **条带集中度**把幅度范围投影到法线方向求加权标准差：能量集中成一条窄带时接近 0，
      均匀摊开时约为 1。这才是「线条」的定义 —— 只看梯度能量总和时，
      「处处都有梯度」的纹理整片压过「只有一条窄带上有梯度」的干净边界。
    * **相干性**来自结构张量，衡量方向是否一致（各向同性的纹理会被它挡掉）。
    """
    sub = _EDGE_SUB
    gray_img = rgba.convert("L")
    width, height = gray_img.size
    size = (cols * sub, rows * sub)

    gray = np.asarray(gray_img.resize(size, Image.Resampling.BOX), np.float32) / 255.0
    gy, gx = np.gradient(gray)

    # 子采样点内的局部幅度范围：先做 k x k 的极大/极小滤波，
    # 让细到一像素的线在降采样后仍然保留幅度
    span_px = max(height / rows, width / cols) / sub
    k = max(3, int(round(span_px)) | 1)
    high = np.asarray(gray_img.filter(ImageFilter.MaxFilter(k)), np.float32)
    low = np.asarray(gray_img.filter(ImageFilter.MinFilter(k)), np.float32)
    span_img = Image.fromarray(np.clip(high - low, 0.0, 255.0).astype(np.uint8), "L")
    span = np.asarray(span_img.resize(size, Image.Resampling.BOX), np.float32) / 255.0

    def cell_sum(values: np.ndarray) -> np.ndarray:
        return values.reshape(rows, sub, cols, sub).sum(axis=(1, 3))

    def cell_block(values: np.ndarray) -> np.ndarray:
        return values.reshape(rows, sub, cols, sub).transpose(0, 2, 1, 3)

    jxx, jyy, jxy = cell_sum(gx * gx), cell_sum(gy * gy), cell_sum(gx * gy)
    energy = jxx + jyy
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / np.maximum(energy, 1e-12)

    # 强度：单元内最强的 1/4 子采样点的幅度范围均值
    per_cell = cell_block(span).reshape(rows, cols, sub * sub)
    keep = max(1, sub * sub // 4)
    strength = np.sort(per_cell, axis=-1)[..., -keep:].mean(axis=-1)

    # 结构张量的主特征向量是**梯度**方向，线条方向与它垂直
    gradient_angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    line_angle = (gradient_angle + np.pi / 2.0) % np.pi

    # 幅度范围沿法线方向的加权重心与标准差（子采样点在单元内的归一化偏移）
    offsets = (np.arange(sub) + 0.5) / sub - 0.5                    # -0.5 ~ 0.5
    normal_x = np.cos(gradient_angle)
    normal_y = np.sin(gradient_angle)
    projected = (offsets[None, None, :, None] * normal_y[..., None, None]
                 + offsets[None, None, None, :] * normal_x[..., None, None])
    weight = cell_block(span * span)                               # (rows, cols, sub, sub)
    total = np.maximum(weight.sum(axis=(2, 3)), 1e-12)
    center = (weight * projected).sum(axis=(2, 3)) / total
    spread = np.sqrt(np.maximum(
        (weight * (projected - center[..., None, None]) ** 2).sum(axis=(2, 3))
        / total, 0.0))
    # 均匀分布的 sigma 约 0.2887，用它归一化：1 = 摊满整格，0 = 极窄一条线
    band = np.clip(spread / 0.2887, 0.0, 2.0)

    # 纵向重心单独用「幅度范围的纵向分布」求：'-' 与 '_' 只关心边界在单元内的
    # 纵向位置，这个量必须与法线朝向无关
    vertical = weight.sum(axis=3)                                  # (rows, cols, sub)
    levels = (np.arange(sub) + 0.5) / sub
    centroid = ((vertical * levels).sum(axis=2)
                / np.maximum(vertical.sum(axis=2), 1e-12))
    return strength, coherence, line_angle, centroid, band


def _shifted(values: np.ndarray, offset_y: np.ndarray, offset_x: np.ndarray):
    """按逐格偏移取邻居；越界处返回填充值与原值，并给出越界掩码。

    偏移是逐格的（每个格子的线条方向不同），所以不能用单纯的 np.roll。
    """
    rows, cols = values.shape
    ys = np.arange(rows)[:, None] + offset_y
    xs = np.arange(cols)[None, :] + offset_x
    inside = (ys >= 0) & (ys < rows) & (xs >= 0) & (xs < cols)
    return values[np.clip(ys, 0, rows - 1), np.clip(xs, 0, cols - 1)], inside


def _aligned_neighbour_mask(
    candidates: np.ndarray, line_angle: np.ndarray
) -> np.ndarray:
    """候选格沿自身线条方向的**两侧**是否都有方向一致的伙伴。

    这是「邻域方向一致性」：单侧有伙伴只说明旁边还有一个点（两格的一小段线也算），
    两侧都有才说明这一段是连成线的 —— 这正是「长直边界」和「孤立误检」的分界。
    代价是每条线两端会各少一格。

    邻域取到 ±2 而不是 8 邻域：线与格子晶格未必对齐，单元空间 28 度的浅斜线相邻格
    相差 (1,2) 列，只看 8 邻域会在拐点处断链。
    """
    positive = np.zeros(candidates.shape, dtype=bool)   # 线条方向的前方有伙伴
    negative = np.zeros(candidates.shape, dtype=bool)   # 后方有伙伴
    direction_x = np.cos(line_angle)
    direction_y = np.sin(line_angle)
    normal_x, normal_y = -direction_y, direction_x
    straight = np.pi / 2.0
    reach = 2
    for offset_y in range(-reach, reach + 1):
        for offset_x in range(-reach, reach + 1):
            if offset_y == 0 and offset_x == 0:
                continue
            # 把偏移分解到「沿线条方向」和「垂直于线条方向」两个分量：
            # 垂直于方向上的偏差要小（确实贴着这条线，而不是隔两行的另一条平行线），
            # 沿方向上的推进要够（确实是在往前接，而不是斜着够到隔壁）
            along = offset_x * direction_x + offset_y * direction_y
            across = offset_x * normal_x + offset_y * normal_y
            close = np.abs(across) <= 0.8
            forward = np.abs(along) >= 0.5
            neighbour, inside = _shifted(candidates, offset_y, offset_x)
            angle, _ = _shifted(line_angle, offset_y, offset_x)
            # 夹角按 mod pi 算（线条方向不分正反）
            delta = np.abs(((angle - line_angle + straight) % np.pi) - straight)
            support = inside & close & forward & neighbour & (delta <= _EDGE_ANGLE_TOL)
            ahead = along >= 0.0
            positive |= support & ahead
            negative |= support & ~ahead
    return candidates & positive & negative


def _edge_glyph_index(
    glyphs: GlyphSet, rgba: Image.Image, cols: int, rows: int, strength: float
) -> tuple[np.ndarray, float]:
    """哪些单元要强制换成方向字符，换成哪一个。返回 ``(序号, 强制比例)``，-1 表示不换。

    ``strength`` 是**条带集中度容差**：0 关闭，越小越严格（只认很细的线条），
    0.3 左右是实测的合适值。判定是**逐格几何门限**，不做分位数排名 ——
    强度测量本身有约 30% 的落点噪声（同一条 1px 线在不同格子里的实测强度会差几十个
    百分点），一旦按排序挑，同一条边的不同格子就会被差别对待、整段漏掉。
    合格 = 相干性达标（方向一致）**且**条带集中度达标（能量集中在一条窄带里）
    **且**局部对比度够大**且**邻域里能找到方向一致的伙伴。条带集中度那一条才是把
    「长直边界」和「细节丰富的纹理」分开的关键 —— 只看梯度能量总和时，纹理处处有
    梯度，会整片压过干净的直边；最后一条则去掉孤立的误检，只留下连成线的部分。
    """
    magnitude, coherence, line_angle, centroid, band = _edge_scores(rgba, cols, rows)
    index = np.full((rows, cols), -1, dtype=np.int32)
    if strength <= 0.0:
        return index, 0.0

    # 逐格判定「这里确实是一条线」：方向一致 + 能量集中在窄带里 + 对比度够大。
    # 不做分位数排名，同一种结构必然得到同样的判定。
    force = ((coherence >= _EDGE_COHERENCE)
             & (band <= max(0.02, strength))
             & (magnitude >= _EDGE_MIN_CONTRAST))
    # 邻域方向一致性：孤立格不算线
    if force.any():
        force = _aligned_neighbour_mask(force, line_angle)
    if not force.any():
        return index, 0.0

    position = {ch: i for i, ch in enumerate(glyphs.chars)}
    if not all(ch in position for ch in EDGE_GLYPHS):
        return index, 0.0

    # 方向分成 0=水平 1='' 2=竖直 3='/'，水平那一档再按纵向位置挑 '-' 还是 '_'
    quadrant = np.rint(np.degrees(line_angle) / 45.0).astype(np.int32) % 4
    # '-' 在单元中部、'_' 贴基线，逐单元比谁离检测到的线条更近
    dash_c = _stroke_centroid(glyphs.atlas[position["-"]])
    under_c = _stroke_centroid(glyphs.atlas[position["_"]])
    horizontal_pick = np.where(
        np.abs(centroid - dash_c) <= np.abs(centroid - under_c),
        position["-"], position["_"],
    ).astype(np.int32)

    table = {
        0: horizontal_pick,
        1: np.int32(position["\\"]),
        2: np.int32(position["|"]),
        3: np.int32(position["/"]),
    }
    for bin_index, glyph_index in table.items():
        picked = force & (quadrant == bin_index)
        index = np.where(picked, glyph_index, index)
    return index, float(force.mean())


# --------------------------------------------------------------------------- #
# 参数与结果
# --------------------------------------------------------------------------- #
@dataclass
class AsciiOptions:
    """转换参数。"""

    #: 字符列数 —— 主要的"字符密度"旋钮：越大字符越多、越细腻。
    cols: int = 120
    #: 另一种密度指定方式：每个字符占原图的多少像素宽。给定时覆盖 ``cols``。
    cell_width: float | None = None
    font_path: str | None = None
    font_size: int = 20
    #: 字符集。顺序无所谓，分级由实测墨量决定。
    chars: str = DEFAULT_CHARS
    metric: str = "luminance"
    #: 查表前对亮度做 ``b ** gamma``；<1 提亮，>1 压暗。
    gamma: float = 1.0
    #: 图像饱和度倍数：调整原图颜色，进而影响选字与配色。1.0 为原样。
    image_saturation: float = 1.0
    #: 直方图均衡化：none / global / local。见 EQUALIZE_MODES。
    equalize: str = "none"
    #: 局部均衡化的窗口边长，单位是**字符单元**（不是像素）。越小越"局部"。
    equalize_window: int = 16
    #: 局部均衡化的对比度限幅，单位是"平均 bin 高度的倍数"，0 = 不限幅（纯 AHE）。
    #: 平坦区域出噪点时调大它；默认 2.0 相当于常见 CLAHE 的温和限幅。
    equalize_clip: float = 2.0
    #: "match" = 解出让区域平均色最接近原图的填充色（默认）；"pure" = 旧的纯度拉满。
    color_mode: str = "match"
    #: 仅 ``pure`` 模式：字符颜色饱和度，1.0 = 最高纯度。
    glyph_purity: float = 1.0
    #: 仅 ``pure`` 模式：色度低于该值的像素视为灰色、输出白色字符。
    chroma_floor: float = 0.04
    #: 边界/线条强制替换（**当前已屏蔽**，见 ``EDGE_DETECTION_ENABLED``）：
    #: 0 = 关闭（默认）；>0 = 条带集中度容差，越小越严格，
    #: 0.3 左右是实测的合适值。逐格几何门限判定（方向一致 + 能量集中在窄带里 +
    #: 局部对比度够大），符合的单元强制换成方向字符
    #: （``|`` ``\`` ``/`` ``-`` ``_``）。字符集会自动补齐缺的方向字符。
    edges: float = 0.0
    #: 高亮保色强度：0 = 逐通道截断（默认，绝对色差最小但高亮会变灰）；
    #: 1 = 等比缩放（色相与彩度完整保留，高亮不变灰，绝对色差略升）。
    highlight: float = 0.0
    #: 反相：暗处用密字符，亮处用空格。配合浅色背景使用。
    invert: bool = False
    #: 查找候选数。1 = 只取"最小的平均亮度不小于目标"的那一个（默认，原始规则）；
    #: >1 时从那一级起往上多考察几个更密的字符，取区域平均色最接近原图的。
    candidates: int = 1
    #: 输出背景色；``None`` 表示透明背景。
    background: object = (0, 0, 0)

    def validate(self) -> None:
        if self.cols < 1:
            raise ValueError("cols must be at least 1")
        if self.cell_width is not None and self.cell_width <= 0:
            raise ValueError("cell_width must be positive")
        if self.font_size < 4:
            raise ValueError("font_size is too small; it must be at least 4")
        if self.metric not in BRIGHTNESS_METRICS:
            raise ValueError(
                f"Unknown brightness metric {self.metric!r}; choose from: {', '.join(BRIGHTNESS_METRICS)}"
            )
        if self.color_mode not in COLOR_MODES:
            raise ValueError(
                f"Unknown color mode {self.color_mode!r}; choose from: {', '.join(COLOR_MODES)}"
            )
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.image_saturation < 0:
            raise ValueError("image_saturation cannot be negative")
        if not 0.0 <= self.glyph_purity <= 1.0:
            raise ValueError("glyph_purity must be between 0.0 and 1.0")
        if not 0.0 <= self.highlight <= 1.0:
            raise ValueError("highlight must be between 0.0 and 1.0")
        if not 0.0 <= self.edges <= 1.0:
            raise ValueError("edges must be between 0.0 and 1.0")
        if self.edges > 0.0 and not EDGE_DETECTION_ENABLED:
            raise ValueError(
                "Edge detection is temporarily disabled because its measured results were not satisfactory. "
                "To re-enable it, set ascii_art.core.EDGE_DETECTION_ENABLED to True and restore the CLI/GUI "
                "controls (see the corresponding README section)."
            )
        if self.candidates < 1:
            raise ValueError("candidates must be at least 1")
        if self.equalize not in EQUALIZE_MODES:
            raise ValueError(
                f"Unknown equalization mode {self.equalize!r}; choose from: {', '.join(EQUALIZE_MODES)}"
            )
        if self.equalize_window < 2:
            raise ValueError("equalize_window must be at least 2 character cells")
        if self.equalize_clip < 0:
            raise ValueError("equalize_clip cannot be negative")


@dataclass
class AsciiResult:
    """转换结果。"""

    image: Image.Image                 #: 输出图片（背景透明时是 RGBA，否则 RGB）
    lines: list[str]                   #: 每行的纯字符文本
    colors: np.ndarray                 #: (rows, cols, 3) uint8，每个字符的填充色
    brightness: np.ndarray             #: (rows, cols) float32，查表用的亮度
    ink: np.ndarray                    #: (rows, cols) float32，选中字符的实测墨量
    grid_index: np.ndarray             #: (rows, cols) int32，选中的分级序号
    base_index: np.ndarray             #: (rows, cols) int32，纯上界查找会选中的序号
    ramp: CharRamp                     #: 本次实际使用的分级表
    cols: int
    rows: int
    cell_w: int                        #: 输出图中一个字符单元的像素宽度
    cell_h: int
    source_size: tuple[int, int]
    font_path: str
    font_size: int
    added_edge_glyphs: str = ""     #: 为边界替换自动补进字符集的字符
    forced_ratio: float = 0.0       #: 被强制替换成方向字符的单元比例

    @property
    def text(self) -> str:
        """纯字符文本（不含颜色），行尾空格已去掉。"""
        return "\n".join(line.rstrip() for line in self.lines)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


# --------------------------------------------------------------------------- #
# 主转换
# --------------------------------------------------------------------------- #
def _grid_size(
    src_w: int, src_h: int, glyphs: GlyphSet, opts: AsciiOptions
) -> tuple[int, int]:
    """算出采样网格的 (列数, 行数)。

    行数由字框宽高比反推，保证每个采样单元与字符单元的宽高比一致，
    这样输出图既不变形，字符也不会被拉长压扁。
    """
    if opts.cell_width:
        cols = max(1, int(round(src_w / float(opts.cell_width))))
    else:
        cols = int(opts.cols)
    cell_w_src = src_w / cols
    cell_h_src = cell_w_src * glyphs.aspect
    rows = max(1, int(round(src_h / cell_h_src)))
    return cols, rows


def _downsample(rgba: Image.Image, cols: int, rows: int) -> np.ndarray:
    """把原图按网格做面积平均，返回 ``(rows, cols, 3)`` 的 float32。

    带 alpha 时先做预乘再平均，否则全透明像素的颜色会渗进邻居里。
    """
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    alpha = rgba.getchannel("A")
    box = Image.Resampling.BOX

    if alpha.getextrema()[0] == 255:          # 完全不透明：直接平均
        small = rgba.convert("RGB").resize((cols, rows), box)
        return np.asarray(small, dtype=np.float32) / 255.0

    r, g, b, a = rgba.split()
    premultiplied = [
        ImageChops.multiply(channel, a).resize((cols, rows), box)
        for channel in (r, g, b)
    ]
    alpha_small = np.asarray(a.resize((cols, rows), box), dtype=np.float32) / 255.0
    rgb = np.stack(
        [np.asarray(ch, dtype=np.float32) for ch in premultiplied], axis=-1
    ) / 255.0
    return rgb / np.maximum(alpha_small, 1e-6)[..., None]


def _lookup_brightness(mean_rgb: np.ndarray, opts: AsciiOptions) -> np.ndarray:
    """单元平均色 -> 查表用的亮度。"""
    brightness = _brightness(mean_rgb, opts.metric)
    if opts.gamma != 1.0:
        brightness = np.clip(brightness, 0.0, 1.0) ** opts.gamma
    brightness = np.clip(brightness, 0.0, 1.0)
    return 1.0 - brightness if opts.invert else brightness


def _solve_fill(
    ink: np.ndarray,
    mean_rgb: np.ndarray,
    bg_rgb: np.ndarray,
    highlight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """给定墨量，解出填充色，并算出它带来的区域平均色残差。

    逐通道最小二乘解是 ``K = (C - (1-ink)*bg) / ink``。它经常超出可绘制范围
    （上限 1），超出时的处理方式由 ``highlight`` 决定，也就是"高亮保色"开关：

    * ``highlight=0``：**逐通道截断** ``min(K, 1)``。绝对色差最小，但高于墨量的
      通道会被压成同一个值 —— 高亮区域因此变成纯灰、丢掉颜色。
    * ``highlight=1``：**等比缩放** ``K / max(K)``。色相与彩度比例完整保留，
      高亮仍是原色（只是暗一档），代价是绝对色差略升。
    * 中间值在两者之间线性过渡。

    返回 ``(填充色, 残差)``，残差是逐单元的平均绝对色差。
    """
    ink_c = ink[..., None]
    drawn = ink_c > 1e-6
    safe_ink = np.where(drawn, ink_c, 1.0)
    ideal = (mean_rgb - (1.0 - ink_c) * bg_rgb) / safe_ink

    color = np.clip(ideal, 0.0, 1.0)
    if highlight > 0.0:
        # 峰值只在真的超出上限时才起作用；没超出时两种规则结果完全一致
        peak = np.maximum(ideal.max(axis=-1, keepdims=True), 1.0)
        color = (1.0 - highlight) * color + highlight * (ideal / peak)
    # 空格不画任何像素，填什么颜色都无所谓，直接给原色
    color = np.clip(np.where(drawn, color, mean_rgb), 0.0, 1.0)

    rendered = ink_c * color + (1.0 - ink_c) * bg_rgb
    return color, np.abs(rendered - mean_rgb).mean(axis=-1)


def _pick_candidates(
    ramp: CharRamp,
    base_index: np.ndarray,
    mean_rgb: np.ndarray,
    bg_rgb: np.ndarray,
    count: int,
    highlight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从亮度上界那一级起，往后多考察几个更密的字符，取配色误差最小的。

    ``count=1`` 就是"只取最小的平均亮度不小于目标的那个字符"，即原来的规则。
    ``count>1`` 时这些候选都满足"平均亮度不小于目标"，在其中挑区域平均色最接近
    原图的 —— 于是字符表负责纹理，配色解算有更多余地，误差只会更小。
    误差并列时保留更稀疏的那一级（也就是原来会选的那个）。
    """
    last = len(ramp.chars) - 1
    index = best_index = base_index
    best_ink = best_color = best_err = None
    for offset in range(max(1, count)):
        index = np.minimum(base_index + offset, last)
        ink = ramp.ink[index]
        color, err = _solve_fill(ink, mean_rgb, bg_rgb, highlight)
        if best_err is None:
            best_index, best_ink, best_color, best_err = index, ink, color, err
            continue
        better = err < best_err - 1e-9          # 严格更优才换
        best_index = np.where(better, index, best_index)
        best_ink = np.where(better, ink, best_ink)
        best_color = np.where(better[..., None], color, best_color)
        best_err = np.where(better, err, best_err)
    return (
        best_index.astype(np.int32),
        best_ink.astype(np.float32),
        best_color,
    )


def _pure_colors(mean_rgb: np.ndarray, opts: AsciiOptions) -> np.ndarray:
    """旧规则：只取色相，饱和度与明度拉满。"""
    hue, sat, val = _rgb_to_hsv(mean_rgb)
    chroma = val * sat
    if opts.chroma_floor > 0:
        purity = np.clip(chroma / opts.chroma_floor, 0.0, 1.0)
    else:
        purity = (chroma > 1e-8).astype(np.float32)
    return _hsv_to_rgb(hue, opts.glyph_purity * purity, np.ones_like(purity))


def convert(source: Image.Image, opts: AsciiOptions | None = None) -> AsciiResult:
    """把彩色图片转换成彩色 ASCII 字符阵列。

    参数
    ----
    source: 任意 PIL 图片（会转成 RGBA 处理）。
    opts:   :class:`AsciiOptions`，为 None 时使用默认参数。

    返回
    ----
    :class:`AsciiResult`，含输出图片、纯文本字符阵列、每个字符的填充色。
    """
    opts = opts or AsciiOptions()
    opts.validate()

    rgba = source if source.mode == "RGBA" else source.convert("RGBA")
    src_w, src_h = rgba.size
    if src_w < 1 or src_h < 1:
        raise ValueError("Input image has no pixels")

    # 边界替换需要方向字符，缺的自动补齐（会在结果里报告补了哪些）
    added = ""
    charset = opts.chars
    if opts.edges > 0.0:
        added = "".join(ch for ch in EDGE_GLYPHS if ch not in charset)
        charset += added

    glyphs = GlyphSet(charset, opts.font_path, opts.font_size)
    ramp = glyphs.ramp
    chars = ramp.chars
    cols, rows = _grid_size(src_w, src_h, glyphs, opts)

    mean_rgb = np.clip(_downsample(rgba, cols, rows), 0.0, 1.0)
    if opts.image_saturation != 1.0:
        mean_rgb = _adjust_saturation(mean_rgb, opts.image_saturation)
    mean_rgb = _equalize_image(mean_rgb, opts)

    brightness = _lookup_brightness(mean_rgb, opts)
    base_index = ramp.indices_for(brightness)

    forced_ratio = 0.0
    bg = as_tuple_rgb(opts.background)
    bg_rgb = np.zeros(3, np.float32) if bg is None else np.array(bg, np.float32) / 255.0

    if opts.color_mode == "pure":
        # 旧配色规则不参与候选比较，选字就是纯上界查找
        grid_index = base_index
        ink = ramp.ink_for(grid_index)
        cell_rgb = _pure_colors(mean_rgb, opts)
    else:
        grid_index, ink, cell_rgb = _pick_candidates(
            ramp, base_index, mean_rgb, bg_rgb, opts.candidates, opts.highlight
        )
        if opts.edges > 0.0:
            # 强制替换只改"用哪个字形"，填充色仍按 K = C/ink 重新解一遍，
            # 所以单元平均色不会被破坏，代价只落在可达范围上
            forced, forced_ratio = _edge_glyph_index(glyphs, rgba, cols, rows, opts.edges)
            if forced_ratio > 0.0:
                grid_index = np.where(forced >= 0, forced, grid_index).astype(np.int32)
                ink = ramp.ink[grid_index]
                cell_rgb, _ = _solve_fill(ink, mean_rgb, bg_rgb, opts.highlight)

    # -- 字符网格（纯文本）------------------------------------------------ #
    flat = np.array(chars, dtype="<U1")
    lines = ["".join(row) for row in flat[grid_index]]

    # -- 着色合成 ---------------------------------------------------------- #
    cell_h, cell_w = glyphs.cell_h, glyphs.cell_w
    out_w, out_h = cols * cell_w, rows * cell_h
    opaque = bg is not None

    rgb8 = np.empty((out_h, out_w, 3), dtype=np.uint8)
    alpha8 = None if opaque else np.empty((out_h, out_w), dtype=np.uint8)

    # 逐"行条带"处理，避免为超大输出图一次性分配 float32 中间量
    strip_cells = max(1, 2_000_000 // max(1, cols * cell_h * cell_w))
    for y0 in range(0, rows, strip_cells):
        y1 = min(rows, y0 + strip_cells)
        band = y1 - y0
        band_rgb = np.zeros((band, cell_h, cols, cell_w, 3), dtype=np.float32)
        band_cov = np.zeros((band, cell_h, cols, cell_w), dtype=np.float32)

        band_idx = grid_index[y0:y1]
        band_colors = cell_rgb[y0:y1]
        for i, _ch in enumerate(chars):
            mask = glyphs.atlas[i]
            if not mask.any():
                continue                      # 空格：不画
            pos = np.nonzero(band_idx == i)
            if pos[0].size == 0:
                continue
            rr, cc = pos
            if opaque:
                # 不透明底：蒙版 × 填充色，未覆盖处留给背景
                band_rgb[rr, :, cc, :, :] = (
                    mask[None, :, :, None] * band_colors[rr, cc][:, None, None, :]
                )
            else:
                # 透明底：整块铺填充色，覆盖率交给 alpha，避免边缘出现暗边
                band_rgb[rr, :, cc, :, :] = band_colors[rr, cc][:, None, None, :]
            band_cov[rr, :, cc, :] = mask

        if opaque:
            band_rgb += (1.0 - band_cov)[..., None] * bg_rgb
        else:
            alpha8[y0 * cell_h:y1 * cell_h, :] = (
                np.clip(band_cov.reshape(band * cell_h, out_w), 0.0, 1.0) * 255.0
            ).astype(np.uint8)

        rgb8[y0 * cell_h:y1 * cell_h, :, :] = (
            np.clip(band_rgb.reshape(band * cell_h, out_w, 3), 0.0, 1.0) * 255.0
        ).astype(np.uint8)

    if opaque:
        out_img = Image.fromarray(rgb8, "RGB")
    else:
        out_img = Image.merge(
            "RGBA",
            [*Image.fromarray(rgb8, "RGB").split(), Image.fromarray(alpha8, "L")],
        )

    return AsciiResult(
        image=out_img,
        lines=lines,
        colors=(np.clip(cell_rgb, 0, 1) * 255).astype(np.uint8),
        brightness=brightness.astype(np.float32),
        ink=ink.astype(np.float32),
        grid_index=grid_index,
        base_index=base_index,
        ramp=ramp,
        cols=cols,
        rows=rows,
        cell_w=cell_w,
        cell_h=cell_h,
        source_size=(src_w, src_h),
        font_path=glyphs.font_path or "",
        font_size=glyphs.font_size,
        added_edge_glyphs=added,
        forced_ratio=forced_ratio,
    )


def convert_file(
    src_path: str, out_path: str | None = None, opts: AsciiOptions | None = None
) -> AsciiResult:
    """从文件读图转换，并把结果图片写到 ``out_path``（给了才写）。"""
    with Image.open(src_path) as img:
        img.load()
        result = convert(img, opts)
    if out_path:
        result.image.save(out_path)
    return result
