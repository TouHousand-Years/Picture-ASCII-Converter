"""核心引擎测试。可直接 ``python tests/test_core.py`` 运行，也兼容 pytest。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ascii_art as aa  # noqa: E402
from ascii_art.core import (  # noqa: E402
    ASCII_CHARS,
    BLOCK_CHARS,
    DEFAULT_CHARS,
    GlyphSet,
    _downsample,
    _equalize_brightness,
    _hsv_to_rgb,
    _rgb_to_hsv,
)

# 题目最初给定的 6 级，必须仍然包含在默认字符集里
SPEC_CHARS = " .:+#@"
BS = chr(92)          # 反斜杠，避免到处都是转义


def edges_enabled() -> bool:
    """边界检测当前是否启用。该功能已临时屏蔽（效果不达预期），
    相关用例自动跳过，代码与断言都留着以便恢复。"""
    from ascii_art.core import EDGE_DETECTION_ENABLED

    return EDGE_DETECTION_ENABLED


def solid(color, size=(220, 220)) -> Image.Image:
    """0~1 的浮点色会被换算成 0~255；整数色直接使用。"""
    if any(isinstance(c, (float, np.floating)) for c in color):
        color = tuple(round(float(c) * 255) for c in color)
    return Image.new("RGB", size, color)


def cell_means(result) -> np.ndarray:
    """从输出图里量出每个字符区域的真实平均颜色。"""
    arr = np.asarray(result.image.convert("RGB"), np.float32) / 255.0
    arr = arr.reshape(result.rows, result.cell_h, result.cols, result.cell_w, 3)
    return arr.mean(axis=(1, 3))


def mean_abs_error(result, source_rgb) -> np.ndarray:
    """每个单元「输出区域平均色 - 原图区域平均色」的逐通道绝对误差。"""
    return np.abs(cell_means(result) - np.asarray(source_rgb, np.float32)).mean(axis=-1)


# --------------------------------------------------------- 墨量与分级表 --
def test_ink_is_measured_from_white_on_black():
    glyphs = GlyphSet(" .#@", None, 20)
    ink = glyphs.ink
    assert ink.shape == (4,)
    assert np.all(ink >= 0) and np.all(ink <= 1)
    assert ink[0] == 0.0                       # 空格不落墨
    assert np.all(np.diff(ink) >= 0)           # 按墨量升序
    assert ink.max() == ink[-1]
    # 墨量必须等于蒙版的真实平均值（白字黑底平均）
    for i in range(len(glyphs.chars)):
        assert abs(glyphs.atlas[i].mean() - ink[i]) < 1e-6


def test_ramp_is_normalised_so_max_ink_is_pure_white():
    ramp = GlyphSet(DEFAULT_CHARS, None, 20).ramp
    assert ramp.coverage[0] == 0.0              # 纯黑仍然是纯黑
    assert abs(ramp.coverage[-1] - 1.0) < 1e-9  # 墨量最大的定义为纯白
    assert np.all(np.diff(ramp.coverage) >= -1e-12)
    assert np.allclose(ramp.coverage, ramp.ink / ramp.ink.max())


def test_default_charset_keeps_the_original_levels():
    assert set(SPEC_CHARS) <= set(DEFAULT_CHARS)
    assert set("~|=%$") <= set(DEFAULT_CHARS)   # 题目补充的字符也都在


def test_charset_levels_are_spread_out():
    """扩展字符集的意义：亮度分级不能留下大空档。"""
    def worst_gap(chars, size):
        return float(np.diff(GlyphSet(chars, None, size).ramp.coverage).max())

    for size in (12, 20, 40):
        extended, bare = worst_gap(DEFAULT_CHARS, size), worst_gap(" .:~|=+%$#@", size)
        assert extended < 0.18                 # 补位后没有明显空档
        assert bare > 0.30                     # 只加点名的 5 个还不够
        assert bare > 1.5 * extended


def test_glyphset_dedups_and_validates():
    glyphs = GlyphSet(" .#@#.", None, 20)
    assert len(glyphs.chars) == 4               # 重复字符被去掉
    assert sorted(glyphs.chars) == sorted(" .#@")
    for bad in ("", None, ["ab", "c"], ["", "a"]):
        try:
            GlyphSet(bad, None, 20)             # type: ignore[arg-type]
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"应当拒绝 {bad!r}")
    try:
        GlyphSet("   ", None, 20)               # 全空白 -> 建不出分级
    except ValueError:
        pass
    else:
        raise AssertionError("全空白字符集应当报错")


def test_ascii_charset_covers_all_printable_ascii():
    from ascii_art.core import ASCII_CHARS

    assert len(ASCII_CHARS) == 95
    assert ASCII_CHARS[0] == " "
    assert [ord(c) for c in ASCII_CHARS[1:]] == list(range(0x21, 0x7F))
    glyphs = GlyphSet(ASCII_CHARS, None, 20)
    assert len(glyphs.chars) == 95
    # 分级仍按实测墨量排，最密的还是 @，墨量上限和 16 字符集一样
    assert glyphs.ramp.chars[-1] == "@"
    assert abs(glyphs.ramp.max_ink - GlyphSet(DEFAULT_CHARS, None, 20).ramp.max_ink) < 1e-9
    # 分级更细：最大空档比 16 字符集小得多
    gaps = np.diff(glyphs.ramp.coverage)
    assert gaps.max() < 0.09
    assert gaps.max() < np.diff(GlyphSet(DEFAULT_CHARS, None, 20).ramp.coverage).max()
    # 但近半数字形的墨量几乎相同，多出来的级其实分不开
    assert (np.diff(glyphs.ink) < 0.002).sum() > 40


def test_ink_threshold_dedup_shrinks_the_full_ascii_set():
    """按墨量最小间隔去重：95 个字符只剩 40 个左右是真正分得开的。"""
    from ascii_art.core import ASCII_CHARS

    ink = GlyphSet(ASCII_CHARS, None, 20).ink
    for threshold, lower, upper in ((0.002, 45, 70), (0.005, 30, 50), (0.010, 18, 32)):
        kept, last = 0, -1e9
        for value in ink:
            if value - last >= threshold:
                kept += 1
                last = value
        assert lower <= kept <= upper, (threshold, kept)


def test_rendered_region_mean_is_source_capped_at_ink():
    """填色规则的直接后果：黑底下渲染出的单元平均色 = min(源色, 字形墨量)，逐通道。

    这条等式把"输出上限"说清楚了：它不是色调分级被截断，而是**逐通道被字形墨量
    封顶**——通道低于墨量的部分被精确还原，高于墨量的部分被压到墨量。

    推论：灰度内容上，因为上界查找选中的字形墨量 ≈ 归一化亮度 x 最大墨量，
    整幅图等于源图乘上最大墨量（0.346），是一个**统一的全局增益**，色调结构完好。
    """
    for chars in (DEFAULT_CHARS, BLOCK_CHARS, ASCII_CHARS):
        for cols, tone in ((24, (0.18, 0.18, 0.18)), (24, (0.62, 0.25, 0.12)),
                           (40, (0.5, 0.5, 0.5))):
            src = solid(tone, (240, 180))
            result = aa.convert(src, aa.AsciiOptions(cols=cols, chars=chars))
            cell = np.clip(_downsample(src.convert("RGBA"), result.cols, result.rows), 0, 1)
            rendered = np.asarray(result.image.convert("RGB"), np.float32).reshape(
                result.rows, result.cell_h, result.cols, result.cell_w, 3
            ).mean(axis=(1, 3)) / 255.0
            predicted = np.minimum(cell, result.ink[..., None])
            assert np.abs(predicted - rendered).max() < 3e-3, (chars, tone)


def test_gray_ramp_is_quantised_into_the_ink_range():
    """归一化把源图的整个亮度范围映射到了字形墨量范围 [0, 最大墨量]。

    所以那个"上限"不是色调范围被截断，而是**输出亮度被整体压进墨量区间**：
    渲染值 = min(源亮度, 墨量)，是阶梯函数，比值在每级顶端为 1、底端降到 ~最大墨量。
    结果是色调结构基本无损（相关 0.99+），但整幅图平均暗了约 1/最大墨量。
    """
    gray = np.linspace(0.0, 1.0, 240, dtype=np.float32)
    src = Image.fromarray((np.tile(gray, (160, 1)) * 255).astype(np.uint8), "L")
    result = aa.convert(src.convert("RGB"), aa.AsciiOptions(cols=48))
    cell = np.clip(_downsample(src.convert("RGBA"), result.cols, result.rows), 0, 1)[..., 2]
    rendered = np.asarray(result.image.convert("RGB"), np.float32).reshape(
        result.rows, result.cell_h, result.cols, result.cell_w, 3
    ).mean(axis=(1, 3))[..., 2] / 255.0

    assert rendered.max() <= result.ramp.max_ink + 2e-3      # 不超过墨量上限
    assert np.corrcoef(rendered.ravel(), cell.ravel())[0, 1] > 0.99   # 结构保留
    gain = (rendered * cell).sum() / (rendered * rendered).sum()
    assert abs(gain - 1.0 / result.ramp.max_ink) / (1.0 / result.ramp.max_ink) < 0.12
    raw = np.abs(rendered - cell).mean()
    after = np.abs(gain * rendered - cell).mean()
    assert after < 0.3 * raw, (raw, after)                   # 一个常数解释了大部分误差


def test_block_chars_remove_the_compression():
    """墨量能到 1.0 的字符集才真正消掉那个压缩：增益回到 1。"""
    gray = np.linspace(0, 1, 240, dtype=np.float32)
    src = Image.fromarray((np.tile(gray, (160, 1)) * 255).astype(np.uint8), "L")
    result = aa.convert(
        src.convert("RGB"), aa.AsciiOptions(cols=48, chars=BLOCK_CHARS, metric="value")
    )
    cell = np.clip(_downsample(src.convert("RGBA"), result.cols, result.rows), 0, 1)[..., 2]
    rendered = np.asarray(result.image.convert("RGB"), np.float32).reshape(
        result.rows, result.cell_h, result.cols, result.cell_w, 3
    ).mean(axis=(1, 3))[..., 2] / 255.0
    gain = (rendered * cell).sum() / (rendered * rendered).sum()
    assert abs(gain - 1.0) < 0.05
    assert np.abs(rendered - cell).mean() < 0.03


def test_highlight_zero_is_the_original_behaviour():
    """highlight=0 必须与原来的逐通道截断完全一致。"""
    img = solid((0.95, 0.6, 0.2), (200, 200))
    base = aa.convert(img, aa.AsciiOptions(cols=12, font_size=16))
    same = aa.convert(img, aa.AsciiOptions(cols=12, font_size=16, highlight=0.0))
    assert np.array_equal(base.colors, same.colors)


def test_highlight_preserves_chroma_of_bright_colours():
    """高亮保色的核心：亮而饱和的颜色不再被压成灰。

    渲染的最大通道不可能超过字形墨量，所以能保住的是**归一化色度**
    （色度 / 最大通道，也就是"有多鲜艳"）。逐通道截断会把它压到接近 0，
    等比缩放则完整保留。
    """
    img = solid((0.95, 0.6, 0.2), (200, 200))

    def saturation(hl):
        result = aa.convert(img, aa.AsciiOptions(cols=12, font_size=16, highlight=hl))
        got = np.asarray(result.image.convert("RGB"), np.float32).reshape(
            result.rows, result.cell_h, result.cols, result.cell_w, 3
        ).mean(axis=(1, 3))[0, 0] / 255.0
        peak = max(float(got.max()), 1e-9)
        return float(got.max() - got.min()) / peak, float(got.max())

    source_saturation = (0.95 - 0.2) / 0.95
    plain, _ = saturation(0.0)
    kept, peak = saturation(1.0)
    assert plain < 0.15 * source_saturation          # 截断：几乎全灰
    assert abs(kept - source_saturation) < 0.03      # 缩放：鲜艳度精确保留
    assert peak <= 0.35                              # 但仍受墨量封顶，不会变亮
    values = [saturation(h / 4)[0] for h in range(5)]
    assert all(a <= b + 1e-9 for a, b in zip(values, values[1:])), values


def test_highlight_never_leaves_unit_range_and_keeps_hue():
    rng = np.random.default_rng(7)
    img = Image.fromarray((rng.random((240, 240, 3)) * 255).astype(np.uint8), "RGB")
    result = aa.convert(img, aa.AsciiOptions(cols=40, font_size=16, highlight=1.0))
    assert result.colors.min() >= 0 and result.colors.max() <= 255
    # 整体色调结构反而更好（除掉常数增益后的残差下降）
    cell = np.clip(_downsample(img.convert("RGBA"), result.cols, result.rows), 0, 1)
    rendered = np.asarray(result.image.convert("RGB"), np.float32).reshape(
        result.rows, result.cell_h, result.cols, result.cell_w, 3
    ).mean(axis=(1, 3)) / 255.0
    gain = (rendered * cell).sum() / (rendered * rendered).sum()
    off = aa.convert(img, aa.AsciiOptions(cols=40, font_size=16, highlight=0.0))
    rendered_off = np.asarray(off.image.convert("RGB"), np.float32).reshape(
        off.rows, off.cell_h, off.cols, off.cell_w, 3
    ).mean(axis=(1, 3)) / 255.0
    gain_off = (rendered_off * cell).sum() / (rendered_off * rendered_off).sum()
    assert np.abs(gain * rendered - cell).mean() < np.abs(gain_off * rendered_off - cell).mean()


def test_highlight_validation():
    for bad in (dict(highlight=-0.1), dict(highlight=1.5)):
        try:
            aa.convert(solid((1, 2, 3), (40, 40)), aa.AsciiOptions(**bad))
        except ValueError:
            continue
        raise AssertionError(f"应当拒绝 {bad!r}")


def edge_scene() -> Image.Image:
    """已知答案的线条图：竖线、靠上的横线、靠下的横线、两条对角线。"""
    px = np.full((400, 400), 40, np.uint8)
    px[100, :] = 230                      # 落在单元上部 -> 应该用 '-'
    px[302, :] = 230                      # 落在单元下部 -> 应该用 '_'
    px[:, 200] = 230                      # 竖线 -> '|'
    for i in range(120):
        px[10 + i, 20 + i] = 230          # 从左上到右下 -> '\'
    for i in range(120):
        px[10 + i, 380 - i] = 230         # 从左下到右上 -> '/'
    return Image.fromarray(px, "L").convert("RGB")


def test_edges_off_changes_nothing():
    src = edge_scene()
    base = aa.convert(src, aa.AsciiOptions(cols=40))
    same = aa.convert(src, aa.AsciiOptions(cols=40, edges=0.0))
    assert base.lines == same.lines
    assert same.added_edge_glyphs == "" and same.forced_ratio == 0.0


def test_edge_glyphs_are_forced_by_orientation():
    if not edges_enabled():
        return          # 功能已屏蔽
    """竖线出 |、对角线出 / 与 \、横线出 - 或 _。"""
    result = aa.convert(edge_scene(), aa.AsciiOptions(cols=40, edges=0.3))
    # 方向字符缺的会被自动补进来；'|' 本来就在默认字符集里，不用补
    edge_set = {"|", BS, "/", "-", "_"}
    assert edge_set <= set(result.ramp.chars)          # 五个方向字符都可用
    assert set(result.added_edge_glyphs) == edge_set - set(DEFAULT_CHARS)
    assert "|" not in result.added_edge_glyphs
    assert result.forced_ratio > 0.02
    grid = np.array([list(line) for line in result.lines])
    rows, cols = grid.shape

    def dominant(values):
        chars, counts = np.unique(values, return_counts=True)
        return str(chars[counts.argmax()])

    # 竖线所在列应当是 |
    col = grid[:, 20]
    assert dominant(col) == "|", "竖线应当强制成 |"
    # 靠上的横线 -> '-'（在单元中部），靠下的横线 -> '_'（贴基线）
    upper_row = int(100 / 400 * rows)      # 用截断定位：源 y 落在哪一行
    lower_row = int(302 / 400 * rows)
    assert dominant(grid[upper_row, :]) == "-", (upper_row, dominant(grid[upper_row, :]))
    assert dominant(grid[lower_row, :]) == "_", dominant(grid[lower_row, :])
    # 两条对角线方向都要出现
    chars = set("".join(result.lines))
    assert "/" in chars and BS in chars


def test_edge_dash_and_underscore_track_position():
    if not edges_enabled():
        return          # 功能已屏蔽
    """'-' 与 '_' 的区别就是边界在单元内的纵向位置，这是本功能的要求。"""
    upper = aa.convert(edge_scene(), aa.AsciiOptions(cols=40, edges=0.3))
    upper_rows = "".join(upper.lines)
    # 把横线整体下移半个单元高度，'-' 应当换成 '_'
    px = np.full((400, 400), 40, np.uint8)
    px[100, :] = 230
    px[110, :] = 230
    shifted = aa.convert(
        Image.fromarray(px, "L").convert("RGB"), aa.AsciiOptions(cols=40, edges=0.3)
    )
    assert upper.lines != shifted.lines
    assert "_" in set("".join(shifted.lines)) or "-" in set("".join(shifted.lines))


def test_aligned_neighbour_mask_rule():
    """邻域方向一致性规则：单侧有伙伴不够，两侧都有（长度 >= 3 的链）才留。"""
    from ascii_art.core import _aligned_neighbour_mask

    candidates = np.zeros((5, 9), dtype=bool)
    angle = np.zeros((5, 9), dtype=np.float32)          # 全部水平
    candidates[2, 1:8] = True                          # 一条 7 格水平线
    candidates[4, 4] = True                            # 一个孤立格
    candidates[0, 4:6] = True                          # 只有两格的一段
    out = _aligned_neighbour_mask(candidates, angle)

    assert out[2, 1:8].sum() == 5, "线内部应当保留"
    assert not out[2, 1] and not out[2, 7], "两端各少一格（只有单侧伙伴）"
    assert not out[4, 4], "孤立格必须去掉"
    assert not out[0, 4:6].any(), "只有两格、单侧伙伴，必须去掉"


def test_aligned_neighbour_mask_respects_direction():
    """方向不一致的邻居不算伙伴：一条竖线和一条横线相邻也互不支撑。"""
    from ascii_art.core import _aligned_neighbour_mask

    candidates = np.zeros((7, 7), dtype=bool)
    angle = np.full((7, 7), np.pi / 2, dtype=np.float32)   # 全部竖直
    candidates[2:5, 3] = True                              # 竖直三格：内部应保留
    candidates[3, 4] = True                                # 右侧邻居方向按竖直算
    out = _aligned_neighbour_mask(candidates, angle)
    assert out[3, 3], "竖直链的内部应当保留"

    angle2 = np.zeros((7, 7), dtype=np.float32)            # 全部水平
    angle2[3, 3] = np.pi / 2                               # 中间那格却是竖直
    out2 = _aligned_neighbour_mask(candidates, angle2)
    assert not out2[3, 3], "朝向与邻居不一致的格子应当去掉"


def test_edges_mark_a_long_line_end_to_end():
    if not edges_enabled():
        return          # 功能已屏蔽
    """一整条长直线应当被强制标记（两端各少一格）。"""
    px = np.full((400, 400), 40, np.uint8)
    px[300, :] = 230
    src = Image.fromarray(px, "L").convert("RGB")
    plain = aa.convert(src, aa.AsciiOptions(cols=40, edges=0.0))
    marked = aa.convert(src, aa.AsciiOptions(cols=40, edges=0.3))
    forced = marked.grid_index != plain.grid_index
    row = int(300 / 400 * marked.rows)
    assert forced[row].mean() > 0.8
    assert marked.forced_ratio > 0


def test_edges_keep_the_region_mean_colour():
    """强制换字形不会破坏单元平均色 —— 填充色按 K = C/ink 会重新补偿。"""
    src = Image.fromarray(
        (np.random.default_rng(4).random((240, 240, 3)) * 255).astype(np.uint8), "RGB"
    )

    def cell_error(edges):
        result = aa.convert(src, aa.AsciiOptions(cols=30, font_size=16, edges=edges))
        want = np.clip(_downsample(src.convert("RGBA"), result.cols, result.rows), 0, 1)
        got = np.asarray(result.image.convert("RGB"), np.float32).reshape(
            result.rows, result.cell_h, result.cols, result.cell_w, 3
        ).mean(axis=(1, 3)) / 255.0
        return float(np.abs(got - want).mean())

    if not edges_enabled():
        return                                  # 功能已屏蔽
    off, on = cell_error(0.0), cell_error(0.1)
    assert on < off * 1.3                       # 代价很小


def test_edges_validation():
    for bad in (dict(edges=-0.1), dict(edges=1.5)):
        try:
            aa.convert(solid((1, 2, 3), (40, 40)), aa.AsciiOptions(**bad))
        except ValueError:
            continue
        raise AssertionError(f"应当拒绝 {bad!r}")


def test_edge_detection_is_disabled_for_now():
    """边界检测已临时屏蔽：开启它必须明确报错，而不是悄悄给个不达预期的结果。"""
    if edges_enabled():
        return
    try:
        aa.convert(solid((120, 90, 60), (120, 120)), aa.AsciiOptions(cols=12, edges=0.3))
    except ValueError as exc:
        assert "temporarily disabled" in str(exc)
    else:
        raise AssertionError("edges>0 should fail while edge detection is disabled")


def test_describe_lists_every_level():
    text = GlyphSet(" .#@", None, 20).describe()
    assert len(text.splitlines()) == 3 + 4      # 表头两行 + 收尾一行 + 4 级
    assert "Maximum ink" in text


# ------------------------------------------------------------- 查表规则 --
def test_lookup_picks_smallest_level_not_below_target():
    """规则：找「最小的平均亮度不小于其的字符」。"""
    ramp = GlyphSet(DEFAULT_CHARS, None, 20).ramp
    for i, level in enumerate(ramp.coverage):
        if i > 0 and level <= ramp.coverage[i - 1]:
            continue                       # 墨量并列的级，首级代表这一档
        assert ramp.index_for(float(level)) == i
        if i > 0:
            just_below = float(level) - 1e-6
            assert ramp.index_for(just_below) == i     # 差一点点也要升到本级
            assert ramp.coverage[i] >= just_below
    assert ramp.index_for(0.0) == 0                       # 纯黑 -> 空格
    assert ramp.index_for(1.0) == len(ramp.coverage) - 1   # 纯白 -> 最密
    assert ramp.index_for(-1.0) == 0                      # 越界不崩
    assert ramp.index_for(9.0) == len(ramp.coverage) - 1


def test_space_only_when_brightness_is_exactly_zero():
    """上界查找的推论：只要还有一点亮度，就会升到下一个非空字符。"""
    ramp = GlyphSet(DEFAULT_CHARS, None, 20).ramp
    assert ramp.char_for(0.0) == " "
    assert ramp.char_for(1e-7) == ramp.chars[1]


def test_vectorised_lookup_matches_scalar():
    ramp = GlyphSet(DEFAULT_CHARS, None, 20).ramp
    values = np.linspace(0.0, 1.0, 401, dtype=np.float32)
    idx = ramp.indices_for(values)
    assert all(ramp.char_for(float(v)) == ramp.chars[i] for v, i in zip(values, idx))


def test_char_for_covers_whole_brightness_range():
    ramp = GlyphSet(DEFAULT_CHARS, None, 20).ramp
    seen = {ramp.char_for(b / 100) for b in range(101)}
    assert len(seen) >= 12      # 16 级字符集在 101 个采样点上至少用到 12 个


# ------------------------------------------------------------- 填色规则 --
def test_fill_color_never_leaves_unit_range():
    rng = np.random.default_rng(3)
    img = Image.fromarray((rng.random((240, 240, 3)) * 255).astype(np.uint8), "RGB")
    result = aa.convert(img, aa.AsciiOptions(cols=40, font_size=16))
    assert result.colors.min() >= 0 and result.colors.max() <= 255
    assert result.image.size == (result.cols * result.cell_w, result.rows * result.cell_h)


def test_match_color_reproduces_region_mean_exactly():
    """块元素字符集的墨量能到 ~1.0，配 value 度量时应当精确还原区域平均色。"""
    opts = aa.AsciiOptions(cols=2, font_size=20, chars=BLOCK_CHARS, metric="value")
    for rgb in ((0.15, 0.15, 0.15), (0.45, 0.2, 0.45), (0.62, 0.62, 0.62),
                (0.8, 0.4, 0.1), (0.9, 0.9, 0.9)):
        got = cell_means(aa.convert(solid(rgb), opts))
        assert np.allclose(got, rgb, atol=0.02), (rgb, got[0, 0])


def test_pure_white_is_limited_by_the_max_ink():
    """纯白也还原不了：最密的字符也盖不满整个字框，这是物理上限。"""
    opts = aa.AsciiOptions(cols=2, font_size=20, chars=BLOCK_CHARS, metric="value")
    result = aa.convert(solid((1.0, 1.0, 1.0)), opts)
    got = float(cell_means(result).mean())
    assert abs(got - result.ramp.max_ink) < 0.02
    assert 0.9 < got < 1.0


def test_match_beats_pure_at_the_stated_objective():
    """规则要求「差值最小」：同一字符下 match 的误差必须不大于 pure。"""
    rng = np.random.default_rng(11)
    colors = rng.random((60, 3)).astype(np.float32) * 0.9
    match_err, pure_err = [], []
    for rgb in colors:
        src = solid(tuple(rgb), (120, 120))
        m = aa.convert(src, aa.AsciiOptions(cols=1, font_size=20, color_mode="match"))
        p = aa.convert(src, aa.AsciiOptions(cols=1, font_size=20, color_mode="pure"))
        match_err.append(mean_abs_error(m, rgb)[0, 0])
        pure_err.append(mean_abs_error(p, rgb)[0, 0])
    match_err, pure_err = np.array(match_err), np.array(pure_err)
    assert (match_err <= pure_err + 1e-6).all()
    assert match_err.mean() < pure_err.mean()          # 而且总体明显更好


def test_match_preserves_channels_that_fit_under_the_ink():
    """墨量不够的通道会被夹住，够得着的通道应当精确命中。"""
    rgb = (0.9, 0.05, 0.05)
    got = cell_means(aa.convert(solid(rgb), aa.AsciiOptions(cols=1, font_size=20)))[0, 0]
    assert abs(got[1] - rgb[1]) < 0.01 and abs(got[2] - rgb[2]) < 0.01
    assert got[0] <= rgb[0] + 1e-6


def test_max_ink_is_the_brightness_ceiling():
    assert 0.25 < GlyphSet(DEFAULT_CHARS, None, 20).ramp.max_ink < 0.45
    assert GlyphSet(BLOCK_CHARS, None, 20).ramp.max_ink > 0.9   # 块元素补上这个天花板


def test_pure_mode_is_still_available():
    opts = dict(cols=4, font_size=16, color_mode="pure")
    assert tuple(aa.convert(solid((150, 60, 60)), aa.AsciiOptions(**opts)).colors[0, 0]) \
        == (255, 0, 0)
    assert tuple(aa.convert(solid((128, 128, 128)), aa.AsciiOptions(**opts)).colors[0, 0]) \
        == (255, 255, 255)


# --------------------------------------------------------------- 饱和度 --
def test_image_saturation_zero_desaturates():
    # 用块元素 + value 度量，配色不会撞上天花板，测的才是真饱和度
    opts = dict(cols=4, font_size=16, chars=BLOCK_CHARS, metric="value")
    result = aa.convert(solid((230, 90, 60)), aa.AsciiOptions(image_saturation=0.0, **opts))
    c = result.colors[0, 0].astype(int)
    assert c.max() - c.min() <= 2                       # 三通道拉平 -> 灰


def test_image_saturation_boosts_chroma():
    opts = dict(cols=6, font_size=16, chars=BLOCK_CHARS, metric="value")

    def chroma(k):
        c = aa.convert(
            solid((170, 110, 90)), aa.AsciiOptions(image_saturation=k, **opts)
        ).colors.astype(np.float32)
        return float((c.max(axis=-1) - c.min(axis=-1)).mean())

    assert chroma(0.0) < chroma(0.5) < chroma(1.0) < chroma(1.8)


def test_image_saturation_keeps_brightness_intact():
    """饱和度旋钮只该改颜色，不该改明暗结构（否则 ASCII 疏密会被顺手改掉）。"""
    img = Image.fromarray(
        (np.random.default_rng(5).random((120, 120, 3)) * 255).astype(np.uint8), "RGB"
    )
    base = aa.convert(img, aa.AsciiOptions(cols=20, font_size=16, metric="luma"))
    for k in (0.0, 0.5, 1.7):
        other = aa.convert(img, aa.AsciiOptions(cols=20, font_size=16, metric="luma",
                                                image_saturation=k))
        assert np.allclose(base.brightness, other.brightness, atol=1e-5)
        assert base.lines == other.lines          # 选字也不受影响


# ------------------------------------------------------------- 候选数 --
def test_candidates_one_is_the_plain_ceiling_rule():
    """默认 candidates=1 必须与"只取上界那一个"完全一致。"""
    src = Image.fromarray(
        (np.random.default_rng(9).random((180, 180, 3)) * 255).astype(np.uint8), "RGB"
    )
    result = aa.convert(src, aa.AsciiOptions(cols=30, font_size=16))
    expected = result.ramp.indices_for(result.brightness)
    assert (result.grid_index == expected).all()
    assert (result.base_index == expected).all()


def test_candidates_never_increase_the_colour_error():
    """候选变多时误差只会更小：规则本身就是在更大的候选集上求最优。"""
    src = Image.open("samples/sample.png") if Path("samples/sample.png").exists() else None
    if src is None:
        src = solid((180, 120, 60), (240, 160))
    opts = dict(cols=40, font_size=16)
    one = aa.convert(src, aa.AsciiOptions(candidates=1, **opts))
    many = aa.convert(src, aa.AsciiOptions(candidates=6, **opts))

    def err(result):
        a = np.asarray(result.image.convert("RGB"), np.float32) / 255.0
        ref = _downsample(src.convert("RGBA"), result.cols, result.rows)
        got = a.reshape(result.rows, result.cell_h, result.cols, result.cell_w, 3)
        return np.abs(got.mean(axis=(1, 3)) - ref).mean(axis=-1)

    e1, e6 = err(one), err(many)
    assert (e6 <= e1 + 1e-6).all()
    assert e6.mean() < e1.mean()


def test_more_candidates_reach_for_denser_glyphs():
    """中亮单元在候选变多后会换用更密的字符，从而更接近原图亮度。"""
    src = solid((0.6, 0.6, 0.6))
    opts = dict(cols=4, font_size=20)
    one = aa.convert(src, aa.AsciiOptions(candidates=1, **opts))
    many = aa.convert(src, aa.AsciiOptions(candidates=8, **opts))
    assert many.grid_index[0, 0] > one.grid_index[0, 0]
    assert many.grid_index[0, 0] == len(many.ramp.chars) - 1     # 一路升到最密的
    assert cell_means(many)[0, 0].mean() > cell_means(one)[0, 0].mean()


def test_candidates_keeps_black_empty():
    """纯黑单元不该因为候选变多而凭空长出笔画（误差并列时保留更稀疏的）。"""
    result = aa.convert(
        solid((0, 0, 0)), aa.AsciiOptions(cols=4, font_size=16, candidates=8)
    )
    assert result.grid_index.max() == 0
    assert (result.base_index == 0).all()


def test_candidates_validation():
    try:
        aa.convert(solid((1, 2, 3), (40, 40)), aa.AsciiOptions(candidates=0))
    except ValueError:
        return
    raise AssertionError("candidates=0 应当被拒绝")


# --------------------------------------------------------- 直方图均衡化 --
def flat_background_scene() -> np.ndarray:
    """教科书式样张：90% 平坦中灰，角上两个小块，块内各有细节条纹。

    全局均衡化会被大片中灰主导，把小块内部的对比度压掉；局部均衡化才能救回来。
    """
    scene = np.full((200, 200), 0.50)
    scene += np.linspace(-0.02, 0.02, 200)[None, :]
    stripes = (np.arange(24) % 8) / 8.0
    scene[10:34, 10:34] = 0.04 + stripes[:, None] * 0.10      # 暗块
    scene[10:34, 160:184] = 0.90 + stripes[:, None] * 0.08    # 亮块
    return scene


def test_equalize_global_is_identity_on_uniform_ramp():
    """均匀分布没什么可拉的，均衡化后应当基本不变。"""
    ramp = np.tile(np.linspace(0, 1, 128), (32, 1))
    out = _equalize_brightness(ramp, "global", 16, 0.0)
    assert np.abs(out - ramp).max() < 0.01


def test_equalize_global_keeps_a_flat_image_flat():
    """退化（单值）块没有可拉伸的对比度，必须原样保留而不是推到全黑或全白。"""
    for value in (0.0, 0.4, 1.0):
        out = _equalize_brightness(np.full((32, 128), value), "global", 16, 0.0)
        assert (out.max() - out.min()) < 0.01          # 仍然是平的
        assert abs(out.mean() - value) < 0.01          # 位置也没被搬走


def test_equalize_global_expands_a_binary_image():
    dark = np.where(np.random.default_rng(0).random((64, 64)) < 0.8, 0.05, 0.25)
    out = _equalize_brightness(dark, "global", 16, 0.0)
    assert np.unique(np.round(out[dark < 0.1], 3)).tolist() == [0.0]
    assert np.unique(np.round(out[dark > 0.1], 3)).tolist() == [1.0]


def test_equalize_global_raises_histogram_entropy():
    def entropy(values, bins=64):
        counts = np.histogram(values, bins=bins, range=(0, 1))[0].astype(float)
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    noisy = np.clip(np.random.default_rng(1).beta(2, 5, (64, 64)), 0, 1)
    out = _equalize_brightness(noisy, "global", 16, 0.0)
    assert entropy(out) > entropy(noisy) + 0.3
    assert entropy(out) <= np.log(64) + 1e-6            # 不能超过均匀分布的上限


def test_equalize_local_recovers_detail_that_global_loses():
    scene = flat_background_scene()
    block = (slice(10, 34), slice(10, 34))
    before = scene[block].std()

    glob = _equalize_brightness(scene, "global", 16, 0.0)
    local = _equalize_brightness(scene, "local", 8, 0.0)
    assert glob[block].std() < before                   # 全局反而把块内细节压掉了
    assert local[block].std() > 3 * before               # 局部把它救回来
    assert local[block].std() > 10 * glob[block].std()


def contrast_gain(before: np.ndarray, after: np.ndarray) -> float:
    """实测局部对比度的放大倍数（梯度方向斜率的中位数比值）。"""
    src = np.abs(np.diff(before, axis=1))
    out = np.abs(np.diff(after, axis=1))
    mask = src > 1e-4
    return float(np.median(out[mask] / src[mask]))


def test_equalize_clip_bounds_the_contrast_gain():
    """限幅的核心作用：给局部对比度的放大倍数封顶。

    不限幅时平滑背景会被放大上百倍、直接拉满量程（画面炸成噪点）；
    限幅后稳在几倍以内，背景还是背景。
    """
    smooth = np.tile(np.linspace(0.47, 0.53, 200), (50, 1))
    raw = _equalize_brightness(smooth, "local", 8, 0.0)
    tame = _equalize_brightness(smooth, "local", 8, 2.0)
    assert contrast_gain(smooth, raw) > 50
    assert np.ptp(raw) > 0.5                       # 不受限：平滑背景被拉满量程
    assert contrast_gain(smooth, tame) < 5
    assert np.ptp(tame) < 0.15                     # 限幅后仍是平的


def test_equalize_clip_is_a_dial_not_a_switch():
    """限幅调大，局部放大跟着变强，且单调。"""
    smooth = np.tile(np.linspace(0.47, 0.53, 200), (50, 1))
    gains = [
        contrast_gain(smooth, _equalize_brightness(smooth, "local", 8, clip))
        for clip in (2.0, 4.0, 8.0)
    ]
    assert gains[0] <= gains[1] <= gains[2]
    assert gains[2] > gains[0]


def test_equalize_local_still_expands_real_detail_at_default_clip():
    """默认限幅之下，真正的局部细节仍然被展开，而且明显强过全局均衡化。"""
    scene = flat_background_scene()
    block = (slice(10, 34), slice(10, 34))
    local = _equalize_brightness(scene, "local", 8, 2.0)
    glob = _equalize_brightness(scene, "global", 16, 0.0)
    assert local[block].std() > 1.5 * scene[block].std()
    assert local[block].std() > 10 * glob[block].std()


def test_equalize_window_controls_locality():
    scene = flat_background_scene()
    block = (slice(10, 34), slice(10, 34))
    small = _equalize_brightness(scene, "local", 8, 0.0)
    big = _equalize_brightness(scene, "local", 200, 0.0)        # 大窗口约等于全局
    assert small[block].std() > 10 * big[block].std()


def test_equalize_none_leaves_the_picture_alone():
    assert _equalize_brightness(flat_background_scene(), "none", 16, 2.0) is not None
    src = solid((120, 90, 60), (160, 160))
    base = aa.convert(src, aa.AsciiOptions(cols=20, font_size=16))
    same = aa.convert(src, aa.AsciiOptions(cols=20, font_size=16, equalize="none"))
    assert np.array_equal(base.brightness, same.brightness)
    assert base.lines == same.lines


def test_equalize_end_to_end_spreads_the_glyph_levels():
    """低对比度图经过全局均衡化后，用到的字符层级会明显铺开。"""
    src = Image.fromarray(
        (60 + np.linspace(0, 30, 240)[None, :].repeat(240, 0)).astype(np.uint8), "L"
    ).convert("RGB")
    plain = aa.convert(src, aa.AsciiOptions(cols=60, font_size=12))
    equal = aa.convert(src, aa.AsciiOptions(cols=60, font_size=12, equalize="global"))
    assert equal.grid_index.std() > plain.grid_index.std()
    assert len(set("".join(equal.lines))) > len(set("".join(plain.lines)))


def test_equalize_changes_colours_too_not_just_glyphs():
    """均衡化是作用在图片上的：配色目标跟着变，不只是换字符。

    用偏斜（暗端堆积）的分布才测得出来 —— 均匀分布本来就是已均衡的，不该被改动。
    """
    skewed = ((np.linspace(0, 1, 240) ** 3) * 255).astype(np.uint8)
    src = Image.fromarray(np.tile(skewed, (240, 1)), "L").convert("RGB")
    plain = aa.convert(src, aa.AsciiOptions(cols=40, font_size=12))
    equal = aa.convert(src, aa.AsciiOptions(cols=40, font_size=12, equalize="global"))
    assert not np.array_equal(plain.colors, equal.colors)      # 颜色被带动了
    assert plain.lines != equal.lines                          # 选字也变了
    assert equal.brightness.mean() > plain.brightness.mean()   # 偏暗的分布被提上来


def test_equalize_validation():
    for bad in (dict(equalize="x"), dict(equalize_window=1), dict(equalize_clip=-1)):
        try:
            aa.convert(solid((1, 2, 3), (40, 40)), aa.AsciiOptions(**bad))
        except ValueError:
            continue
        raise AssertionError(f"应当拒绝 {bad!r}")


# ----------------------------------------------------------------- 颜色 --
def test_hsv_roundtrip_for_pure_hues():
    hues = np.array([0.0, 1 / 6, 0.5, 0.75, 0.999], dtype=np.float32)
    rgb = _hsv_to_rgb(hues, np.ones_like(hues), np.ones_like(hues))
    back, sat, val = _rgb_to_hsv(rgb)
    assert np.allclose(back, hues, atol=1e-4)
    assert np.allclose(sat, 1.0, atol=1e-5)
    assert np.allclose(val, 1.0, atol=1e-5)


# ----------------------------------------------------------------- 网格 --
def test_output_size_and_aspect_match_glyph_cell():
    result = aa.convert(solid((200, 150, 100), (900, 600)), aa.AsciiOptions(cols=100))
    assert result.cols == 100
    assert result.image.size == (result.cols * result.cell_w, result.rows * result.cell_h)
    cell_w_src = 900 / result.cols
    cell_h_src = 600 / result.rows
    assert abs((cell_h_src / cell_w_src) - (result.cell_h / result.cell_w)) < 0.06


def test_density_changes_cell_count_not_output_scale():
    src = solid((10, 200, 90), (800, 800))
    small = aa.convert(src, aa.AsciiOptions(cols=40, font_size=12))
    large = aa.convert(src, aa.AsciiOptions(cols=160, font_size=12))
    assert small.cols == 40 and large.cols == 160
    assert large.rows > small.rows
    assert small.cell_w == large.cell_w and small.cell_h == large.cell_h


def test_cell_width_overrides_cols():
    opts = aa.AsciiOptions(cols=999, cell_width=8, font_size=20)
    assert aa.convert(solid((10, 200, 90), (800, 800)), opts).cols == 100


def test_text_grid_shape():
    result = aa.convert(solid((30, 30, 30), (400, 240)), aa.AsciiOptions(cols=50))
    assert len(result.lines) == result.rows
    assert all(len(line) == result.cols for line in result.lines)
    assert result.grid_index.shape == (result.rows, result.cols)


# ------------------------------------------------------------- 亮度行为 --
def test_brightness_is_monotonic_on_gray_ramp():
    # 每个单元内部是常量，这样第一格正好是纯黑、最后一格正好是纯白
    ramp = np.repeat(np.linspace(0, 1, 48, dtype=np.float32), 10)
    img = Image.fromarray((np.tile(ramp, (60, 1)) * 255).astype(np.uint8), "L")
    result = aa.convert(img.convert("RGB"), aa.AsciiOptions(cols=48))
    assert np.all(np.diff(result.brightness[2]) >= -1e-6)
    assert result.lines[2][0] == " "                       # 纯黑 -> 空格
    assert result.lines[2][-1] == result.ramp.chars[-1]    # 纯白 -> 最密


def test_invert_flips_dark_and_bright():
    densest = GlyphSet(DEFAULT_CHARS, None, 16).ramp.chars[-1]
    dark, bright = solid((0, 0, 0)), solid((255, 255, 255))
    normal = aa.AsciiOptions(cols=8, font_size=16)
    inverted = aa.AsciiOptions(cols=8, font_size=16, invert=True)
    assert aa.convert(dark, normal).lines[0][0] == " "
    assert aa.convert(bright, normal).lines[0][0] == densest
    assert aa.convert(dark, inverted).lines[0][0] == densest
    assert aa.convert(bright, inverted).lines[0][0] == " "


def test_gamma_pushes_brightness_down():
    mid, extra = solid((140, 140, 140)), dict(cols=8, font_size=16)
    plain = aa.convert(mid, aa.AsciiOptions(**extra)).brightness[0, 0]
    darker = aa.convert(mid, aa.AsciiOptions(gamma=2.0, **extra)).brightness[0, 0]
    brighter = aa.convert(mid, aa.AsciiOptions(gamma=0.5, **extra)).brightness[0, 0]
    assert darker < plain < brighter


def test_metric_choice_changes_brightness():
    src = solid((0, 0, 255))          # 纯蓝：Rec.709 亮度很低，HSV 明度很高
    lum = aa.convert(src, aa.AsciiOptions(cols=4, font_size=12)).brightness[0, 0]
    val = aa.convert(
        src, aa.AsciiOptions(cols=4, font_size=12, metric="value")
    ).brightness[0, 0]
    assert lum < 0.2 and val > 0.9


# ------------------------------------------------------------ 背景 / alpha --
def test_opaque_background_returns_rgb():
    result = aa.convert(solid((255, 0, 0), (80, 80)), aa.AsciiOptions(cols=6, font_size=16))
    assert result.image.mode == "RGB"


def test_transparent_background_keeps_alpha_and_flat_fill():
    opts = aa.AsciiOptions(cols=6, font_size=16, background=None)
    result = aa.convert(solid((255, 255, 255), (80, 80)), opts)
    assert result.image.mode == "RGBA"
    alpha = np.asarray(result.image.getchannel("A"))
    rgb = np.asarray(result.image.convert("RGB"))
    assert alpha.max() == 255
    lit = alpha > 0
    # 透明底导出的是未预乘的图：笔画内部颜色恒定，覆盖率只体现在 alpha 上
    assert len(np.unique(rgb[lit].reshape(-1, 3), axis=0)) == 1
    assert (rgb[lit] == 255).all()          # 纯白原图 -> 白色字符


def test_background_color_is_used():
    result = aa.convert(
        solid((0, 0, 0), (64, 64)),
        aa.AsciiOptions(cols=4, font_size=16, background="white"),
    )
    assert set(np.unique(np.asarray(result.image).reshape(-1, 3)[:, 0])) == {255}


def test_alpha_channel_does_not_bleed_color():
    img = Image.new("RGBA", (64, 64), (255, 0, 0, 0))
    img.paste(Image.new("RGBA", (32, 64), (0, 0, 255, 255)), (32, 0))
    result = aa.convert(img, aa.AsciiOptions(cols=16, font_size=12, background=None))
    assert result.brightness[0, 0] < 0.05


# ------------------------------------------------------------- 输入输出 --
def test_options_validation():
    for bad in (dict(cols=0), dict(metric="nope"), dict(color_mode="x"),
                dict(gamma=0), dict(glyph_purity=2.0), dict(font_size=1),
                dict(image_saturation=-1)):
        try:
            aa.convert(solid((1, 2, 3), (40, 40)), aa.AsciiOptions(**bad))
        except ValueError:
            continue
        raise AssertionError(f"应当拒绝 {bad!r}")


def test_convert_file_roundtrip(tmp_path=Path("samples/_tmp")):
    tmp_path.mkdir(parents=True, exist_ok=True)
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    solid((200, 120, 40), (320, 200)).save(src)
    result = aa.convert_file(str(src), str(dst), aa.AsciiOptions(cols=40))
    assert dst.exists()
    with Image.open(dst) as reopened:
        assert reopened.size == result.image.size
    for path in (src, dst):
        path.unlink()
    tmp_path.rmdir()


def test_font_discovery_works():
    assert aa.find_default_font() is not None


# ------------------------------------------------------------------ CLI --
def test_cli_defaults_track_the_core_defaults():
    """命令行的默认值不能和核心默认值各写一份 —— 曾经因此把限幅默认值漂成了 0。"""
    from ascii_art.cli import build_parser

    args = build_parser().parse_args(["x.png"])
    opts = aa.AsciiOptions()
    for arg_name, field in (
        ("cols", "cols"), ("cell_width", "cell_width"), ("font", "font_path"),
        ("font_size", "font_size"), ("chars", "chars"), ("metric", "metric"),
        ("gamma", "gamma"), ("image_saturation", "image_saturation"),
        ("color_mode", "color_mode"), ("candidates", "candidates"),
        ("equalize", "equalize"), ("equalize_window", "equalize_window"),
        ("equalize_clip", "equalize_clip"), ("invert", "invert"),
        ("glyph_purity", "glyph_purity"),
        ("chroma_threshold", "chroma_floor"),
    ):
        assert getattr(args, arg_name) == getattr(opts, field), (
            f"--{arg_name} 的默认值 {getattr(args, arg_name)!r} "
            f"与 AsciiOptions.{field} 的 {getattr(opts, field)!r} 不一致"
        )
    # --bg 命令行给的是颜色名，比较解析后的结果
    assert aa.as_tuple_rgb(args.bg) == opts.background


def test_gui_control_defaults_track_the_core_defaults():
    """界面控件的初值同样来自核心默认值；没有图形环境就直接跳过。"""
    try:
        import tkinter as tk

        from ascii_art.gui import AsciiArtApp

        root = tk.Tk()
    except Exception:  # noqa: BLE001 - 无图形环境
        return
    try:
        root.withdraw()
        app = AsciiArtApp(root)
        opts = aa.AsciiOptions()
        for var, field in (
            (app.cols_var, "cols"), (app.font_var, "font_size"),
            (app.chars_var, "chars"), (app.metric_var, "metric"),
            (app.candidates_var, "candidates"),
            (app.image_sat_var, "image_saturation"),
            (app.equalize_var, "equalize"),
            (app.eq_window_var, "equalize_window"),
            (app.eq_clip_var, "equalize_clip"), (app.gamma_var, "gamma"),
            (app.color_mode_var, "color_mode"),
            (app.sat_var, "glyph_purity"), (app.invert_var, "invert"),
        ):
            assert var.get() == getattr(opts, field), field

        # 英文说明必须交给统一的自适应换行机制，不能再写死为旧版的 252px。
        wrapped = {label for label, _parent, _inset in app._wrapping_labels}
        assert app.candidates_hint in wrapped
        assert app.metric_hint in wrapped
        assert app.equalize_hint in wrapped
        assert app.mode_hint in wrapped
        assert app.info_label in wrapped
        assert int(app.panel.canvas.cget("width")) >= 360
        assert all(int(label.cget("wraplength")) != 252 for label in wrapped)
        assert app.watermark_var.get() is False
        assert app.watermark_position_var.get() == "Bottom Right"
        assert str(app.watermark_position_combo.cget("state")) == "disabled"
    finally:
        root.destroy()


def test_watermark_can_be_placed_in_each_corner():
    from ascii_art.gui import WATERMARK_LINES, WATERMARK_POSITIONS, _add_watermark

    source = solid((90, 120, 160), (900, 400))
    for position in WATERMARK_POSITIONS:
        result = aa.convert(source, aa.AsciiOptions(cols=80))
        before = result.image.copy()
        colors_before = result.colors.copy()
        _add_watermark(result, position, (0, 0, 0))

        first_row = 0 if position.startswith("Top") else result.rows - 2
        for offset, expected in enumerate(WATERMARK_LINES):
            row = result.lines[first_row + offset]
            if position.endswith("Left"):
                assert row.startswith(expected)
                assert row[len(expected)] == " "
            else:
                assert row.endswith(expected)
                assert row[result.cols - len(expected) - 1] == " "
        assert result.image.mode == "RGB"
        assert ImageChops.difference(before, result.image).getbbox() is not None
        assert np.array_equal(result.colors, colors_before)


def test_watermark_is_clipped_to_a_small_character_grid_and_exported_as_text():
    from ascii_art.gui import WATERMARK_LINES, _add_watermark

    result = aa.convert(
        solid((20, 30, 40), (160, 200)),
        aa.AsciiOptions(cols=20, background=None),
    )
    _add_watermark(result, "Top Left", None)
    assert result.lines[0] == WATERMARK_LINES[0][:result.cols]
    assert result.lines[1] == WATERMARK_LINES[1][:result.cols]
    assert result.text.splitlines()[:2] == result.lines[:2]
    assert result.image.mode == "RGBA"


def test_cli_list_ramp_and_convert(tmp_path=Path("samples/_tmp_cli")):
    root = Path(__file__).resolve().parents[1]

    # --help 里嵌了默认字符集，含 % 时必须转义，否则 argparse 会当场崩
    proc = subprocess.run(
        [sys.executable, "-m", "ascii_art", "--help"],
        cwd=root, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--candidates" in proc.stdout and DEFAULT_CHARS in proc.stdout

    proc = subprocess.run(
        [sys.executable, "-m", "ascii_art", "x.png", "--list-ramp"],
        cwd=root, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Character ramp" in proc.stdout and "Maximum ink" in proc.stdout

    tmp_path.mkdir(parents=True, exist_ok=True)
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    solid((180, 90, 40), (240, 160)).save(src)
    proc = subprocess.run(
        [sys.executable, "-m", "ascii_art", str(src), "-o", str(dst),
         "-c", "40", "--image-saturation", "1.4", "--candidates", "3"],
        cwd=root, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert dst.exists()
    assert "Candidates 3" in proc.stderr

    # --full-chars 可用，且与 --block-chars 互斥
    proc = subprocess.run(
        [sys.executable, "-m", "ascii_art", str(src), "-o", str(dst), "-c", "20",
         "--full-chars"],
        cwd=root, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Character set: 95 levels" in proc.stderr, proc.stderr
    proc = subprocess.run(
        [sys.executable, "-m", "ascii_art", str(src), "--full-chars", "--block-chars"],
        cwd=root, capture_output=True, text=True,
    )
    assert proc.returncode != 0 and "not allowed" in proc.stderr
    src.unlink()
    dst.unlink()
    tmp_path.rmdir()


# ------------------------------------------------------------------ 入口 --
def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
