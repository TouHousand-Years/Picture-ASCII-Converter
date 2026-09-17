"""命令行入口：图片 -> 彩色 ASCII 图片 / 文本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from .core import (
    ASCII_CHARS,
    BLOCK_CHARS,
    BRIGHTNESS_METRICS,
    COLOR_MODES,
    DEFAULT_CHARS,
    EQUALIZE_MODES,
    AsciiOptions,
    GlyphSet,
    convert,
    find_default_font,
)


def _esc(text: str) -> str:
    """argparse 会对 help 文本做 %-格式化，字符集里的 % 必须转义。"""
    return text.replace("%", "%%")


#: 命令行默认值一律引用核心的默认值，免得两处各写一份、改一处忘一处。
_DEFAULTS = AsciiOptions()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ascii_art",
        description="把彩色图片转换成彩色 ASCII 字符阵列，并可导出为新的图片。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m ascii_art photo.jpg -o ascii.png\n"
            "  python -m ascii_art photo.jpg -o ascii.png -c 200              # 更密的字符\n"
            "  python -m ascii_art photo.jpg -o ascii.png --cell-width 6       # 按字符像素宽控密度\n"
            "  python -m ascii_art photo.jpg -o ascii.png --image-saturation 1.5\n"
            "  python -m ascii_art photo.jpg -o ascii.png --equalize local --equalize-window 8\n"
            "  python -m ascii_art photo.jpg -o ascii.png --highlight 1        # 高亮保住颜色\n"
            "  python -m ascii_art photo.jpg -o ascii.png --chars \" ░▒▓█\"      # 高保真配色\n"
            "  python -m ascii_art photo.jpg --list-ramp                      # 看实测分级表\n"
            "  python -m ascii_art photo.jpg --ansi                           # 终端里彩色查看\n"
            "\n"
            f"默认字符集: {DEFAULT_CHARS!r}\n"
            "分级不是手写的，而是运行时实测每个字符的墨量后排出来的，\n"
            "用 --list-ramp 可以看到实测结果。\n"
        ),
    )
    parser.add_argument("input", nargs="?", help="输入图片路径（PNG/JPG/BMP/WebP/GIF 等）")
    parser.add_argument("-o", "--output", help="输出的 ASCII 图片路径；不写则只打印字符文本")

    density = parser.add_argument_group("字符密度")
    density.add_argument(
        "-c", "--cols", "--density", type=int, default=_DEFAULTS.cols, metavar="N",
        help="字符列数，越大越细腻（默认 120）",
    )
    density.add_argument(
        "--cell-width", type=float, metavar="PX",
        help="每个字符占原图的像素宽度；给定时覆盖 --cols",
    )

    font = parser.add_argument_group("字体与字符集")
    font.add_argument("--font", help="等宽字体文件路径（.ttf/.ttc）")
    font.add_argument(
        "--font-size", type=int, default=_DEFAULTS.font_size, metavar="PX",
        help="输出图里的字号，同时决定输出图大小（默认 20）",
    )
    charset = font.add_mutually_exclusive_group()
    charset.add_argument(
        "--chars", default=_DEFAULTS.chars, metavar="STR",
        help=_esc(f"字符集，顺序无所谓（默认 {DEFAULT_CHARS!r}）"),
    )
    charset.add_argument(
        "--block-chars", action="store_true", dest="block_chars",
        help=_esc(f"改用块元素字符集 {BLOCK_CHARS!r}：墨量能到 1.0，配色可精确还原亮度"),
    )
    charset.add_argument(
        "--full-chars", action="store_true", dest="full_chars",
        help=f"改用全部可打印 ASCII（{len(ASCII_CHARS)} 个字符）：分级更细，"
             "但近半数字形墨量几乎相同，纹理也更杂",
    )
    font.add_argument(
        "--list-ramp", action="store_true",
        help="打印实测的字符分级表（字符 / 归一化亮度 / 墨量）后退出",
    )

    look = parser.add_argument_group("亮度与颜色")
    look.add_argument(
        "--metric", choices=sorted(BRIGHTNESS_METRICS), default=_DEFAULTS.metric,
        help="亮度度量方式（默认 luminance）",
    )
    look.add_argument(
        "--gamma", type=float, default=_DEFAULTS.gamma,
        help="查表前的亮度伽马校正，<1 提亮、>1 压暗（默认 1.0）",
    )
    look.add_argument(
        "--image-saturation", type=float, default=_DEFAULTS.image_saturation, metavar="K",
        help="图像饱和度倍数，1.0 原样、0 变灰、>1 更艳（默认 1.0）",
    )
    look.add_argument(
        "--color-mode", choices=sorted(COLOR_MODES), default=_DEFAULTS.color_mode,
        help="match = 解出让区域平均色最接近原图的填充色（默认）；pure = 旧的纯度拉满",
    )
    look.add_argument(
        "--candidates", type=int, default=_DEFAULTS.candidates, metavar="N",
        help="查找候选数：1 = 只取「最小的平均亮度不小于目标」的那一个（默认）；"
             ">1 时从那一级起往上多考察 N 个更密的字符，取区域平均色最接近原图的",
    )
    look.add_argument(
        "--equalize", choices=sorted(EQUALIZE_MODES), default=_DEFAULTS.equalize,
        help="直方图均衡化：none(默认) / global(全局) / local(局部自适应)",
    )
    look.add_argument(
        "--equalize-window", type=int, default=_DEFAULTS.equalize_window, metavar="N",
        help="局部均衡化的窗口边长，单位是字符单元而非像素，越小越局部（默认 16）",
    )
    look.add_argument(
        "--equalize-clip", type=float, default=_DEFAULTS.equalize_clip, metavar="F",
        help="局部均衡化的对比度限幅，0 = 不限幅；平坦区域出噪点时调大到 2~4",
    )
    look.add_argument(
        "--highlight", type=float, default=_DEFAULTS.highlight, metavar="F",
        help="高亮保色强度 0~1：0(默认) = 逐通道截断，绝对色差最小但高亮会变灰；"
             "1 = 等比缩放，色相与彩度完整保留（高亮不变灰），绝对色差略升",
    )
    look.add_argument(
        "--invert", action="store_true",
        help="反相：暗处用密字符、亮处用空格，配合浅色背景使用",
    )
    look.add_argument(
        "--bg", default="black", metavar="COLOR",
        help='背景色：black / white / transparent / "#rrggbb"（默认 black）',
    )

    pure = parser.add_argument_group("仅 --color-mode pure 生效")
    pure.add_argument(
        "--glyph-purity", type=float, default=_DEFAULTS.glyph_purity, metavar="K",
        help="字符颜色饱和度，1.0 = 最高纯度（默认 1.0）",
    )
    pure.add_argument(
        "--chroma-threshold", type=float, default=_DEFAULTS.chroma_floor, metavar="C",
        help="色度低于此值视为灰色、输出白色字符（默认 0.04）",
    )

    extra = parser.add_argument_group("其它输出")
    extra.add_argument("--text", metavar="PATH", help="额外把纯字符文本写到该文件")
    extra.add_argument(
        "--ansi", action="store_true",
        help="把彩色字符阵列直接打印到终端（24 位真彩色）",
    )
    extra.add_argument("-q", "--quiet", action="store_true", help="不打印统计信息")
    extra.add_argument(
        "--list-fonts", action="store_true", help="显示默认选用的字体后退出",
    )
    return parser


def _print_ansi(result) -> None:
    """用 ANSI 真彩色把字符阵列打到终端。"""
    colors = result.colors
    out = []
    for y, line in enumerate(result.lines):
        row = colors[y]
        buf = []
        last = None
        for x, ch in enumerate(line):
            rgb = tuple(int(v) for v in row[x])
            if rgb != last:
                buf.append(f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m")
                last = rgb
            buf.append(ch)
        buf.append("\x1b[0m")
        out.append("".join(buf))
    sys.stdout.write("\n".join(out) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_fonts:
        print(find_default_font() or "（未找到可用的等宽字体）")
        return 0

    if args.block_chars:
        chars = BLOCK_CHARS
    elif args.full_chars:
        chars = ASCII_CHARS
    else:
        chars = args.chars

    if args.list_ramp:
        try:
            print(GlyphSet(chars, args.font, args.font_size).describe())
        except (ValueError, RuntimeError) as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 1
        return 0

    if not args.input:
        parser.error("需要给出输入图片路径（或用 gui 打开图形界面）")
        return 2

    opts = AsciiOptions(
        cols=args.cols,
        cell_width=args.cell_width,
        font_path=args.font,
        font_size=args.font_size,
        chars=chars,
        metric=args.metric,
        gamma=args.gamma,
        image_saturation=args.image_saturation,
        color_mode=args.color_mode,
        candidates=args.candidates,
        highlight=args.highlight,
        equalize=args.equalize,
        equalize_window=args.equalize_window,
        equalize_clip=args.equalize_clip,
        glyph_purity=args.glyph_purity,
        chroma_floor=args.chroma_threshold,
        invert=args.invert,
        background=args.bg,
    )

    try:
        with Image.open(args.input) as img:
            img.load()
            result = convert(img, opts)
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {args.input}", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    if args.output:
        out_path = Path(args.output)
        if out_path.parent and not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
        result.image.save(out_path)

    if args.text:
        Path(args.text).write_text(result.text, encoding="utf-8")

    if args.ansi:
        _print_ansi(result)
    elif not args.output and not args.text:
        # 什么都没指定时，退化成"打印字符文本"这一最直观的行为
        print(result.text)

    if not args.quiet:
        sw, sh = result.source_size
        ow, oh = result.size
        used = [ch for ch in result.ramp.chars if ch in set("".join(result.lines))]
        info = [
            f"源图 {sw}x{sh} -> 网格 {result.cols} 列 x {result.rows} 行",
            f"字框 {result.cell_w}x{result.cell_h}px  字号 {result.font_size}"
            f"  {Path(result.font_path).name}",
            f"输出 {ow}x{oh}",
            f"字符集 {len(result.ramp.chars)} 级，本图用到 {len(used)} 级："
            f"{''.join(used)!r}",
            f"最大墨量 {result.ramp.max_ink:.3f}"
            f"（黑底下输出亮度上限约 {result.ramp.max_ink:.0%}）",
        ]
        if result.added_edge_glyphs:
            info.append(
                f"边界替换：补进字符集 {result.added_edge_glyphs!r}，"
                f"强制替换 {result.forced_ratio:.1%} 的单元"
            )
        if args.candidates > 1:
            moved = int((result.grid_index != result.base_index).sum())
            total = result.grid_index.size
            info.append(
                f"候选 {args.candidates} 级：{moved}/{total} 个单元换用了更密的字符"
            )
        if args.output:
            info.append(f"已保存 {args.output}")
        if args.text:
            info.append(f"已保存文本 {args.text}")
        print("\n".join(info), file=sys.stderr)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
