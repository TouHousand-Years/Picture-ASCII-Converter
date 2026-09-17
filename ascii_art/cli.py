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
        description="Convert a color image to a color ASCII art grid and optionally export it as an image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m ascii_art photo.jpg -o ascii.png\n"
            "  python -m ascii_art photo.jpg -o ascii.png -c 200              # denser characters\n"
            "  python -m ascii_art photo.jpg -o ascii.png --cell-width 6       # control density by cell width\n"
            "  python -m ascii_art photo.jpg -o ascii.png --image-saturation 1.5\n"
            "  python -m ascii_art photo.jpg -o ascii.png --equalize local --equalize-window 8\n"
            "  python -m ascii_art photo.jpg -o ascii.png --highlight 1        # preserve highlight colors\n"
            "  python -m ascii_art photo.jpg -o ascii.png --chars \" \\u2591\\u2592\\u2593\\u2588\" # high-fidelity color matching\n"
            "  python -m ascii_art photo.jpg --list-ramp                      # view the measured character ramp\n"
            "  python -m ascii_art photo.jpg --ansi                           # view in color in the terminal\n"
            "\n"
            f"Default character set: {DEFAULT_CHARS!r}\n"
            "Levels are sorted at runtime using each character's measured ink coverage,\n"
            "which you can inspect with --list-ramp.\n"
        ),
    )
    parser.add_argument("input", nargs="?", help="Input image path (PNG/JPG/BMP/WebP/GIF, etc.)")
    parser.add_argument("-o", "--output", help="Output ASCII image path; if omitted, print character text only")

    density = parser.add_argument_group("Character density")
    density.add_argument(
        "-c", "--cols", "--density", type=int, default=_DEFAULTS.cols, metavar="N",
        help="Number of character columns; higher values add detail (default: 120)",
    )
    density.add_argument(
        "--cell-width", type=float, metavar="PX",
        help="Pixel width of each character cell; overrides --cols when provided",
    )

    font = parser.add_argument_group("Font and character set")
    font.add_argument("--font", help="Monospace font file path (.ttf/.ttc)")
    font.add_argument(
        "--font-size", type=int, default=_DEFAULTS.font_size, metavar="PX",
        help="Font size in the output image; also controls output dimensions (default: 20)",
    )
    charset = font.add_mutually_exclusive_group()
    charset.add_argument(
        "--chars", default=_DEFAULTS.chars, metavar="STR",
        help=_esc(f"Character set; order does not matter (default: {DEFAULT_CHARS!r})"),
    )
    charset.add_argument(
        "--block-chars", action="store_true", dest="block_chars",
        help=_esc(f"Use the block-character set {ascii(BLOCK_CHARS)}; ink reaches 1.0 for accurate brightness matching"),
    )
    charset.add_argument(
        "--full-chars", action="store_true", dest="full_chars",
        help=f"Use all printable ASCII characters ({len(ASCII_CHARS)} chars): finer levels, "
             "but many glyphs have nearly identical ink coverage and add visual noise",
    )
    font.add_argument(
        "--list-ramp", action="store_true",
        help="Print the measured character ramp (character / normalized brightness / ink) and exit",
    )

    look = parser.add_argument_group("Brightness and color")
    look.add_argument(
        "--metric", choices=sorted(BRIGHTNESS_METRICS), default=_DEFAULTS.metric,
        help="Brightness metric (default: luminance)",
    )
    look.add_argument(
        "--gamma", type=float, default=_DEFAULTS.gamma,
        help="Gamma correction before lookup; <1 brightens and >1 darkens (default: 1.0)",
    )
    look.add_argument(
        "--image-saturation", type=float, default=_DEFAULTS.image_saturation, metavar="K",
        help="Image saturation multiplier; 1.0 is unchanged, 0 is grayscale, >1 is more vivid (default: 1.0)",
    )
    look.add_argument(
        "--color-mode", choices=sorted(COLOR_MODES), default=_DEFAULTS.color_mode,
        help="match = fill color closest to the source area's average (default); pure = maximize color purity",
    )
    look.add_argument(
        "--candidates", type=int, default=_DEFAULTS.candidates, metavar="N",
        help="Candidate count: 1 = use only the least dense glyph at or above the target (default); "
             ">1 = inspect N denser glyphs and choose the one with the closest average color",
    )
    look.add_argument(
        "--equalize", choices=sorted(EQUALIZE_MODES), default=_DEFAULTS.equalize,
        help="Histogram equalization: none (default) / global / local (adaptive)",
    )
    look.add_argument(
        "--equalize-window", type=int, default=_DEFAULTS.equalize_window, metavar="N",
        help="Local equalization window size in character cells, not pixels; smaller is more local (default: 16)",
    )
    look.add_argument(
        "--equalize-clip", type=float, default=_DEFAULTS.equalize_clip, metavar="F",
        help="Local equalization clip limit; 0 = unlimited; increase to 2-4 if flat areas become noisy",
    )
    look.add_argument(
        "--highlight", type=float, default=_DEFAULTS.highlight, metavar="F",
        help="Highlight color preservation 0-1: 0 (default) clips each channel for minimum error; "
             "1 scales proportionally to preserve hue and saturation",
    )
    look.add_argument(
        "--invert", action="store_true",
        help="Invert: use dense glyphs for dark areas and spaces for bright areas; use with a light background",
    )
    look.add_argument(
        "--bg", default="black", metavar="COLOR",
        help='Background color: black / white / transparent / "#rrggbb" (default: black)',
    )

    pure = parser.add_argument_group("Only active with --color-mode pure")
    pure.add_argument(
        "--glyph-purity", type=float, default=_DEFAULTS.glyph_purity, metavar="K",
        help="Glyph color saturation; 1.0 = maximum purity (default: 1.0)",
    )
    pure.add_argument(
        "--chroma-threshold", type=float, default=_DEFAULTS.chroma_floor, metavar="C",
        help="Treat colors below this chroma as gray and output white glyphs (default: 0.04)",
    )

    extra = parser.add_argument_group("Other output")
    extra.add_argument("--text", metavar="PATH", help="Also write plain character text to this file")
    extra.add_argument(
        "--ansi", action="store_true",
        help="Print the color ASCII grid directly to the terminal (24-bit true color)",
    )
    extra.add_argument("-q", "--quiet", action="store_true", help="Do not print statistics")
    extra.add_argument(
        "--list-fonts", action="store_true", help="Show the default font selection and exit",
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
        print(find_default_font() or "No usable monospace font found")
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
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    if not args.input:
        parser.error("an input image path is required (or use --gui to open the graphical interface)")
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
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
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
            f"Source {sw}x{sh} -> grid {result.cols} cols x {result.rows} rows",
            f"Cell {result.cell_w}x{result.cell_h}px  font size {result.font_size}"
            f"  {Path(result.font_path).name}",
            f"Output {ow}x{oh}",
            f"Character set: {len(result.ramp.chars)} levels, {len(used)} used:"
            f"{''.join(used)!r}",
            f"Maximum ink {result.ramp.max_ink:.3f}"
            f" (brightness ceiling on black: about {result.ramp.max_ink:.0%})",
        ]
        if result.added_edge_glyphs:
            info.append(
                f"Edge replacement: added {result.added_edge_glyphs!r} to the character set; "
                f"forced replacement in {result.forced_ratio:.1%} of cells"
            )
        if args.candidates > 1:
            moved = int((result.grid_index != result.base_index).sum())
            total = result.grid_index.size
            info.append(
                f"Candidates {args.candidates}: {moved}/{total} cells use a denser glyph"
            )
        if args.output:
            info.append(f"Saved image: {args.output}")
        if args.text:
            info.append(f"Saved text: {args.text}")
        print("\n".join(info), file=sys.stderr)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
