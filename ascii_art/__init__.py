"""彩色图片 -> 彩色 ASCII 字符阵列。

快速开始::

    import ascii_art

    result = ascii_art.convert_file("photo.jpg", "ascii.png", ascii_art.AsciiOptions(cols=160))
    print(result.text)
    print(result.ramp.table())      # 实测出来的字符分级表
"""

from .core import (
    ASCII_CHARS,
    BLOCK_CHARS,
    BRIGHTNESS_METRICS,
    COLOR_MODES,
    DEFAULT_CHARS,
    EQUALIZE_MODES,
    AsciiOptions,
    AsciiResult,
    CharRamp,
    GlyphSet,
    as_tuple_rgb,
    convert,
    convert_file,
    find_default_font,
)

__version__ = "3.0.0"

__all__ = [
    "AsciiOptions",
    "AsciiResult",
    "CharRamp",
    "GlyphSet",
    "ASCII_CHARS",
    "BLOCK_CHARS",
    "BRIGHTNESS_METRICS",
    "COLOR_MODES",
    "DEFAULT_CHARS",
    "EQUALIZE_MODES",
    "as_tuple_rgb",
    "convert",
    "convert_file",
    "find_default_font",
    "__version__",
]
