"""``python -m ascii_art`` 入口。

* ``python -m ascii_art``            -> 打开图形界面
* ``python -m ascii_art --gui``      -> 同上
* ``python -m ascii_art img.jpg ...`` -> 命令行转换
"""

from __future__ import annotations

import sys

GUI_FLAGS = {"gui", "--gui", "-g"}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in GUI_FLAGS:
        from .gui import main as gui_main

        return gui_main(argv[1:])

    from .cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
