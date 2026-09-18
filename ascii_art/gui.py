"""图形界面：打开图片 -> 拖动字符密度滑块实时预览 -> 导出图片。

渲染放在后台线程里做，主线程只负责画预览，所以拖动滑块时界面不会卡住；
连续拖动只会保留最后一次请求，中间结果直接丢弃。
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .core import (
    ASCII_CHARS,
    BLOCK_CHARS,
    BRIGHTNESS_METRICS,
    COLOR_MODES,
    DEFAULT_CHARS,
    EQUALIZE_MODES,
    AsciiOptions,
    AsciiResult,
    GlyphSet,
    as_tuple_rgb,
    convert,
    find_default_font,
)

#: 背景下拉项 -> AsciiOptions.background 取值
BACKGROUNDS: dict[str, object] = {
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Transparent": None,
}

#: 密度快捷按钮
DENSITY_PRESETS = (60, 100, 160, 240)

# 英文界面比中文文案更占横向空间。右栏给控件留出舒服的基础宽度，
# 文字的实际换行宽度仍会在窗口/面板尺寸变化时动态更新。
PANEL_WIDTH = 360
MIN_WRAP_LENGTH = 120

WATERMARK_LINES = (
    "Powered by Picture ASCII Converter from THY-Workshop",
    "https://github.com/TouHousand-Years/Picture-ASCII-Converter",
)
WATERMARK_TEXT = "\n".join(WATERMARK_LINES)
WATERMARK_POSITIONS = ("Top Left", "Top Right", "Bottom Left", "Bottom Right")

#: 界面控件的初值一律引用核心默认值，免得两处各写一份、改一处忘一处。
_DEFAULTS = AsciiOptions()


def _checkerboard(size: tuple[int, int], cell: int = 10) -> Image.Image:
    """透明预览用的棋盘格底图。"""
    w, h = size
    yy, xx = np.indices((h, w))
    mask = ((xx // cell) + (yy // cell)) % 2 == 0
    arr = np.empty((h, w, 3), np.uint8)
    arr[mask] = (66, 66, 66)
    arr[~mask] = (92, 92, 92)
    return Image.fromarray(arr, "RGB")


def _add_watermark(
    result: AsciiResult,
    position: str,
    background: object,
) -> AsciiResult:
    """用水印文字强制替换角落里的字符单元，并同步更新图像与纯文本。"""
    if position not in WATERMARK_POSITIONS:
        raise ValueError(f"Unknown watermark position: {position}")

    if result.cols < 1 or result.rows < 1:
        return result

    spacer = (lambda line: f"{line} ") if position.endswith("Left") else (
        lambda line: f" {line}"
    )
    visible_lines = tuple(
        spacer(line) for line in WATERMARK_LINES[: min(len(WATERMARK_LINES), result.rows)]
    )
    first_row = 0 if position.startswith("Top") else result.rows - len(visible_lines)
    bg = as_tuple_rgb(background)

    glyphs = GlyphSet(
        "".join(dict.fromkeys("".join(WATERMARK_LINES))),
        font_path=result.font_path,
        font_size=result.font_size,
    )
    masks = {
        ch: Image.fromarray((glyphs.atlas[index] * 255).astype(np.uint8), "L")
        for index, ch in enumerate(glyphs.chars)
    }
    image = result.image.copy()
    lines = [list(line.ljust(result.cols)[:result.cols]) for line in result.lines]

    for line_offset, watermark_line in enumerate(visible_lines):
        text = watermark_line[:result.cols]
        first_col = 0 if position.endswith("Left") else result.cols - len(text)
        row = first_row + line_offset
        for offset, ch in enumerate(text):
            col = first_col + offset
            lines[row][col] = ch
            text_color = tuple(int(channel) for channel in result.colors[row, col])
            x0, y0 = col * result.cell_w, row * result.cell_h
            box = (x0, y0, x0 + result.cell_w, y0 + result.cell_h)
            if bg is None:
                cell = Image.new("RGBA", (result.cell_w, result.cell_h), (*text_color, 0))
                cell.putalpha(masks[ch])
                image.paste(cell, (x0, y0))
            else:
                image.paste(bg, box)
                ink = Image.new(image.mode, (result.cell_w, result.cell_h), text_color)
                image.paste(ink, (x0, y0), masks[ch])

    result.lines = ["".join(line) for line in lines]
    result.image = image
    return result


class ScrollableFrame(ttk.Frame):
    """可滚动的容器：内容比窗口高时用滚轮滚动，宽度始终随容器自适应。"""

    def __init__(self, parent: tk.Widget, width: int = 300) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(
            self, borderwidth=0, highlightthickness=0, width=width, takefocus=0
        )
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.body = ttk.Frame(self.canvas, padding=(12, 0, 4, 0))
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", self._on_content_resize)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # 滚轮是全局事件，按指针位置判断该不该滚，免得影响左边的预览区
        self.bind_all("<MouseWheel>", self._on_wheel)              # Windows / macOS
        self.bind_all("<Button-4>", lambda e: self._wheel(-1, e))  # Linux 上滚
        self.bind_all("<Button-5>", lambda e: self._wheel(1, e))   # Linux 下滚

    def _on_content_resize(self, _event: tk.Event | None = None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_resize(self, event: tk.Event) -> None:
        # 让内层框架跟容器一样宽，参数栏才会横向铺满
        self.canvas.itemconfigure(self._window, width=event.width)

    def _pointer_inside(self, event: tk.Event) -> bool:
        x0, y0 = self.canvas.winfo_rootx(), self.canvas.winfo_rooty()
        return (
            x0 <= event.x_root <= x0 + self.canvas.winfo_width()
            and y0 <= event.y_root <= y0 + self.canvas.winfo_height()
        )

    def _wheel(self, steps: int, event: tk.Event | None = None) -> None:
        if event is not None and not self._pointer_inside(event):
            return
        self.canvas.yview_scroll(steps, "units")

    def _on_wheel(self, event: tk.Event) -> None:
        self._wheel(-1 if event.delta > 0 else 1, event)


class AsciiArtApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.source: Image.Image | None = None
        self.source_path: Path | None = None
        self.result: AsciiResult | None = None
        self.last_elapsed = 0.0

        self._render_job: str | None = None
        self._fit_job: str | None = None
        self._token = 0
        self._queue: queue.Queue = queue.Queue()
        self._render_lock = threading.Lock()
        self._canvas_image: ImageTk.PhotoImage | None = None
        self._checker_cache: tuple[tuple[int, int], Image.Image] | None = None
        self._wrapping_labels: list[tuple[ttk.Label, tk.Widget, int]] = []

        root.title("Color ASCII Art Converter")
        root.geometry("1360x880")
        root.minsize(1000, 520)

        self._build_ui()
        self._sync_metric_hint()
        self._sync_mode_hint()
        self._sync_equalize_hint()
        self._on_equalize_change(None)
        root.after(40, self._poll)
        self._draw_preview()

    # ----------------------------------------------------------------- UI --
    def _build_ui(self) -> None:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")

        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(side="top", fill="x")
        ttk.Button(toolbar, text="Open Image…", command=self.open_image).pack(side="left")
        self.export_btn = ttk.Button(
            toolbar, text="Export Image…", command=self.export_image, state="disabled"
        )
        self.export_btn.pack(side="left", padx=(6, 0))
        self.text_btn = ttk.Button(
            toolbar, text="Export Text…", command=self.export_text, state="disabled"
        )
        self.text_btn.pack(side="left", padx=(6, 0))
        self.file_label = ttk.Label(toolbar, text="No image opened", foreground="#666")
        self.file_label.pack(side="left", padx=(14, 0))

        status = ttk.Frame(self.root, padding=(12, 6))
        status.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="Open an image to begin.")
        ttk.Label(status, textvariable=self.status_var, foreground="#333").pack(side="left")
        self.font_hint_var = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.font_hint_var, foreground="#888").pack(side="right")

        body = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        body.pack(side="top", fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(
            left, background="#1b1b1b", highlightthickness=1,
            highlightbackground="#3a3a3a",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # 参数栏比窗口高，套一层可滚动容器
        self.panel = ScrollableFrame(body, width=PANEL_WIDTH)
        self.panel.pack(side="right", fill="y")
        self._build_panel(self.panel.body)
        self.panel.body.bind("<Configure>", self._refresh_wrapping, add="+")
        self.panel.body.update_idletasks()
        self._refresh_wrapping()
        self.panel._on_content_resize()

    def _build_panel(self, panel: ttk.Frame) -> None:
        row = 0

        # -- 字符密度（主角）----------------------------------------------- #
        box = ttk.LabelFrame(panel, text="Character Density", padding=10)
        box.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        row += 1

        self.cols_var = tk.IntVar(value=_DEFAULTS.cols)
        self.cols_scale, self.cols_label = self._add_slider(
            box, "Columns — higher means more detail", 20, 400, _DEFAULTS.cols,
            lambda v: f"{int(round(v))} cols", self._on_cols_change,
        )

        presets = ttk.Frame(box)
        presets.pack(fill="x")
        self._add_wrapped_label(presets, "Presets:").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 3)
        )
        for column, value in enumerate(DENSITY_PRESETS):
            ttk.Button(
                presets, text=str(value),
                command=lambda v=value: self.set_cols(v),
            ).grid(row=1, column=column, sticky="ew", padx=(0 if column == 0 else 3, 0))
            presets.columnconfigure(column, weight=1, uniform="density-preset")

        # -- 外观 ---------------------------------------------------------- #
        box = ttk.LabelFrame(panel, text="Appearance", padding=10)
        box.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        row += 1

        self.font_var = tk.IntVar(value=_DEFAULTS.font_size)
        self.font_scale, self.font_label = self._add_slider(
            box, "Font size — controls output image size", 8, 48, _DEFAULTS.font_size,
            lambda v: f"{int(round(v))} px", self._on_font_change,
        )

        self._add_wrapped_label(
            box, "Character set (order does not matter; levels use measured ink)"
        ).pack(fill="x", anchor="w")
        self.chars_var = tk.StringVar(value=_DEFAULTS.chars)
        ttk.Entry(box, textvariable=self.chars_var).pack(fill="x", pady=(2, 4))
        self.chars_var.trace_add("write", lambda *_: self.schedule_render())

        char_btns = ttk.Frame(box)
        char_btns.pack(fill="x", pady=(0, 6))
        ttk.Button(
            char_btns, text="Measured Ramp…", command=self.show_ramp_table
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Button(
            char_btns, text="Block Characters",
            command=lambda: self.chars_var.set(BLOCK_CHARS),
        ).grid(row=1, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(
            char_btns, text="Full ASCII",
            command=lambda: self.chars_var.set(ASCII_CHARS),
        ).grid(row=1, column=1, sticky="ew", padx=(2, 0))
        char_btns.columnconfigure(0, weight=1, uniform="character-action")
        char_btns.columnconfigure(1, weight=1, uniform="character-action")

        ttk.Label(box, text="Background").pack(anchor="w")
        self.bg_var = tk.StringVar(value="Black")
        bg_combo = ttk.Combobox(
            box, textvariable=self.bg_var, values=list(BACKGROUNDS), state="readonly"
        )
        bg_combo.pack(fill="x")
        bg_combo.bind("<<ComboboxSelected>>", lambda _e: self.schedule_render())

        # -- 亮度与颜色 ---------------------------------------------------- #
        box = ttk.LabelFrame(panel, text="Brightness & Color", padding=10)
        box.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        row += 1

        ttk.Label(box, text="Brightness metric").pack(anchor="w")
        self.metric_var = tk.StringVar(value=_DEFAULTS.metric)
        metric_combo = ttk.Combobox(
            box, textvariable=self.metric_var,
            values=sorted(BRIGHTNESS_METRICS), state="readonly",
        )
        metric_combo.pack(fill="x", pady=(2, 2))
        metric_combo.bind("<<ComboboxSelected>>", self._on_metric_change)
        self.metric_hint = self._add_wrapped_label(box, "", foreground="#666")
        self.metric_hint.pack(fill="x", anchor="w", pady=(0, 6))

        self.candidates_var = tk.IntVar(value=_DEFAULTS.candidates)
        self.candidates_scale, self.candidates_label = self._add_slider(
            box, "Candidate count (1 = use only the ceiling)", 1, 10, _DEFAULTS.candidates,
            lambda v: f"{int(round(v))} candidates", self._on_candidates_change,
        )
        self.candidates_hint = self._add_wrapped_label(
            box,
            text="Starting at the ceiling level, inspect denser characters and choose the one "
                 "whose average color is closest to the source; the error can only decrease.",
            foreground="#666",
        )
        self.candidates_hint.pack(fill="x", anchor="w", pady=(0, 6))

        self.highlight_var = tk.DoubleVar(value=_DEFAULTS.highlight)
        self.highlight_scale, self.highlight_label = self._add_slider(
            box, "Highlight color preservation (1 = keep highlights colored)", 0.0, 1.0, _DEFAULTS.highlight,
            lambda v: f"{v:.2f}", self._on_highlight_change,
        )

        self.image_sat_var = tk.DoubleVar(value=_DEFAULTS.image_saturation)
        self.image_sat_scale, self.image_sat_label = self._add_slider(
            box, "Image saturation (0 = grayscale, >1 = more vivid)", 0.0, 2.0, _DEFAULTS.image_saturation,
            lambda v: f"{v:.2f}", self._on_image_sat_change,
        )

        ttk.Label(box, text="Histogram equalization").pack(anchor="w")
        self.equalize_var = tk.StringVar(value=_DEFAULTS.equalize)
        eq_combo = ttk.Combobox(
            box, textvariable=self.equalize_var,
            values=sorted(EQUALIZE_MODES), state="readonly",
        )
        eq_combo.pack(fill="x", pady=(2, 2))
        eq_combo.bind("<<ComboboxSelected>>", self._on_equalize_change)
        self.equalize_hint = self._add_wrapped_label(box, "", foreground="#666")
        self.equalize_hint.pack(fill="x", anchor="w", pady=(0, 6))

        self.eq_window_var = tk.IntVar(value=_DEFAULTS.equalize_window)
        self.eq_window_scale, self.eq_window_label = self._add_slider(
            box, "Local window (character cells; smaller = more local)", 2, 64, _DEFAULTS.equalize_window,
            lambda v: f"{int(round(v))} cells", self._on_eq_window_change,
        )

        self.eq_clip_var = tk.DoubleVar(value=_DEFAULTS.equalize_clip)
        self.eq_clip_scale, self.eq_clip_label = self._add_slider(
            box, "Local clip limit (0 = unlimited / pure AHE)", 0.0, 8.0, _DEFAULTS.equalize_clip,
            lambda v: "Unlimited" if v < 0.05 else f"{v:.1f}x", self._on_eq_clip_change,
        )

        self.gamma_var = tk.DoubleVar(value=_DEFAULTS.gamma)
        self.gamma_scale, self.gamma_label = self._add_slider(
            box, "Brightness gamma (<1 = brighter, >1 = darker)", 0.3, 3.0, _DEFAULTS.gamma,
            lambda v: f"{v:.2f}", self._on_gamma_change,
        )

        ttk.Label(box, text="Color mode").pack(anchor="w")
        self.color_mode_var = tk.StringVar(value=_DEFAULTS.color_mode)
        mode_combo = ttk.Combobox(
            box, textvariable=self.color_mode_var,
            values=sorted(COLOR_MODES), state="readonly",
        )
        mode_combo.pack(fill="x", pady=(2, 2))
        mode_combo.bind("<<ComboboxSelected>>", self._on_color_mode_change)
        self.mode_hint = self._add_wrapped_label(box, "", foreground="#666")
        self.mode_hint.pack(fill="x", anchor="w", pady=(0, 6))

        self.sat_var = tk.DoubleVar(value=_DEFAULTS.glyph_purity)
        self.sat_scale, self.sat_label = self._add_slider(
            box, "Glyph color purity (only active in 'pure' mode)", 0.0, 1.0, _DEFAULTS.glyph_purity,
            lambda v: f"{v:.2f}", self._on_sat_change,
        )

        self.invert_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            box, text="Invert (for light backgrounds)", variable=self.invert_var,
            command=self.schedule_render,
        ).pack(anchor="w")

        # -- 水印 ---------------------------------------------------------- #
        box = ttk.LabelFrame(panel, text="Watermark", padding=10)
        box.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        row += 1

        self.watermark_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            box,
            text="Show attribution watermark",
            variable=self.watermark_var,
            command=self._on_watermark_change,
        ).pack(anchor="w", pady=(0, 5))
        self._add_wrapped_label(box, "Corner").pack(fill="x", anchor="w")
        self.watermark_position_var = tk.StringVar(value="Bottom Right")
        self.watermark_position_combo = ttk.Combobox(
            box,
            textvariable=self.watermark_position_var,
            values=WATERMARK_POSITIONS,
            state="disabled",
        )
        self.watermark_position_combo.pack(fill="x", pady=(2, 5))
        self.watermark_position_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self.schedule_render()
        )
        self._add_wrapped_label(
            box, WATERMARK_TEXT, foreground="#666"
        ).pack(fill="x", anchor="w")

        # -- 输出信息 ------------------------------------------------------ #
        box = ttk.LabelFrame(panel, text="Output Info", padding=10)
        box.grid(row=row, column=0, sticky="ew")
        self.info_var = tk.StringVar(value="—")
        self.info_label = self._add_wrapped_label(
            box, textvariable=self.info_var
        )
        self.info_label.pack(fill="x", anchor="w")

        for child in panel.winfo_children():
            child.grid_configure(sticky="ew")
        panel.columnconfigure(0, weight=1)

    def _add_wrapped_label(
        self,
        parent: tk.Widget,
        text: str | None = None,
        *,
        inset: int = 20,
        **kwargs,
    ) -> ttk.Label:
        """创建会跟随父容器宽度自动换行的标签。"""
        if text is not None:
            kwargs["text"] = text
        label = ttk.Label(
            parent,
            justify="left",
            anchor="w",
            wraplength=max(MIN_WRAP_LENGTH, PANEL_WIDTH - inset),
            **kwargs,
        )
        self._wrapping_labels.append((label, parent, inset))
        return label

    def _refresh_wrapping(self, _event: tk.Event | None = None) -> None:
        """按标签所在容器的可用宽度刷新换行，避免英文文案撑破右栏。"""
        for label, parent, inset in self._wrapping_labels:
            if not label.winfo_exists():
                continue
            available = parent.winfo_width() - inset
            if available > 1:
                label.configure(wraplength=max(MIN_WRAP_LENGTH, available))

    def _add_slider(
        self,
        parent: ttk.Frame,
        title: str,
        low: float,
        high: float,
        value: float,
        formatter: Callable[[float], str],
        on_change: Callable[[float, Callable[[float], str], ttk.Label], None],
    ) -> tuple[ttk.Scale, ttk.Label]:
        """加一行「标题 + 滑块 + 数值」。

        先建好数值标签再 ``set()``，否则 set 触发的回调会引用到还不存在的控件。
        """
        self._add_wrapped_label(parent, title).pack(fill="x", anchor="w")
        scale = ttk.Scale(parent, from_=low, to=high, orient="horizontal")
        scale.pack(fill="x")
        label = ttk.Label(parent, text=formatter(value), font=("Segoe UI", 9, "bold"))
        label.pack(anchor="w", pady=(2, 6))
        scale.configure(command=lambda raw: on_change(float(raw), formatter, label))
        scale.set(value)
        return scale, label

    # ------------------------------------------------------------- 回调 --
    def _on_cols_change(self, value, formatter, label) -> None:
        self.cols_var.set(int(round(value)))
        label.configure(text=formatter(value))
        self.schedule_render()

    def _on_font_change(self, value, formatter, label) -> None:
        self.font_var.set(int(round(value)))
        label.configure(text=formatter(value))
        self.schedule_render()

    def _on_sat_change(self, value, formatter, label) -> None:
        self.sat_var.set(value)
        label.configure(text=formatter(value))
        self.schedule_render()

    def _on_watermark_change(self) -> None:
        self.watermark_position_combo.configure(
            state="readonly" if self.watermark_var.get() else "disabled"
        )
        self.schedule_render()

    def _on_image_sat_change(self, value, formatter, label) -> None:
        self.image_sat_var.set(value)
        label.configure(text=formatter(value))
        self.schedule_render()

    def _on_gamma_change(self, value, formatter, label) -> None:
        self.gamma_var.set(value)
        label.configure(text=formatter(value))
        self.schedule_render()

    def _on_metric_change(self, _event: tk.Event) -> None:
        self._sync_metric_hint()
        self.schedule_render()

    def _on_candidates_change(self, value, formatter, label) -> None:
        self.candidates_var.set(int(round(value)))
        label.configure(text=formatter(value))
        self.schedule_render()

    def _on_highlight_change(self, value, formatter, label) -> None:
        self.highlight_var.set(value)
        label.configure(text=formatter(value))
        self.schedule_render()

    def _on_equalize_change(self, _event: tk.Event) -> None:
        self._sync_equalize_hint()
        local = self.equalize_var.get() == "local"
        state = "normal" if local else "disabled"
        for widget in (self.eq_window_scale, self.eq_clip_scale):
            widget.configure(state=state)
        self.schedule_render()

    def _on_eq_window_change(self, value, formatter, label) -> None:
        self.eq_window_var.set(int(round(value)))
        label.configure(text=formatter(value))
        self.schedule_render()

    def _on_eq_clip_change(self, value, formatter, label) -> None:
        self.eq_clip_var.set(value)
        label.configure(text=formatter(value))
        self.schedule_render()

    def _sync_equalize_hint(self) -> None:
        self.equalize_hint.configure(
            text=EQUALIZE_MODES.get(self.equalize_var.get(), "")
        )

    def _on_color_mode_change(self, _event: tk.Event) -> None:
        self._sync_mode_hint()
        self.schedule_render()

    def _sync_metric_hint(self) -> None:
        self.metric_hint.configure(
            text=BRIGHTNESS_METRICS.get(self.metric_var.get(), "")
        )

    def _sync_mode_hint(self) -> None:
        self.mode_hint.configure(text=COLOR_MODES.get(self.color_mode_var.get(), ""))

    def show_ramp_table(self) -> None:
        """弹窗展示本次参数下实测出来的字符分级表。"""
        try:
            glyphs = GlyphSet(
                self.chars_var.get(), font_path=None, font_size=self.font_var.get()
            )
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Invalid Character Set", str(exc))
            return
        win = tk.Toplevel(self.root)
        win.title("Measured Character Ramp")
        win.geometry("560x520")
        text = tk.Text(win, wrap="none", font=("Consolas", 10))
        text.pack(fill="both", expand=True, padx=8, pady=8)
        text.insert("1.0", glyphs.describe())
        text.configure(state="disabled")

    def set_cols(self, value: int) -> None:
        self.cols_scale.set(value)          # 会触发 command，值和预览一起更新

    # ------------------------------------------------------------- 文件 --
    def open_image(self, path: str | None = None) -> None:
        if not path:
            path = filedialog.askopenfilename(
                title="Choose Image",
                filetypes=[
                    ("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.gif *.tif *.tiff"),
                    ("All files", "*.*"),
                ],
            )
        if not path:
            return
        try:
            with Image.open(path) as img:
                img.load()
                self.source = img.convert("RGBA")
        except OSError as exc:
            messagebox.showerror("Open Failed", str(exc))
            return
        self.source_path = Path(path)
        w, h = self.source.size
        self.file_label.configure(
            text=f"{self.source_path.name}   {w}x{h}", foreground="#222"
        )
        self.export_btn.configure(state="normal")
        self.text_btn.configure(state="normal")
        self.schedule_render(delay=0)

    def export_image(self) -> None:
        if self.result is None:
            return
        stem = self.source_path.stem if self.source_path else "ascii"
        path = filedialog.asksaveasfilename(
            title="Export ASCII Image", defaultextension=".png",
            initialfile=f"{stem}_ascii.png",
            filetypes=[("PNG Image", "*.png"), ("WebP Image", "*.webp"),
                       ("JPEG Image", "*.jpg"), ("BMP Image", "*.bmp")],
        )
        if not path:
            return
        image = self.result.image
        if Path(path).suffix.lower() in (".jpg", ".jpeg") and image.mode == "RGBA":
            # JPEG 不支持透明，压到白底上
            flat = Image.new("RGB", image.size, (255, 255, 255))
            flat.paste(image, mask=image.getchannel("A"))
            image = flat
        try:
            image.save(path)
        except OSError as exc:
            messagebox.showerror("Save Failed", str(exc))
            return
        self.status_var.set(f"Exported image: {path}")

    def export_text(self) -> None:
        if self.result is None:
            return
        stem = self.source_path.stem if self.source_path else "ascii"
        path = filedialog.asksaveasfilename(
            title="Export Character Text", defaultextension=".txt",
            initialfile=f"{stem}_ascii.txt", filetypes=[("Text File", "*.txt")],
        )
        if not path:
            return
        Path(path).write_text(self.result.text, encoding="utf-8")
        self.status_var.set(f"Exported text: {path}")

    # ------------------------------------------------------------- 渲染 --
    def _collect_options(self) -> AsciiOptions:
        return AsciiOptions(
            cols=self.cols_var.get(),
            font_size=self.font_var.get(),
            chars=self.chars_var.get() or DEFAULT_CHARS,
            metric=self.metric_var.get(),
            gamma=self.gamma_var.get(),
            image_saturation=self.image_sat_var.get(),
            highlight=self.highlight_var.get(),
            equalize=self.equalize_var.get(),
            equalize_window=self.eq_window_var.get(),
            equalize_clip=self.eq_clip_var.get(),
            color_mode=self.color_mode_var.get(),
            candidates=self.candidates_var.get(),
            glyph_purity=self.sat_var.get(),
            invert=self.invert_var.get(),
            background=BACKGROUNDS.get(self.bg_var.get(), (0, 0, 0)),
        )

    def schedule_render(self, delay: int = 140) -> None:
        if self.source is None:
            return
        if self._render_job is not None:
            self.root.after_cancel(self._render_job)
        self._render_job = self.root.after(delay, self._start_render)

    def _start_render(self) -> None:
        self._render_job = None
        if self.source is None:
            return
        self._token += 1
        self.status_var.set("Rendering…")
        threading.Thread(
            target=self._worker,
            args=(
                self._token,
                self.source,
                self._collect_options(),
                self.watermark_var.get(),
                self.watermark_position_var.get(),
            ),
            daemon=True,
        ).start()

    def _worker(
        self,
        token: int,
        source: Image.Image,
        opts: AsciiOptions,
        watermark: bool,
        watermark_position: str,
    ) -> None:
        started = time.perf_counter()
        try:
            with self._render_lock:      # 串行化：快速拖动会叠出多个请求
                result = convert(source, opts)
                if watermark:
                    result = _add_watermark(result, watermark_position, opts.background)
            self._queue.put((token, result, time.perf_counter() - started, None))
        except Exception as exc:  # noqa: BLE001 - 线程里必须兜住所有异常
            self._queue.put((token, None, 0.0, exc))

    def _poll(self) -> None:
        try:
            while True:
                token, result, elapsed, error = self._queue.get_nowait()
                if token != self._token:
                    continue                       # 过期结果，丢掉
                if error is not None:
                    self.status_var.set(f"Render failed: {error}")
                    traceback.print_exception(type(error), error, error.__traceback__)
                    continue
                self.result = result
                self.last_elapsed = elapsed
                self._update_info()
                self._draw_preview()
        except queue.Empty:
            pass
        self.root.after(40, self._poll)

    def _update_info(self) -> None:
        res = self.result
        if res is None:
            return
        sw, sh = res.source_size
        ow, oh = res.size
        font = Path(res.font_path).name if res.font_path else "?"
        used = [ch for ch in res.ramp.chars if ch in set("".join(res.lines))]
        self.info_var.set(
            f"Grid: {res.cols} cols x {res.rows} rows\n"
            f"Cell: {res.cell_w} x {res.cell_h} px\n"
            f"Output: {ow} x {oh} px\n"
            f"Source: {sw} x {sh} px\n"
            f"Font: {font} @ {res.font_size}\n"
            f"Levels: {len(res.ramp.chars)} total, {len(used)} used\n"
            f"Maximum ink: {res.ramp.max_ink:.3f}"
            f" (brightness ceiling {res.ramp.max_ink:.0%})\n"
            f"Candidates: {self.candidates_var.get()}"
            f"; {int((res.grid_index != res.base_index).sum())} cells use a denser character\n"
            f"Elapsed: {self.last_elapsed * 1000:.0f} ms"
        )
        self.font_hint_var.set(f"{font} @ {res.font_size}")
        self.status_var.set(
            f"{res.cols} x {res.rows} chars · Output {ow}x{oh} · "
            f"{self.last_elapsed * 1000:.0f} ms"
        )

    # ------------------------------------------------------------- 预览 --
    def _on_canvas_resize(self, _event: tk.Event) -> None:
        if self._fit_job is not None:
            self.root.after_cancel(self._fit_job)
        self._fit_job = self.root.after(80, self._draw_preview)

    def _draw_preview(self) -> None:
        self._fit_job = None
        self.canvas.delete("all")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        if self.result is None:
            self.canvas.create_text(
                cw // 2, ch // 2,
                text="Click Open Image… in the top-left\nthen adjust the Character Density slider on the right",
                fill="#8a8a8a", justify="center", font=("Segoe UI", 13),
            )
            self._canvas_image = None
            return

        if self.result.image.mode == "RGBA":
            base = self._checker((cw, ch))
            scaled = self._fit(self.result.image, (cw, ch))
            base.paste(scaled, ((cw - scaled.width) // 2, (ch - scaled.height) // 2), scaled)
        else:
            base = Image.new("RGB", (cw, ch), (27, 27, 27))
            scaled = self._fit(self.result.image, (cw, ch))
            base.paste(scaled, ((cw - scaled.width) // 2, (ch - scaled.height) // 2))

        self._canvas_image = ImageTk.PhotoImage(base)
        self.canvas.create_image(cw // 2, ch // 2, image=self._canvas_image)

    def _checker(self, size: tuple[int, int]) -> Image.Image:
        if self._checker_cache is None or self._checker_cache[0] != size:
            self._checker_cache = (size, _checkerboard(size))
        return self._checker_cache[1].copy()

    @staticmethod
    def _fit(image: Image.Image, box: tuple[int, int]) -> Image.Image:
        """等比缩放塞进 box；放大时用 NEAREST，方便看清字符。"""
        bw, bh = max(1, box[0] - 8), max(1, box[1] - 8)
        scale = min(bw / image.width, bh / image.height)
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        resample = Image.Resampling.NEAREST if scale > 1 else Image.Resampling.LANCZOS
        return image.resize(size, resample)


def main(argv: list[str] | None = None) -> int:
    av = list(argv or [])
    if not find_default_font():
        # 早失败，别让用户点了按钮才报错
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "No Monospace Font",
            "No usable monospace font was found. Install Consolas / DejaVu Sans Mono, "
            "or specify a font file with --font on the command line.",
        )
        root.destroy()
        return 1

    root = tk.Tk()
    app = AsciiArtApp(root)
    if av and Path(av[0]).is_file():
        root.after(120, lambda: app.open_image(av[0]))
    root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
