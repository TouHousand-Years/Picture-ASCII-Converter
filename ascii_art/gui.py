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
    convert,
    find_default_font,
)

#: 背景下拉项 -> AsciiOptions.background 取值
BACKGROUNDS: dict[str, object] = {
    "黑色": (0, 0, 0),
    "白色": (255, 255, 255),
    "透明": None,
}

#: 密度快捷按钮
DENSITY_PRESETS = (60, 100, 160, 240)

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

        root.title("彩色 ASCII 艺术转换器")
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
        ttk.Button(toolbar, text="打开图片…", command=self.open_image).pack(side="left")
        self.export_btn = ttk.Button(
            toolbar, text="导出图片…", command=self.export_image, state="disabled"
        )
        self.export_btn.pack(side="left", padx=(6, 0))
        self.text_btn = ttk.Button(
            toolbar, text="导出字符文本…", command=self.export_text, state="disabled"
        )
        self.text_btn.pack(side="left", padx=(6, 0))
        self.file_label = ttk.Label(toolbar, text="尚未打开图片", foreground="#666")
        self.file_label.pack(side="left", padx=(14, 0))

        status = ttk.Frame(self.root, padding=(12, 6))
        status.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="打开一张图片开始。")
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
        self.panel = ScrollableFrame(body, width=300)
        self.panel.pack(side="right", fill="y")
        self._build_panel(self.panel.body)
        self.panel.body.update_idletasks()
        self.panel._on_content_resize()

    def _build_panel(self, panel: ttk.Frame) -> None:
        row = 0

        # -- 字符密度（主角）----------------------------------------------- #
        box = ttk.LabelFrame(panel, text="字符密度", padding=10)
        box.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        row += 1

        self.cols_var = tk.IntVar(value=_DEFAULTS.cols)
        self.cols_scale, self.cols_label = self._add_slider(
            box, "列数 —— 越大字符越多、越细腻", 20, 400, _DEFAULTS.cols,
            lambda v: f"{int(round(v))} 列", self._on_cols_change,
        )

        presets = ttk.Frame(box)
        presets.pack(fill="x")
        ttk.Label(presets, text="快捷：").pack(side="left")
        for value in DENSITY_PRESETS:
            ttk.Button(
                presets, text=str(value), width=4,
                command=lambda v=value: self.set_cols(v),
            ).pack(side="left", padx=2)

        # -- 外观 ---------------------------------------------------------- #
        box = ttk.LabelFrame(panel, text="外观", padding=10)
        box.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        row += 1

        self.font_var = tk.IntVar(value=_DEFAULTS.font_size)
        self.font_scale, self.font_label = self._add_slider(
            box, "字号 —— 决定输出图片的大小", 8, 48, _DEFAULTS.font_size,
            lambda v: f"{int(round(v))} px", self._on_font_change,
        )

        ttk.Label(box, text="字符集（顺序无所谓，分级按实测墨量排）").pack(anchor="w")
        self.chars_var = tk.StringVar(value=_DEFAULTS.chars)
        ttk.Entry(box, textvariable=self.chars_var).pack(fill="x", pady=(2, 4))
        self.chars_var.trace_add("write", lambda *_: self.schedule_render())

        char_btns = ttk.Frame(box)
        char_btns.pack(fill="x", pady=(0, 6))
        ttk.Button(
            char_btns, text="实测分级表…", command=self.show_ramp_table
        ).pack(side="left")
        ttk.Button(
            char_btns, text="块元素字符集",
            command=lambda: self.chars_var.set(BLOCK_CHARS),
        ).pack(side="left", padx=(4, 0))
        ttk.Button(
            char_btns, text="全 ASCII",
            command=lambda: self.chars_var.set(ASCII_CHARS),
        ).pack(side="left", padx=(4, 0))

        ttk.Label(box, text="背景").pack(anchor="w")
        self.bg_var = tk.StringVar(value="黑色")
        bg_combo = ttk.Combobox(
            box, textvariable=self.bg_var, values=list(BACKGROUNDS), state="readonly"
        )
        bg_combo.pack(fill="x")
        bg_combo.bind("<<ComboboxSelected>>", lambda _e: self.schedule_render())

        # -- 亮度与颜色 ---------------------------------------------------- #
        box = ttk.LabelFrame(panel, text="亮度与颜色", padding=10)
        box.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        row += 1

        ttk.Label(box, text="亮度度量方式").pack(anchor="w")
        self.metric_var = tk.StringVar(value=_DEFAULTS.metric)
        metric_combo = ttk.Combobox(
            box, textvariable=self.metric_var,
            values=sorted(BRIGHTNESS_METRICS), state="readonly",
        )
        metric_combo.pack(fill="x", pady=(2, 2))
        metric_combo.bind("<<ComboboxSelected>>", self._on_metric_change)
        self.metric_hint = ttk.Label(
            box, text="", foreground="#666", wraplength=252, justify="left"
        )
        self.metric_hint.pack(anchor="w", pady=(0, 6))

        self.candidates_var = tk.IntVar(value=_DEFAULTS.candidates)
        self.candidates_scale, self.candidates_label = self._add_slider(
            box, "查找候选数（1 = 只取上界那一个）", 1, 10, _DEFAULTS.candidates,
            lambda v: f"{int(round(v))} 个", self._on_candidates_change,
        )
        self.candidates_hint = ttk.Label(
            box,
            text="从亮度上界那级起往上多考察几个更密的字符，"
                 "取区域平均色最接近原图的；误差只会更小。",
            foreground="#666", wraplength=252, justify="left",
        )
        self.candidates_hint.pack(anchor="w", pady=(0, 6))

        self.highlight_var = tk.DoubleVar(value=_DEFAULTS.highlight)
        self.highlight_scale, self.highlight_label = self._add_slider(
            box, "高亮保色（1 = 高亮不变灰）", 0.0, 1.0, _DEFAULTS.highlight,
            lambda v: f"{v:.2f}", self._on_highlight_change,
        )

        self.image_sat_var = tk.DoubleVar(value=_DEFAULTS.image_saturation)
        self.image_sat_scale, self.image_sat_label = self._add_slider(
            box, "图像饱和度（0 = 灰度，>1 更艳）", 0.0, 2.0, _DEFAULTS.image_saturation,
            lambda v: f"{v:.2f}", self._on_image_sat_change,
        )

        ttk.Label(box, text="直方图均衡化").pack(anchor="w")
        self.equalize_var = tk.StringVar(value=_DEFAULTS.equalize)
        eq_combo = ttk.Combobox(
            box, textvariable=self.equalize_var,
            values=sorted(EQUALIZE_MODES), state="readonly",
        )
        eq_combo.pack(fill="x", pady=(2, 2))
        eq_combo.bind("<<ComboboxSelected>>", self._on_equalize_change)
        self.equalize_hint = ttk.Label(
            box, text="", foreground="#666", wraplength=252, justify="left"
        )
        self.equalize_hint.pack(anchor="w", pady=(0, 6))

        self.eq_window_var = tk.IntVar(value=_DEFAULTS.equalize_window)
        self.eq_window_scale, self.eq_window_label = self._add_slider(
            box, "局部窗口（字符单元，越小越局部）", 2, 64, _DEFAULTS.equalize_window,
            lambda v: f"{int(round(v))} 格", self._on_eq_window_change,
        )

        self.eq_clip_var = tk.DoubleVar(value=_DEFAULTS.equalize_clip)
        self.eq_clip_scale, self.eq_clip_label = self._add_slider(
            box, "局部限幅（0 = 不限幅 / 纯 AHE）", 0.0, 8.0, _DEFAULTS.equalize_clip,
            lambda v: "不限幅" if v < 0.05 else f"{v:.1f}x", self._on_eq_clip_change,
        )

        self.gamma_var = tk.DoubleVar(value=_DEFAULTS.gamma)
        self.gamma_scale, self.gamma_label = self._add_slider(
            box, "亮度伽马（<1 提亮，>1 压暗）", 0.3, 3.0, _DEFAULTS.gamma,
            lambda v: f"{v:.2f}", self._on_gamma_change,
        )

        ttk.Label(box, text="配色方式").pack(anchor="w")
        self.color_mode_var = tk.StringVar(value=_DEFAULTS.color_mode)
        mode_combo = ttk.Combobox(
            box, textvariable=self.color_mode_var,
            values=sorted(COLOR_MODES), state="readonly",
        )
        mode_combo.pack(fill="x", pady=(2, 2))
        mode_combo.bind("<<ComboboxSelected>>", self._on_color_mode_change)
        self.mode_hint = ttk.Label(
            box, text="", foreground="#666", wraplength=252, justify="left"
        )
        self.mode_hint.pack(anchor="w", pady=(0, 6))

        self.sat_var = tk.DoubleVar(value=_DEFAULTS.glyph_purity)
        self.sat_scale, self.sat_label = self._add_slider(
            box, "字符颜色纯度（仅「pure」配色生效）", 0.0, 1.0, _DEFAULTS.glyph_purity,
            lambda v: f"{v:.2f}", self._on_sat_change,
        )

        self.invert_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            box, text="反相（浅色背景用）", variable=self.invert_var,
            command=self.schedule_render,
        ).pack(anchor="w")

        # -- 输出信息 ------------------------------------------------------ #
        box = ttk.LabelFrame(panel, text="输出信息", padding=10)
        box.grid(row=row, column=0, sticky="ew")
        self.info_var = tk.StringVar(value="—")
        ttk.Label(box, textvariable=self.info_var, justify="left").pack(anchor="w")

        for child in panel.winfo_children():
            child.grid_configure(sticky="ew")
        panel.columnconfigure(0, weight=1)

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
        ttk.Label(parent, text=title).pack(anchor="w")
        scale = ttk.Scale(parent, from_=low, to=high, orient="horizontal", length=252)
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
            messagebox.showerror("字符集有问题", str(exc))
            return
        win = tk.Toplevel(self.root)
        win.title("字符分级表（实测）")
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
                title="选择图片",
                filetypes=[
                    ("图片", "*.png *.jpg *.jpeg *.bmp *.webp *.gif *.tif *.tiff"),
                    ("所有文件", "*.*"),
                ],
            )
        if not path:
            return
        try:
            with Image.open(path) as img:
                img.load()
                self.source = img.convert("RGBA")
        except OSError as exc:
            messagebox.showerror("打开失败", str(exc))
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
            title="导出 ASCII 图片", defaultextension=".png",
            initialfile=f"{stem}_ascii.png",
            filetypes=[("PNG 图片", "*.png"), ("WebP 图片", "*.webp"),
                       ("JPEG 图片", "*.jpg"), ("BMP 图片", "*.bmp")],
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
            messagebox.showerror("保存失败", str(exc))
            return
        self.status_var.set(f"已导出图片 {path}")

    def export_text(self) -> None:
        if self.result is None:
            return
        stem = self.source_path.stem if self.source_path else "ascii"
        path = filedialog.asksaveasfilename(
            title="导出字符文本", defaultextension=".txt",
            initialfile=f"{stem}_ascii.txt", filetypes=[("文本文件", "*.txt")],
        )
        if not path:
            return
        Path(path).write_text(self.result.text, encoding="utf-8")
        self.status_var.set(f"已导出文本 {path}")

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
        self.status_var.set("渲染中…")
        threading.Thread(
            target=self._worker,
            args=(self._token, self.source, self._collect_options()),
            daemon=True,
        ).start()

    def _worker(self, token: int, source: Image.Image, opts: AsciiOptions) -> None:
        started = time.perf_counter()
        try:
            with self._render_lock:      # 串行化：快速拖动会叠出多个请求
                result = convert(source, opts)
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
                    self.status_var.set(f"渲染失败：{error}")
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
            f"网格：{res.cols} 列 x {res.rows} 行\n"
            f"字框：{res.cell_w} x {res.cell_h} px\n"
            f"输出：{ow} x {oh} px\n"
            f"源图：{sw} x {sh} px\n"
            f"字体：{font} @ {res.font_size}\n"
            f"分级：{len(res.ramp.chars)} 级，用到 {len(used)} 级\n"
            f"最大墨量：{res.ramp.max_ink:.3f}"
            f"（亮度上限 {res.ramp.max_ink:.0%}）\n"
            f"候选：{self.candidates_var.get()} 个"
            f"，{int((res.grid_index != res.base_index).sum())} 个单元换用了更密的字符\n"
            f"耗时：{self.last_elapsed * 1000:.0f} ms"
        )
        self.font_hint_var.set(f"{font} @ {res.font_size}")
        self.status_var.set(
            f"{res.cols} x {res.rows} 个字符 · 输出 {ow}x{oh} · "
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
                text="点击左上角「打开图片…」\n然后拖动右侧的字符密度滑块",
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
            "缺少等宽字体",
            "系统里找不到可用的等宽字体，请安装 Consolas / DejaVu Sans Mono 等，"
            "或在命令行用 --font 指定字体文件。",
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
