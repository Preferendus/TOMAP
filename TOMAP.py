"""TOMAP – multi-criteria decision analysis with a trade-off matrix."""

import math
import struct
import sys
import tkinter as tk
import zlib
from dataclasses import dataclass
from tkinter import font as tkfont
from tkinter import messagebox, ttk

import numpy as np

APP_NAME = "TOMAP"
APP_TAGLINE = "Multi-criteria decision analysis"
VERSION = "20261008"


@dataclass(frozen=True)
class RankedAlternative:
    rank: int
    index: int
    name: str
    pct: int


def equal_split(n):
    """Split 100 into n integer parts, e.g. 3 -> [33, 33, 34]."""
    base, remainder = divmod(100, n)
    return [base + (1 if i >= n - remainder else 0) for i in range(n)]


def compute_rankings(scores, sub_weights, stakeholder_weights):
    """Return the alternatives ranked best-first on a relative 0-100 scale."""
    for i, stakeholder_score in enumerate(scores):
        for r, row in enumerate(stakeholder_score):
            for c, val in enumerate(row):
                if not (0 <= val <= 100):
                    raise ValueError(
                        f"Invalid score {val} in Actor {i + 1}, Alternative {r + 1}, "
                        f"Criterion {c + 1} (must be 0–100)"
                    )

    scores = np.array(scores, dtype=float)
    sub_weights = np.array(sub_weights, dtype=float)
    stakeholder_weights = np.array(stakeholder_weights, dtype=float)

    global_criteria_weights = stakeholder_weights[:, None] * sub_weights
    mean_scores = scores.mean(axis=1)
    std_scores = np.std(scores, axis=1, ddof=0)
    std_scores_safe = np.where(std_scores == 0, 1e-6, std_scores)

    z_scores = (scores - mean_scores[:, np.newaxis, :]) / std_scores_safe[:, np.newaxis, :]
    global_criteria_weights_T = global_criteria_weights.T

    LSD = np.einsum("sak,ks->a", z_scores, global_criteria_weights_T)
    LSD_min, LSD_max = np.min(LSD), np.max(LSD)
    percentages = (LSD - LSD_min) / (LSD_max - LSD_min) if LSD_max != LSD_min else np.full_like(LSD, 0.5)

    ordered = sorted(
        ((int(round(p * 100)), i) for i, p in enumerate(percentages)),
        key=lambda item: (-item[0], item[1]),
    )
    ranked = []
    rank = 1
    prev_pct = None
    for position, (pct, index) in enumerate(ordered):
        if pct != prev_pct:
            rank = position + 1
            prev_pct = pct
        ranked.append(RankedAlternative(rank, index, f"Alternative {index + 1}", pct))
    return ranked


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Palette:
    name: str
    bg: str
    surface: str
    surface_alt: str
    border: str
    text: str
    muted: str
    accent: str
    accent_hover: str
    accent_fg: str
    accent_soft: str
    danger: str
    danger_soft: str
    success: str
    success_soft: str
    warning_fg: str
    warning_soft: str
    warning_border: str
    track: str


LIGHT = Palette(
    name="light", bg="#F3F5F9", surface="#FFFFFF", surface_alt="#F1F5F9", border="#D9DFE8",
    text="#1E293B", muted="#64748B", accent="#2563EB", accent_hover="#1D4ED8", accent_fg="#FFFFFF",
    accent_soft="#DBEAFE", danger="#DC2626", danger_soft="#FEE2E2", success="#15803D",
    success_soft="#DCFCE7", warning_fg="#92400E", warning_soft="#FEF3C7", warning_border="#FCD34D",
    track="#E2E8F0",
)
DARK = Palette(
    name="dark", bg="#0F172A", surface="#1E293B", surface_alt="#273549", border="#3B4A61",
    text="#F1F5F9", muted="#94A3B8", accent="#3B82F6", accent_hover="#60A5FA", accent_fg="#FFFFFF",
    accent_soft="#1E3A8A", danger="#F87171", danger_soft="#7F1D1D", success="#4ADE80",
    success_soft="#14532D", warning_fg="#FDE68A", warning_soft="#3B2F0B", warning_border="#A16207",
    track="#334155",
)


def _rgb(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def _lerp_color(a, b, t):
    ra, ga, ba = _rgb(a)
    rb, gb, bb = _rgb(b)
    return "#%02X%02X%02X" % (round(ra + (rb - ra) * t), round(ga + (gb - ga) * t), round(ba + (bb - ba) * t))


def _png_from_rgba(width, height, rows):
    raw = bytearray()
    for row in rows:
        raw.append(0)
        for px in row:
            raw.extend(px)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw)))
        + chunk(b"IEND", b"")
    )


def _antialiased_png(width, height, color_at, supersample=4):
    """Rasterise a shape given by color_at(x, y) -> rgb | None into a PNG with alpha."""
    samples = supersample * supersample
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            r = g = b = cover = 0
            for sy in range(supersample):
                for sx in range(supersample):
                    c = color_at(x + (sx + 0.5) / supersample, y + (sy + 0.5) / supersample)
                    if c:
                        r += c[0]
                        g += c[1]
                        b += c[2]
                        cover += 1
            row.append((r // cover, g // cover, b // cover, round(255 * cover / samples)) if cover else (0, 0, 0, 0))
        rows.append(row)
    return _png_from_rgba(width, height, rows)


def _thumb_png(size, radius, fill, ring, ring_width):
    cx = cy = size / 2
    fill_rgb, ring_rgb = _rgb(fill), _rgb(ring)

    def color_at(x, y):
        d = math.hypot(x - cx, y - cy)
        if d <= radius - ring_width:
            return fill_rgb
        if d <= radius:
            return ring_rgb
        return None

    return _antialiased_png(size, size, color_at)


def _trough_png(width, height, bar_height, color):
    cy, rad = height / 2, bar_height / 2
    ax, bx = rad, width - rad
    rgb = _rgb(color)

    def color_at(x, y):
        px = min(max(x, ax), bx)
        return rgb if math.hypot(x - px, y - cy) <= rad else None

    return _antialiased_png(width, height, color_at)


class ThemeManager:
    """Owns the ttk style, fonts and palette; notifies non-ttk widgets on theme change."""

    THUMB_SIZE = 22

    def __init__(self, root):
        self.root = root
        self.palette = DARK
        self._subscribers = []
        self.style = ttk.Style(root)
        self.style.theme_create("afa", parent="clam")
        self.style.theme_use("afa")
        self.fonts = self._make_fonts()
        self._thumb = tk.PhotoImage(master=root)
        self._thumb_hover = tk.PhotoImage(master=root)
        self._trough = tk.PhotoImage(master=root)
        self.style.element_create(
            "Horizontal.Scale.trough", "image", self._trough, border=(10, 0, 10, 0), sticky="we"
        )
        self.style.element_create(
            "Horizontal.Scale.slider", "image", self._thumb,
            ("pressed", self._thumb_hover), ("active", self._thumb_hover), sticky="",
        )
        self.apply(DARK)

    def _make_fonts(self):
        families = set(tkfont.families(self.root))
        family = "Segoe UI" if "Segoe UI" in families else tkfont.nametofont("TkDefaultFont").actual("family")
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont"):
            tkfont.nametofont(name).configure(family=family, size=10)

        def make(size, weight="normal"):
            return tkfont.Font(root=self.root, family=family, size=size, weight=weight)

        return {
            "base": tkfont.nametofont("TkDefaultFont"),
            "bold": make(10, "bold"),
            "small": make(9),
            "small_bold": make(9, "bold"),
            "h2": make(12, "bold"),
            "title": make(17, "bold"),
            "marker": make(10, "bold"),
        }

    def bind(self, widget, callback):
        """Call callback(palette) on every theme change until the widget is destroyed."""
        self._subscribers.append(callback)

        def _forget(event):
            if event.widget is widget and callback in self._subscribers:
                self._subscribers.remove(callback)

        widget.bind("<Destroy>", _forget, add="+")

    def toggle(self):
        self.apply(DARK if self.palette is LIGHT else LIGHT)

    def apply(self, palette):
        self.palette = palette
        self._update_images(palette)
        self._configure_styles(palette)
        self.root.configure(bg=palette.bg)
        for callback in list(self._subscribers):
            try:
                callback(palette)
            except tk.TclError:
                self._subscribers.remove(callback)

    def _update_images(self, p):
        size = self.THUMB_SIZE
        self._thumb.configure(data=_thumb_png(size, 9.5, p.accent, p.surface, 2.5))
        self._thumb_hover.configure(data=_thumb_png(size, 10.5, p.accent_hover, p.surface, 2.5))
        self._trough.configure(data=_trough_png(28, size, 6, p.track))

    def _configure_styles(self, p):
        s, f = self.style, self.fonts
        s.configure(
            ".", background=p.bg, foreground=p.text, font=f["base"], bordercolor=p.border,
            lightcolor=p.surface, darkcolor=p.border, troughcolor=p.track, focuscolor=p.accent,
            selectbackground=p.accent, selectforeground=p.accent_fg, insertcolor=p.text,
            fieldbackground=p.surface,
        )
        # Frames
        s.configure("TFrame", background=p.bg)
        s.configure("Surface.TFrame", background=p.surface)
        s.configure("Border.TFrame", background=p.border)
        s.configure("Card.TFrame", background=p.surface, bordercolor=p.border, relief="solid", borderwidth=1)
        s.configure("Banner.TFrame", background=p.warning_soft, bordercolor=p.warning_border, relief="solid", borderwidth=1)
        # Labels
        s.configure("TLabel", background=p.bg, foreground=p.text)
        s.configure("Surface.TLabel", background=p.surface, foreground=p.text)
        s.configure("Bold.TLabel", background=p.surface, foreground=p.text, font=f["bold"])
        s.configure("H2.TLabel", background=p.surface, foreground=p.text, font=f["h2"])
        s.configure("Title.TLabel", background=p.surface, foreground=p.text, font=f["title"])
        s.configure("Muted.TLabel", background=p.surface, foreground=p.muted, font=f["small"])
        s.configure("Error.TLabel", background=p.surface, foreground=p.danger, font=f["small_bold"])
        s.configure("Banner.TLabel", background=p.warning_soft, foreground=p.warning_fg, font=f["small"])
        s.configure("Chip.TLabel", background=p.accent_soft, foreground=p.accent if p is LIGHT else p.accent_hover, font=f["small_bold"], padding=(10, 3))
        s.configure("Ok.Chip.TLabel", background=p.success_soft, foreground=p.success, font=f["small_bold"], padding=(10, 3))
        s.configure("Bad.Chip.TLabel", background=p.danger_soft, foreground=p.danger, font=f["small_bold"], padding=(10, 3))
        s.configure("Rank.TLabel", background=p.accent, foreground=p.accent_fg, font=f["small_bold"], padding=(8, 2), anchor="center")
        # Buttons
        s.configure(
            "Accent.TButton", background=p.accent, foreground=p.accent_fg, bordercolor=p.accent,
            lightcolor=p.accent, darkcolor=p.accent, focuscolor=p.accent_fg, borderwidth=1,
            padding=(18, 8), font=f["bold"],
        )
        s.map(
            "Accent.TButton",
            background=[("disabled", p.border), ("pressed", p.accent_hover), ("active", p.accent_hover)],
            bordercolor=[("disabled", p.border), ("pressed", p.accent_hover), ("active", p.accent_hover)],
            lightcolor=[("disabled", p.border), ("pressed", p.accent_hover), ("active", p.accent_hover)],
            darkcolor=[("disabled", p.border), ("pressed", p.accent_hover), ("active", p.accent_hover)],
            foreground=[("disabled", p.muted)],
        )
        s.configure(
            "Secondary.TButton", background=p.surface, foreground=p.text, bordercolor=p.border,
            lightcolor=p.border, darkcolor=p.border, focuscolor=p.accent, borderwidth=1,
            relief="solid", padding=(14, 7), font=f["base"],
        )
        s.map(
            "Secondary.TButton",
            relief=[("pressed", "solid"), ("active", "solid")],
            background=[("pressed", p.surface_alt), ("active", p.surface_alt)],
            bordercolor=[("pressed", p.accent), ("active", p.accent)],
            lightcolor=[("pressed", p.accent), ("active", p.accent)],
            darkcolor=[("pressed", p.accent), ("active", p.accent)],
        )
        # Entry / Spinbox
        for name in ("TEntry", "TSpinbox"):
            s.configure(
                name, fieldbackground=p.surface, background=p.surface, foreground=p.text,
                bordercolor=p.border, lightcolor=p.border, darkcolor=p.border, insertcolor=p.text,
                arrowcolor=p.muted, arrowsize=14, padding=(6, 4),
            )
            s.map(
                name,
                bordercolor=[("focus", p.accent)], lightcolor=[("focus", p.accent)],
                darkcolor=[("focus", p.accent)], arrowcolor=[("active", p.accent)],
                fieldbackground=[("readonly", p.surface_alt)],
            )
        # Notebook
        s.configure("TNotebook", background=p.bg, borderwidth=0, bordercolor=p.border, lightcolor=p.surface, darkcolor=p.surface, tabmargins=(0, 0, 0, 0))
        s.configure(
            "TNotebook.Tab", background=p.bg, foreground=p.muted, padding=(18, 10), borderwidth=1,
            bordercolor=p.bg, lightcolor=p.bg, font=f["bold"],
        )
        s.map(
            "TNotebook.Tab",
            background=[("selected", p.surface)], foreground=[("selected", p.accent), ("active", p.text)],
            lightcolor=[("selected", p.surface)], bordercolor=[("selected", p.border)],
            expand=[("selected", (0, 0, 0, 0))],
        )
        # Scale, scrollbar, separator
        s.configure("Horizontal.TScale", background=p.surface, borderwidth=0)
        s.configure(
            "TScrollbar", background=p.border, troughcolor=p.surface, bordercolor=p.surface,
            lightcolor=p.border, darkcolor=p.border, arrowsize=12, gripcount=0, relief="flat",
        )
        s.map("TScrollbar", background=[("active", p.muted), ("pressed", p.muted)])
        s.layout("Vertical.TScrollbar", [("Vertical.Scrollbar.trough", {"sticky": "ns", "children": [("Vertical.Scrollbar.thumb", {"expand": 1, "sticky": "nswe"})]})])
        s.layout("Horizontal.TScrollbar", [("Horizontal.Scrollbar.trough", {"sticky": "we", "children": [("Horizontal.Scrollbar.thumb", {"expand": 1, "sticky": "nswe"})]})])
        s.configure("TSeparator", background=p.border)


# --------------------------------------------------------------------------- #
# Reusable widgets
# --------------------------------------------------------------------------- #
class ScrollableFrame(ttk.Frame):
    """A frame whose `inner` child scrolls vertically (and horizontally when it must)."""

    def __init__(self, master, theme, padding: int | tuple[int, int] = 0):
        super().__init__(master, style="Surface.TFrame")
        self.theme = theme
        self.width_listeners = []
        self._hbar_visible = False
        self._vbar_visible = True
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0, bg=theme.palette.surface)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)
        self.inner = ttk.Frame(self.canvas, style="Surface.TFrame", padding=padding)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        theme.bind(self, lambda p: self.canvas.configure(bg=p.surface))

    def _on_inner_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._update_scrollbars()

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._window, width=max(event.width, self.inner.winfo_reqwidth()))
        for listener in self.width_listeners:
            listener(event.width)
        self._update_scrollbars()

    def _update_scrollbars(self):
        need_h = self.inner.winfo_reqwidth() > self.canvas.winfo_width()
        if need_h and not self._hbar_visible:
            self.hbar.grid(row=1, column=0, sticky="ew")
            self._hbar_visible = True
        elif not need_h and self._hbar_visible:
            self.hbar.grid_remove()
            self._hbar_visible = False
        need_v = self.inner.winfo_reqheight() > self.canvas.winfo_height()
        if need_v and not self._vbar_visible:
            self.vbar.grid(row=0, column=1, sticky="ns")
            self._vbar_visible = True
        elif not need_v and self._vbar_visible:
            self.vbar.grid_remove()
            self._vbar_visible = False

    def scroll(self, delta, horizontal=False):
        if horizontal:
            self.canvas.xview_scroll(delta, "units")
        else:
            self.canvas.yview_scroll(delta, "units")


class Card(ttk.Frame):
    """Bordered section with a title row (`head`), optional subtitle and a `body`."""

    def __init__(self, master, title, subtitle=None):
        super().__init__(master, style="Card.TFrame", padding=(20, 12, 20, 14))
        self.head = ttk.Frame(self, style="Surface.TFrame")
        self.head.pack(fill="x")
        ttk.Label(self.head, text=title, style="H2.TLabel").pack(side="left")
        self.subtitle = None
        if subtitle:
            self.subtitle = ttk.Label(self, text=subtitle, style="Muted.TLabel", justify="left", wraplength=640)
            self.subtitle.pack(anchor="w", pady=(2, 14))
        self.body = ttk.Frame(self, style="Surface.TFrame")
        self.body.pack(fill="x")

    def set_available_width(self, width):
        if self.subtitle is not None:
            self.subtitle.configure(wraplength=max(240, width - 120))


class Banner(ttk.Frame):
    def __init__(self, master, text):
        super().__init__(master, style="Banner.TFrame", padding=(14, 10))
        self.label = ttk.Label(self, text=text, style="Banner.TLabel", justify="left", wraplength=640)
        self.label.pack(anchor="w")

    def set_available_width(self, width):
        self.label.configure(wraplength=max(240, width - 110))


class ValueInput(ttk.Frame):
    """Slider (0-100) with a linked numeric box; both stay in sync."""

    def __init__(self, master, value=0, length=110, on_change=None):
        super().__init__(master, style="Surface.TFrame")
        self._on_change = on_change
        self._syncing = False
        self._value = int(value)
        self._text = tk.StringVar(value=str(self._value))

        self.scale = ttk.Scale(self, from_=0, to=100, orient="horizontal", length=length, command=self._from_scale)
        self.scale.set(self._value)
        validate = (self.register(self._validate), "%P")
        self.entry = ttk.Entry(
            self, textvariable=self._text, width=4, justify="center",
            validate="key", validatecommand=validate,
        )
        self.entry.bind("<FocusOut>", self._commit_entry)
        self.entry.bind("<Return>", self._commit_entry)
        self._text.trace_add("write", self._from_entry)

        self.scale.pack(side="left", padx=(0, 8))
        self.entry.pack(side="left")

    @staticmethod
    def _validate(proposed):
        return proposed == "" or (proposed.isdigit() and len(proposed) <= 3 and int(proposed) <= 100)

    def get(self):
        return self._value

    def set(self, value):
        value = max(0, min(100, int(value)))
        self._syncing = True
        try:
            self._value = value
            self.scale.set(value)
            self._text.set(str(value))
        finally:
            self._syncing = False
        self._notify()

    def _notify(self):
        if self._on_change is not None:
            self._on_change()

    def _from_scale(self, raw):
        if self._syncing:
            return
        value = int(round(float(raw)))
        if value == self._value and self._text.get() == str(value):
            return
        self._syncing = True
        try:
            self._value = value
            self._text.set(str(value))
        finally:
            self._syncing = False
        self._notify()

    def _from_entry(self, *_args):
        if self._syncing:
            return
        text = self._text.get()
        if text == "":
            return
        value = int(text)
        self._syncing = True
        try:
            self._value = value
            self.scale.set(value)
        finally:
            self._syncing = False
        self._notify()

    def _commit_entry(self, _event=None):
        if self._text.get() == "":
            self._syncing = True
            try:
                self._text.set(str(self._value))
            finally:
                self._syncing = False


class FixedValue:
    """Stand-in for an input whose value is implied (single criterion / single Actor)."""

    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, _value):
        pass


class RankingScale(tk.Canvas):
    """Draws the alternatives as markers on a 0-100 scale, avoiding overlaps."""

    PAD_X = 88
    R = 16
    TRACK_H = 10

    def __init__(self, master, theme):
        super().__init__(master, highlightthickness=0, bd=0, width=400, height=160)
        self.theme = theme
        self.results = []
        self.bind("<Configure>", lambda _e: self.redraw())
        theme.bind(self, lambda _p: self.redraw())

    def set_results(self, results):
        self.results = list(results)
        self.redraw()

    def redraw(self):
        p, f = self.theme.palette, self.theme.fonts
        self.configure(bg=p.surface)
        self.delete("all")
        width = self.winfo_width()
        if width < 120:
            return

        r = self.R
        row_h = f["bold"].metrics("linespace")
        tick_text_h = f["small"].metrics("linespace")
        half = self.TRACK_H / 2
        caption_w = max(f["small_bold"].measure("RELATIVELY WORST"), f["small_bold"].measure("RELATIVELY BEST"))
        pad = max(self.PAD_X, r + 14 + caption_w + 8)
        x0, x1 = pad, width - pad
        track_w = x1 - x0
        # Vertical budget above the track: major tick (12) + gap + tick number.
        tick_zone = 12 + 6 + tick_text_h
        # Distance from the track centre to the bottom of a stacked marker's halo.
        stack_base = half + tick_zone + 10

        # Lay out markers left-to-right; a marker plus its label is one unit per lane.
        items = sorted(self.results, key=lambda a: (a.pct, a.rank))
        units = []
        lane_right = []
        for item in items:
            x = x0 + track_w * item.pct / 100
            name_w = f["bold"].measure(item.name)
            score_w = f["bold"].measure(str(item.pct))
            label_w = name_w + 6 + score_w
            unit_w = max(2 * r, label_w)
            left = min(max(x - unit_w / 2, 6), width - 6 - unit_w)
            for lane, right in enumerate(lane_right):
                if left >= right + 12:
                    lane_right[lane] = left + unit_w
                    break
            else:
                lane = len(lane_right)
                lane_right.append(left + unit_w)
            units.append((item, x, left, name_w, label_w, lane))

        max_lane = max((u[5] for u in units), default=0)
        band_h = 2 * r + row_h + 10
        if max_lane == 0:
            y_track = 10 + tick_zone + half
        else:
            y_track = 10 + row_h + 6 + r + stack_base + r + (max_lane - 1) * band_h
        height = int(y_track + half + 12 + row_h + 12)
        if int(self.cget("height")) != height:
            self.configure(height=height)

        # Track with a soft gradient from "worst" to "best".
        segments = 72
        for i in range(segments):
            xa = x0 + track_w * i / segments
            xb = x0 + track_w * (i + 1) / segments + 1
            self.create_rectangle(xa, y_track - half, xb, y_track + half, fill=_lerp_color(p.track, p.accent, i / (segments - 1)), outline="")
        self.create_oval(x0 - half, y_track - half, x0 + half, y_track + half, fill=p.track, outline="")
        self.create_oval(x1 - half, y_track - half, x1 + half, y_track + half, fill=p.accent, outline="")

        for value in range(0, 101, 5):
            x = x0 + track_w * value / 100
            major = value % 25 == 0
            top = y_track - half - (12 if major else 6)
            self.create_line(x, y_track - half - 2, x, top, fill=p.muted if major else p.border, width=1)
            if major:
                self.create_text(x, top - 6 - tick_text_h / 2, text=str(value), fill=p.muted, font=f["small"])
        # End captions sit beside the track, clear of markers and tick numbers.
        self.create_text(x0 - r - 14, y_track, text="RELATIVELY WORST", anchor="e", fill=p.muted, font=f["small_bold"])
        self.create_text(x1 + r + 14, y_track, text="RELATIVELY BEST", anchor="w", fill=p.accent, font=f["small_bold"])

        # Connectors first so markers are drawn on top of them.
        for item, x, left, name_w, label_w, lane in units:
            if lane > 0:
                y_marker = y_track - stack_base - r - (lane - 1) * band_h
                self.create_line(x, y_marker + r, x, y_track, fill=p.border, width=2)

        for item, x, left, name_w, label_w, lane in units:
            if lane == 0:
                y_marker = y_track
                y_label = y_track + half + 12 + row_h / 2
            else:
                y_marker = y_track - stack_base - r - (lane - 1) * band_h
                y_label = y_marker - r - 6 - row_h / 2
            if item.rank == 1:
                self.create_oval(x - r - 5, y_marker - r - 5, x + r + 5, y_marker + r + 5, fill="", outline=p.accent_soft, width=3)
            self.create_oval(x - r, y_marker - r, x + r, y_marker + r, fill=p.accent, outline=p.surface, width=2)
            self.create_text(x, y_marker, text=str(item.rank), fill=p.accent_fg, font=f["marker"])
            label_left = left + (max(2 * r, label_w) - label_w) / 2
            self.create_text(label_left, y_label, text=item.name, anchor="w", fill=p.text, font=f["bold"])
            self.create_text(label_left + name_w + 6, y_label, text=str(item.pct), anchor="w", fill=p.accent, font=f["bold"])


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
class SetupView(ttk.Frame):
    FIELDS = (
        ("alternatives", "Alternatives", "The options you are comparing. At least 2.", 3, 2),
        ("criteria", "Criteria", "What the alternatives are judged on. At least 1.", 3, 1),
        ("stakeholders", "Actors", "People or parties whose judgement counts. At least 1.", 2, 1),
    )

    def __init__(self, master, theme, on_submit):
        super().__init__(master, style="TFrame")
        self.on_submit = on_submit
        card = ttk.Frame(self, style="Card.TFrame", padding=(36, 32))
        self.card = card
        card.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(card, text="Set up your trade-off matrix", style="Title.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            card, style="Muted.TLabel", justify="left", wraplength=440,
            text=(
                "Define the size of the decision problem. Next, every Actor scores each "
                "alternative on each criterion and assigns weights; the tool then ranks the alternatives."
            ),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 22))

        self.vars = {}
        for i, (key, label, helper, default, minimum) in enumerate(self.FIELDS):
            row = 2 + i
            text = ttk.Frame(card, style="Surface.TFrame")
            text.grid(row=row, column=0, sticky="w", pady=9, padx=(0, 40))
            ttk.Label(text, text=label, style="Bold.TLabel").pack(anchor="w")
            ttk.Label(text, text=helper, style="Muted.TLabel").pack(anchor="w")
            var = tk.StringVar(value=str(default))
            self.vars[key] = var
            ttk.Spinbox(card, from_=minimum, to=99, textvariable=var, width=6, justify="center", font=theme.fonts["bold"]).grid(row=row, column=1, sticky="e")

        self.error = ttk.Label(card, text="", style="Error.TLabel", wraplength=440, justify="left")
        self.error.grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Button(card, text="Create trade-off matrix", style="Accent.TButton", command=self.submit).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(14, 0))

    def submit(self):
        try:
            values = {key: int(var.get().strip()) for key, var in self.vars.items()}
        except ValueError:
            self.error.configure(text="Please enter whole numbers for alternatives, criteria and Actors.")
            return
        if values["alternatives"] < 2:
            self.error.configure(text="You must define at least 2 alternatives to proceed.")
            return
        if values["criteria"] < 1 or values["stakeholders"] < 1:
            self.error.configure(text="You need at least 1 criterion and at least 1 Actor.")
            return
        self.error.configure(text="")
        self.on_submit(values["alternatives"], values["criteria"], values["stakeholders"])


class ResultsView(ScrollableFrame):
    EMPTY_TEXT = "No results yet. Fill in the trade-off matrix for every Actor and press Solve."

    def __init__(self, master, theme):
        super().__init__(master, theme, padding=20)
        self.card = Card(
            self.inner, "Ranking of the alternatives",
            "Each alternative is placed on a relative 0–100 scale: 100 marks the best among "
            "alternatives and 0 the worst among alternatives. Alternatives with the same score share a rank.",
        )
        self.card.pack(fill="x")
        self.width_listeners.append(self.card.set_available_width)
        self.empty = ttk.Label(self.card.body, text=self.EMPTY_TEXT, style="Muted.TLabel")
        self.empty.pack(anchor="w", pady=16)
        self.scale = RankingScale(self.card.body, theme)
        self.separator = ttk.Separator(self.card.body, orient="horizontal")
        self.table = ttk.Frame(self.card.body, style="Surface.TFrame")

    def show(self, results):
        self.empty.pack_forget()
        self.scale.pack(fill="x", pady=(4, 8))
        self.scale.set_results(results)
        self.separator.pack(fill="x", pady=(4, 14))

        for child in self.table.winfo_children():
            child.destroy()
        for col, text in enumerate(("Rank", "Alternative", "Score")):
            ttk.Label(self.table, text=text, style="Muted.TLabel").grid(row=0, column=col, sticky="w", padx=(0, 28), pady=(0, 6))
        for i, item in enumerate(results):
            row = 1 + i
            ttk.Label(self.table, text=str(item.rank), style="Rank.TLabel", width=3).grid(row=row, column=0, sticky="w", padx=(0, 28), pady=3)
            ttk.Label(self.table, text=item.name, style="Bold.TLabel").grid(row=row, column=1, sticky="w", padx=(0, 28), pady=3)
            ttk.Label(self.table, text=f"{item.pct} / 100", style="Surface.TLabel").grid(row=row, column=2, sticky="w", pady=3)
        self.table.pack(fill="x")

    def clear(self):
        self.scale.pack_forget()
        self.separator.pack_forget()
        self.table.pack_forget()
        self.empty.pack(anchor="w", pady=16)


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
class MCDAApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_NAME)
        self._size_window(1120, 760, 700, 320)
        self.theme = ThemeManager(root)

        self.n_scenarios = self.n_criteria = self.n_stakeholders = 0
        self.stakeholder_tables = []
        self.setup_view = None
        self.notebook = None
        self.results_view = None
        self._global_chips = []

        self._build_chrome()
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Shift-MouseWheel>", lambda e: self._on_mousewheel(e, horizontal=True))
        self.root.bind("<Return>", self._on_return)
        self.setup_initial_ui()

    # ----- window chrome ---------------------------------------------------- #
    def _size_window(self, width, height, min_width, min_height):
        # Tk scales fonts with the display DPI but not pixel geometry; follow suit.
        factor = max(1.0, self.root.winfo_fpixels("1i") / 96.0)
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w = min(int(width * factor), screen_w - 40)
        h = min(int(height * factor), screen_h - 120)
        self._max_window_size = (w, h)
        x, y = max(0, (screen_w - w) // 2), max(0, (screen_h - h) // 2 - 30)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.minsize(min(int(min_width * factor), w), min(int(min_height * factor), h))

    def _fit_window_to_content(self):
        self.root.update_idletasks()
        if self.setup_view is not None:
            content_w = self.setup_view.card.winfo_reqwidth() + 48
            content_h = self.setup_view.card.winfo_reqheight() + 48
        elif self.notebook is not None:
            pages = [self.root.nametowidget(tab) for tab in self.notebook.tabs()]
            content_w = max(page.inner.winfo_reqwidth() for page in pages) + 52 + pages[0].vbar.winfo_reqwidth()
        else:
            return
        chrome = [child for child in self.root.winfo_children() if child is not self.content]
        min_w, min_h = self.root.minsize()
        width = min(self._max_window_size[0], max(min_w, content_w, *(child.winfo_reqwidth() for child in chrome)))
        self.root.geometry(f"{width}x{self.root.winfo_height()}")
        self.root.update_idletasks()
        if self.notebook is not None:
            tab_height = self.notebook.winfo_reqheight() - max(page.winfo_reqheight() for page in pages)
            content_h = max(page.inner.winfo_reqheight() for page in pages) + tab_height + 44
            if content_w > self._max_window_size[0]:
                content_h += pages[0].hbar.winfo_reqheight()
        height = min(self._max_window_size[1], max(min_h, content_h + sum(child.winfo_reqheight() for child in chrome)))
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        window_x = min(self.root.winfo_x(), max(0, screen_w - width - 20))
        window_y = min(self.root.winfo_y(), max(0, screen_h - height - 80))
        self.root.geometry(f"{width}x{height}+{max(0, window_x)}+{max(0, window_y)}")

    def _build_chrome(self):
        header = ttk.Frame(self.root, style="Surface.TFrame", padding=(24, 14))
        header.pack(fill="x")
        ttk.Frame(self.root, style="Border.TFrame", height=1).pack(fill="x")
        self.content = ttk.Frame(self.root, style="TFrame")
        self.content.pack(fill="both", expand=True)
        ttk.Frame(self.root, style="Border.TFrame", height=1).pack(fill="x")
        footer = ttk.Frame(self.root, style="Surface.TFrame", padding=(24, 10))
        footer.pack(fill="x")

        titles = ttk.Frame(header, style="Surface.TFrame")
        titles.pack(side="left")
        self.subtitle_var = tk.StringVar(value=APP_TAGLINE)
        ttk.Label(titles, textvariable=self.subtitle_var, style="Muted.TLabel").pack(anchor="w")

        self.theme_button = ttk.Button(header, style="Secondary.TButton", command=self._toggle_theme)
        self.theme_button.pack(side="right")
        self.start_over_button = ttk.Button(header, text="Start over", style="Secondary.TButton", command=self._confirm_start_over)
        self._update_theme_button(self.theme.palette)

        version = ttk.Label(footer, text=f"Version {VERSION}", style="Muted.TLabel")
        version.pack(side="left", padx=(0, 24))
        self.actions = ttk.Frame(footer, style="Surface.TFrame")
        self.actions.pack(side="right")
        self.solve_button = ttk.Button(self.actions, text="Solve", style="Accent.TButton", command=self.calculate_rankings)
        self.reset_scores_button = ttk.Button(self.actions, text="Reset scores", style="Secondary.TButton", command=self.reset_ratings_only)
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(footer, textvariable=self.status_var, style="Muted.TLabel", justify="left")
        self.status_label.pack(side="left", fill="x", expand=True, padx=(0, 16))

        def _fit_status(event):
            available = event.width - version.winfo_reqwidth() - self.actions.winfo_reqwidth() - 100
            self.status_label.configure(wraplength=max(200, available))

        footer.bind("<Configure>", _fit_status)

    def _toggle_theme(self):
        self.theme.toggle()
        self._update_theme_button(self.theme.palette)

    def _update_theme_button(self, palette):
        self.theme_button.configure(text="Dark mode" if palette is LIGHT else "Light mode")

    def set_status(self, text, error=False):
        self.status_var.set(text)
        self.status_label.configure(style="Error.TLabel" if error else "Muted.TLabel")

    def _clear_content(self):
        for child in self.content.winfo_children():
            child.destroy()
        self.setup_view = None
        self.notebook = None
        self.results_view = None
        self.stakeholder_tables = []
        self._global_chips = []

    def _on_return(self, _event):
        if self.setup_view is not None:
            self.setup_view.submit()

    def _on_mousewheel(self, event, horizontal=False):
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        while widget is not None and not isinstance(widget, ScrollableFrame):
            widget = widget.master
        if widget is not None:
            widget.scroll(-int(event.delta / 120), horizontal=horizontal)

    # ----- screens ---------------------------------------------------------- #
    def setup_initial_ui(self):
        self._clear_content()
        self.subtitle_var.set(APP_TAGLINE)
        self.start_over_button.pack_forget()
        self.solve_button.pack_forget()
        self.reset_scores_button.pack_forget()
        self.set_status("")
        self.setup_view = SetupView(self.content, self.theme, self.generate_tables)
        self.setup_view.pack(fill="both", expand=True)
        self._fit_window_to_content()

    def generate_tables(self, n_scenarios, n_criteria, n_stakeholders):
        self.n_scenarios, self.n_criteria, self.n_stakeholders = n_scenarios, n_criteria, n_stakeholders
        self._clear_content()

        self.notebook = ttk.Notebook(self.content)
        self.notebook.pack(fill="both", expand=True, padx=24, pady=(18, 24))
        for s in range(n_stakeholders):
            self._build_stakeholder_tab(s)
        self.results_view = ResultsView(self.notebook, self.theme)
        self.notebook.add(self.results_view, text="Results")

        plural = "s" if n_stakeholders > 1 else ""
        self.subtitle_var.set(f"{n_scenarios} alternatives  ·  {n_criteria} criteri{'a' if n_criteria > 1 else 'on'}  ·  {n_stakeholders} Actor{plural}")
        self.start_over_button.pack(side="right", padx=(0, 10))
        self.solve_button.pack(side="right")
        self.reset_scores_button.pack(side="right", padx=(0, 10))
        self._update_global_total()
        self.set_status("Score every alternative on every criterion for each Actor, set the weights, then press Solve.")
        self._fit_window_to_content()

    def _build_stakeholder_tab(self, s):
        tab = ScrollableFrame(self.notebook, self.theme, padding=(20, 12))
        self.notebook.add(tab, text=f"Actor {s + 1}")
        inner = tab.inner

        matrix = Card(
            inner, "Trade-off matrix",
            "Rate how well each alternative performs on each criterion (0 = relatively worst, 100 = relatively best). "
            "Then set how important each criterion is; the criterion weights must add up to 100 %.",
        )
        matrix.pack(fill="x", pady=(0, 16))
        tab.width_listeners.append(matrix.set_available_width)
        grid = matrix.body

        ttk.Label(grid, text="", style="Surface.TLabel").grid(row=0, column=0)
        for c in range(self.n_criteria):
            ttk.Label(grid, text=f"Criterion {c + 1}", style="Bold.TLabel").grid(row=0, column=c + 1, sticky="w", padx=(0, 18), pady=(0, 6))

        score_entries = []
        for r in range(self.n_scenarios):
            ttk.Label(grid, text=f"Alternative {r + 1}", style="Bold.TLabel").grid(row=r + 1, column=0, sticky="w", padx=(0, 18), pady=5)
            row_entries = []
            for c in range(self.n_criteria):
                cell = ValueInput(grid, value=0)
                cell.grid(row=r + 1, column=c + 1, sticky="w", padx=(0, 18), pady=5)
                row_entries.append(cell)
            score_entries.append(row_entries)

        weight_row = self.n_scenarios + 2
        ttk.Separator(grid, orient="horizontal").grid(row=weight_row - 1, column=0, columnspan=self.n_criteria + 1, sticky="ew", pady=(10, 10))
        if self.n_criteria > 1:
            local_chip = ttk.Label(matrix.head, style="Ok.Chip.TLabel")
            local_chip.pack(side="right")
            ttk.Label(grid, text="Criterion weight [%]", style="Bold.TLabel").grid(row=weight_row, column=0, sticky="w", padx=(0, 18), pady=5)
            sub_weight_entries = []
            for c, default in enumerate(equal_split(self.n_criteria)):
                cell = ValueInput(grid, value=default, on_change=lambda s=s: self._update_local_total(s))
                cell.grid(row=weight_row, column=c + 1, sticky="w", padx=(0, 18), pady=5)
                sub_weight_entries.append(cell)
        else:
            local_chip = None
            ttk.Label(grid, text="Criterion weight [%]", style="Bold.TLabel").grid(row=weight_row, column=0, sticky="w", padx=(0, 18), pady=5)
            ttk.Label(grid, text="100  (single criterion)", style="Muted.TLabel").grid(row=weight_row, column=1, sticky="w", pady=6)
            sub_weight_entries = [FixedValue(100)]

        if self.n_stakeholders > 1:
            influence = Card(
                inner, "Actor weight",
                "How much this Actor's judgement counts in the final result. "
                "The weights of all Actors together must add up to 100 %.",
            )
            influence.pack(fill="x")
            tab.width_listeners.append(influence.set_available_width)
            global_chip = ttk.Label(influence.head, style="Ok.Chip.TLabel")
            global_chip.pack(side="right")
            self._global_chips.append(global_chip)
            row = ttk.Frame(influence.body, style="Surface.TFrame")
            row.pack(anchor="w")
            ttk.Label(row, text="Weight [%]", style="Bold.TLabel").pack(side="left", padx=(0, 24))
            global_weight_entry = ValueInput(row, value=equal_split(self.n_stakeholders)[s], length=240, on_change=self._update_global_total)
            global_weight_entry.pack(side="left")
        else:
            global_weight_entry = FixedValue(100)

        self.stakeholder_tables.append({
            "sub_weights": sub_weight_entries,
            "global_weight": global_weight_entry,
            "scores": score_entries,
            "local_chip": local_chip,
        })
        if local_chip is not None:
            self._update_local_total(s)

    # ----- live totals ------------------------------------------------------ #
    @staticmethod
    def _chip_style(total):
        return "Ok.Chip.TLabel" if total == 100 else "Bad.Chip.TLabel"

    def _update_local_total(self, s):
        if s >= len(self.stakeholder_tables):
            return
        table = self.stakeholder_tables[s]
        if table["local_chip"] is None:
            return
        total = sum(w.get() for w in table["sub_weights"])
        table["local_chip"].configure(text=f"Weights total: {total} %", style=self._chip_style(total))

    def _update_global_total(self):
        if self.n_stakeholders <= 1 or not self.stakeholder_tables:
            return
        total = sum(t["global_weight"].get() for t in self.stakeholder_tables)
        style = self._chip_style(total)
        for chip in self._global_chips:
            chip.configure(text=f"All Actors: {total} %", style=style)

    # ----- actions ---------------------------------------------------------- #
    def calculate_rankings(self):
        try:
            scores = [[[cell.get() for cell in row] for row in t["scores"]] for t in self.stakeholder_tables]
            sub_weights = [[w.get() / 100 for w in t["sub_weights"]] for t in self.stakeholder_tables]
            stakeholder_weights = [t["global_weight"].get() / 100 for t in self.stakeholder_tables]

            for i, sw in enumerate(sub_weights):
                if not np.isclose(sum(sw), 1.0):
                    self.notebook.select(i)
                    self.set_status(
                        f"The criterion weights of Actor {i + 1} must add up to 100 % "
                        f"(currently {round(sum(sw) * 100)} %).", error=True,
                    )
                    return
            if not np.isclose(sum(stakeholder_weights), 1.0):
                self.set_status(
                    f"The Actor weights must add up to 100 % "
                    f"(currently {round(sum(stakeholder_weights) * 100)} %).", error=True,
                )
                return

            results = compute_rankings(scores, sub_weights, stakeholder_weights)
        except Exception as exc:  # noqa: BLE001 - surface anything unexpected to the user
            messagebox.showerror("Calculation Error", str(exc))
            return

        self.results_view.show(results)
        self.notebook.select(self.results_view)
        best = [r.name for r in results if r.rank == 1]
        self.set_status("Ranking computed. Best: " + (" and ".join(best) if len(best) <= 2 else f"{len(best)} alternatives tied") + ".")
        self._fit_window_to_content()

    def reset_ratings_only(self):
        for table in self.stakeholder_tables:
            for row in table["scores"]:
                for cell in row:
                    cell.set(0)
            for cell, default in zip(table["sub_weights"], equal_split(len(table["sub_weights"]))):
                cell.set(default)
        for table, default in zip(self.stakeholder_tables, equal_split(len(self.stakeholder_tables))):
            table["global_weight"].set(default)
        if self.results_view is not None:
            self.results_view.clear()
        if self.notebook is not None:
            self.notebook.select(0)
        self.set_status("Scores reset to 0 and weights reset to an equal split.")

    def _confirm_start_over(self):
        if messagebox.askyesno("Start over", "Discard all scores and weights and return to the setup screen?"):
            self.setup_initial_ui()


def _enable_high_dpi():
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa: BLE001 - purely cosmetic
            pass


if __name__ == "__main__":
    _enable_high_dpi()
    root = tk.Tk()
    app = MCDAApp(root)
    root.mainloop()
