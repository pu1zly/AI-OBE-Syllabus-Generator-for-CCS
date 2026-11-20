"""
gui.py
================================================================================
AI-Powered OBE Syllabus Generator Microservice
Milestone 1 — Desktop GUI (modern redesign)

Standard-library Tkinter only — no extra installs.

The interface uses a sidebar workspace, course and settings cards, a live
generation log, and a results view with clean K/S/A summaries and JSON output.
Fast, Balanced, and Quality modes share the engine's validation rules while
trading off call count and audit depth.

Run:
    python gui.py
================================================================================
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from pdf_export import write_course_summary_pdf
from obe_schemas import BatchGenerationResult
from llm_engine import (
    COURSE_CATALOG,
    LLMProvider,
    OllamaProvider,
    _build_mock_provider,
    apply_generation_preset,
    build_custom_subject,
    find_builtin_subject,
    list_installed_models,
    next_custom_course_code,
    run_batch,
    write_json_atomic,
)

OUTPUT_FILENAME = "sample_validated_output.json"

# ==============================================================================
# THEME
# ==============================================================================
BG = "#F3F5F9"            # app background
SURFACE = "#FFFFFF"       # cards
BORDER = "#E2E7EF"
NAVY = "#19243D"
NAVY_SOFT = "#243653"
NAVY_DEEP = "#121B30"
STEEL = "#4A6382"
MUTED = "#6B7A90"
PERIWINKLE = "#A9B5CC"
SKY = "#E8EDF4"
ACCENT = "#E3A77F"
ACCENT_DARK = "#D18E63"
ACCENT_INK = "#8B5636"
WHITE = "#FFFFFF"
OK_GREEN = "#2F8F6B"
OK_BG = "#E3F4EC"
FAIL_RED = "#C0504D"
FAIL_BG = "#FBE9E8"
WARN_AMBER = "#8A5A10"

# Domain badge colors: (background, foreground)
DOMAIN_COLORS = {
    "Knowledge": ("#E6EDF9", "#2B4C8C"),
    "Skills": ("#E3F4EC", "#2F7D5F"),
    "Attitude": ("#FBEFE4", "#9A5B2E"),
}

FONT = "Segoe UI"


# ==============================================================================
# SMALL REUSABLE WIDGETS
# ==============================================================================
class FlatButton(tk.Label):
    """A flat, hover-aware button built on Label (supports state/text/command)."""

    STYLES = {
        "primary": dict(bg=ACCENT, fg=NAVY, hover=ACCENT_DARK, border=ACCENT,
                        dis_bg=SKY, dis_fg=PERIWINKLE, dis_border=SKY),
        "secondary": dict(bg=SURFACE, fg=NAVY, hover=SKY, border=BORDER,
                          dis_bg=BG, dis_fg=PERIWINKLE, dis_border=BORDER),
        "ghost": dict(bg=SURFACE, fg=STEEL, hover=SKY, border=SURFACE,
                      dis_bg=SURFACE, dis_fg=PERIWINKLE, dis_border=SURFACE),
        "danger": dict(bg=SURFACE, fg=FAIL_RED, hover=FAIL_BG, border=BORDER,
                       dis_bg=BG, dis_fg=PERIWINKLE, dis_border=BORDER),
    }

    def __init__(self, parent, text, command=None, kind="secondary",
                 size="normal", **kw):
        self._s = self.STYLES[kind]
        self._command = command
        self._disabled = False
        self._focused = False
        if size == "large":
            font, padx, pady = (FONT, 11, "bold"), 26, 11
        elif size == "small":
            font, padx, pady = (FONT, 9), 10, 4
        else:
            font, padx, pady = (FONT, 10, "bold" if kind == "primary" else "normal"), 14, 7
        super().__init__(
            parent, text=text, bg=self._s["bg"], fg=self._s["fg"], font=font,
            padx=padx, pady=pady, cursor="hand2", takefocus=True,
            highlightthickness=1, highlightbackground=self._s["border"],
            highlightcolor=ACCENT_DARK, **kw,
        )
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        for key in ("<Return>", "<space>", "<KP_Enter>"):
            self.bind(key, self._on_key_activate)

    def _apply(self, hover=False):
        s = self._s
        if self._disabled:
            super().configure(bg=s["dis_bg"], fg=s["dis_fg"],
                              highlightbackground=ACCENT_DARK if self._focused else s["dis_border"],
                              cursor="arrow")
        else:
            super().configure(bg=s["hover"] if hover else s["bg"], fg=s["fg"],
                              highlightbackground=(
                                  ACCENT_DARK
                                  if self._focused
                                  else s["border"] if not hover else s["hover"]
                              ),
                              cursor="hand2")

    def _on_enter(self, _e):
        if not self._disabled:
            self._apply(hover=True)

    def _on_leave(self, _e):
        self._apply(hover=False)

    def _on_release(self, e):
        if self._disabled or not self._command:
            return
        if 0 <= e.x <= self.winfo_width() and 0 <= e.y <= self.winfo_height():
            self._command()

    def _on_key_activate(self, _e):
        if not self._disabled and self._command:
            self._command()
        return "break"

    def _on_focus_in(self, _e):
        self._focused = True
        self._apply()

    def _on_focus_out(self, _e):
        self._focused = False
        self._apply()

    def configure(self, cnf=None, **kw):
        if "state" in kw:
            self._disabled = kw.pop("state") == "disabled"
            self._apply()
        if "command" in kw:
            self._command = kw.pop("command")
        if cnf or kw:
            super().configure(cnf or {}, **kw)

    config = configure


class NavItem(tk.Frame):
    """Sidebar navigation entry with an accent bar when active."""

    def __init__(self, parent, number, text, command):
        super().__init__(
            parent,
            bg=NAVY,
            cursor="hand2",
            takefocus=True,
            highlightthickness=0,
            highlightbackground=ACCENT,
        )
        self._command = command
        self._active = False
        self.bar = tk.Frame(self, bg=NAVY, width=4)
        self.bar.pack(side="left", fill="y")
        self.body = tk.Frame(self, bg=NAVY)
        self.body.pack(side="left", fill="both", expand=True)
        self.num = tk.Label(self.body, text=number, bg=NAVY, fg=PERIWINKLE,
                            font=(FONT, 9, "bold"), width=2)
        self.num.pack(side="left", padx=(14, 4), pady=12)
        self.lbl = tk.Label(self.body, text=text, bg=NAVY, fg=PERIWINKLE,
                            font=(FONT, 11), anchor="w")
        self.lbl.pack(side="left", fill="x", expand=True)
        for w in (self, self.bar, self.body, self.num, self.lbl):
            w.bind("<Button-1>", lambda _e: self._command())
            w.bind("<Enter>", lambda _e: self._paint(hover=True))
            w.bind("<Leave>", lambda _e: self._paint(hover=False))
        self.bind("<Return>", self._on_key_activate)
        self.bind("<space>", self._on_key_activate)
        self.bind("<FocusIn>", lambda _e: self.configure(highlightthickness=1))
        self.bind("<FocusOut>", lambda _e: self.configure(highlightthickness=0))

    def _on_key_activate(self, _e):
        self._command()
        return "break"

    def _paint(self, hover=False):
        if self._active:
            bg, fg, bar = NAVY_SOFT, WHITE, ACCENT
        elif hover:
            bg, fg, bar = "#1E2C48", WHITE, "#1E2C48"
        else:
            bg, fg, bar = NAVY, PERIWINKLE, NAVY
        self.configure(bg=bg)
        self.bar.configure(bg=bar)
        self.body.configure(bg=bg)
        self.num.configure(bg=bg, fg=ACCENT if self._active else fg)
        self.lbl.configure(bg=bg, fg=fg,
                           font=(FONT, 11, "bold" if self._active else "normal"))

    def set_active(self, active: bool):
        self._active = active
        self._paint()


class Segmented(tk.Frame):
    """Pill-style segmented control bound to a StringVar.

    options: list of (value, label) tuples.
    """

    def __init__(self, parent, variable: tk.StringVar, options, bg=SURFACE):
        super().__init__(parent, bg=SKY, padx=3, pady=3)
        self.var = variable
        self.buttons = {}
        for value, label in options:
            lb = tk.Label(
                self,
                text=label,
                padx=14,
                pady=5,
                cursor="hand2",
                font=(FONT, 9),
                takefocus=True,
                highlightthickness=0,
                highlightbackground=ACCENT_DARK,
            )
            lb.pack(side="left")
            lb.bind("<Button-1>", lambda _e, v=value: self.var.set(v))
            lb.bind("<Return>", lambda _e, v=value: self._activate(v))
            lb.bind("<space>", lambda _e, v=value: self._activate(v))
            lb.bind("<FocusIn>", lambda _e, widget=lb: widget.configure(highlightthickness=1))
            lb.bind("<FocusOut>", lambda _e, widget=lb: widget.configure(highlightthickness=0))
            self.buttons[value] = lb
        self._values = list(self.buttons)
        for index, value in enumerate(self._values):
            self.buttons[value].bind(
                "<Left>",
                lambda _e, i=index: self._focus_option((i - 1) % len(self._values)),
            )
            self.buttons[value].bind(
                "<Right>",
                lambda _e, i=index: self._focus_option((i + 1) % len(self._values)),
            )
        self.var.trace_add("write", lambda *_: self._refresh())
        self._refresh()

    def _activate(self, value):
        self.var.set(value)
        return "break"

    def _focus_option(self, index):
        self.buttons[self._values[index]].focus_set()
        return "break"

    def _refresh(self):
        current = self.var.get()
        for value, lb in self.buttons.items():
            if value == current:
                lb.configure(bg=SURFACE, fg=NAVY, font=(FONT, 9, "bold"))
            else:
                lb.configure(bg=SKY, fg=STEEL, font=(FONT, 9))


class ScrollableFrame(tk.Frame):
    """Vertically scrollable container; put children in `.inner`."""

    def __init__(self, parent, bg=BG):
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, borderwidth=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview,
                                 style="Modern.Vertical.TScrollbar")
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda _e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(
            self._win, width=e.width))
        self.bind_all("<MouseWheel>", self._on_wheel, add="+")

    def _on_wheel(self, e):
        try:
            widget = self.winfo_containing(e.x_root, e.y_root)
        except Exception:
            return
        if widget is None:
            return
        if widget.winfo_class() in ("Treeview", "Text"):
            return
        w = widget
        while w is not None:
            if w is self:
                break
            w = w.master
        else:
            return
        if self.inner.winfo_reqheight() > self.canvas.winfo_height():
            self.canvas.yview_scroll(int(-e.delta / 120), "units")


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================
class OBEGeneratorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CCS OBE Syllabus Generator")
        self.root.geometry("1320x820")
        self.root.minsize(1080, 680)
        self.root.configure(bg=BG)

        self.msg_queue: "queue.Queue[tuple]" = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        self.last_result: Optional[BatchGenerationResult] = None
        self.output_path: Optional[str] = None
        self._run_active = False
        self.result_items: dict[str, dict] = {}
        self.subject_queue: list = []
        self.selected_result_id: Optional[str] = None
        self._result_sequence = 0
        self._stats = {"total": 0, "ok": 0, "fail": 0, "cancelled": 0, "retries": 0}

        self._setup_style()
        self._build_widgets()
        self._detect_model_async()
        self.root.after(100, self._poll_queue)

    # ----------------------------------------------------------- styling ----

    def _setup_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.font_body = tkfont.Font(family=FONT, size=10)
        self.font_bold = tkfont.Font(family=FONT, size=10, weight="bold")
        self.font_small = tkfont.Font(family=FONT, size=9)
        self.font_mono = tkfont.Font(family="Consolas", size=10)
        self.font_mono_small = tkfont.Font(family="Consolas", size=9)

        self.root.option_add("*TCombobox*Listbox.font", (FONT, 10))
        self.root.option_add("*TCombobox*Listbox.background", WHITE)
        self.root.option_add("*TCombobox*Listbox.foreground", NAVY)
        self.root.option_add("*TCombobox*Listbox.selectBackground", SKY)
        self.root.option_add("*TCombobox*Listbox.selectForeground", NAVY)

        style.configure(".", background=BG, foreground=NAVY, font=self.font_body)

        # Inputs ----------------------------------------------------------
        for name in ("TCombobox", "TEntry", "TSpinbox"):
            style.configure(
                name, fieldbackground=WHITE, background=WHITE, foreground=NAVY,
                bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                padding=6, arrowcolor=STEEL, arrowsize=14,
            )
            style.map(
                name,
                bordercolor=[("focus", ACCENT_DARK)],
                lightcolor=[("focus", ACCENT_DARK)],
                darkcolor=[("focus", ACCENT_DARK)],
                fieldbackground=[("readonly", WHITE), ("disabled", BG)],
            )

        # Check buttons ---------------------------------------------------
        style.configure("Card.TCheckbutton", background=SURFACE, foreground=NAVY,
                        focuscolor=SURFACE, font=self.font_body)
        style.map(
            "Card.TCheckbutton",
            background=[("active", SURFACE)],
            indicatorcolor=[("selected", ACCENT_DARK), ("!selected", WHITE)],
        )

        # Progress bar ----------------------------------------------------
        style.configure(
            "Modern.Horizontal.TProgressbar", troughcolor=SKY, background=ACCENT,
            bordercolor=SKY, lightcolor=ACCENT, darkcolor=ACCENT, thickness=8,
        )

        # Treeview --------------------------------------------------------
        try:
            style.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        except tk.TclError:
            pass
        style.configure(
            "Treeview", background=WHITE, fieldbackground=WHITE, foreground=NAVY,
            rowheight=32, borderwidth=0, font=self.font_body,
        )
        style.configure(
            "Treeview.Heading", background=BG, foreground=STEEL,
            font=(FONT, 9, "bold"), relief="flat", borderwidth=0, padding=(8, 7),
        )
        style.map("Treeview.Heading", background=[("active", SKY)])
        style.map(
            "Treeview",
            background=[("selected", "#E6ECF7")],
            foreground=[("selected", NAVY)],
        )

        # Scrollbars (slim, arrow-less) ----------------------------------
        try:
            style.layout(
                "Modern.Vertical.TScrollbar",
                [("Vertical.Scrollbar.trough", {
                    "sticky": "ns",
                    "children": [("Vertical.Scrollbar.thumb",
                                  {"expand": "1", "sticky": "nswe"})],
                })],
            )
        except tk.TclError:
            pass
        style.configure(
            "Modern.Vertical.TScrollbar", troughcolor=SURFACE, background=PERIWINKLE,
            bordercolor=SURFACE, lightcolor=PERIWINKLE, darkcolor=PERIWINKLE,
            gripcount=0, width=10,
        )
        style.map("Modern.Vertical.TScrollbar", background=[("active", STEEL)])
        try:
            style.layout(
                "Modern.Horizontal.TScrollbar",
                [("Horizontal.Scrollbar.trough", {
                    "sticky": "we",
                    "children": [("Horizontal.Scrollbar.thumb",
                                  {"expand": "1", "sticky": "nswe"})],
                })],
            )
        except tk.TclError:
            pass
        style.configure(
            "Modern.Horizontal.TScrollbar", troughcolor=SURFACE, background=PERIWINKLE,
            bordercolor=SURFACE, lightcolor=PERIWINKLE, darkcolor=PERIWINKLE,
            gripcount=0, height=10,
        )
        style.map("Modern.Horizontal.TScrollbar", background=[("active", STEEL)])

    # ---------------------------------------------------------- helpers ----

    def _card(self, parent, title=None, subtitle=None, step=None):
        """White bordered card. Returns (outer_frame, body_frame)."""
        outer = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        if title:
            header = tk.Frame(outer, bg=SURFACE)
            header.pack(fill="x", padx=18, pady=(16, 0))
            if step:
                tk.Label(header, text=str(step), bg=NAVY, fg=WHITE, width=2,
                         font=(FONT, 9, "bold")).pack(side="left", padx=(0, 10))
            tk.Label(header, text=title, bg=SURFACE, fg=NAVY,
                     font=(FONT, 12, "bold")).pack(side="left")
            self._card_header_right = tk.Frame(header, bg=SURFACE)
            self._card_header_right.pack(side="right")
            if subtitle:
                tk.Label(outer, text=subtitle, bg=SURFACE, fg=MUTED, font=self.font_small,
                         anchor="w", justify="left").pack(fill="x", padx=18, pady=(2, 0))
        body = tk.Frame(outer, bg=SURFACE)
        body.pack(fill="both", expand=True, padx=18, pady=(12, 16))
        return outer, body

    @staticmethod
    def _caption(parent, text):
        return tk.Label(parent, text=text, bg=SURFACE, fg=MUTED,
                        font=(FONT, 9), anchor="w")

    @staticmethod
    def _divider(parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=14)

    def _tree_with_scroll(self, parent, columns, height, horizontal=False, fixed_columns=False):
        wrap = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        tree = ttk.Treeview(wrap, columns=columns, show="headings", height=height)
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview,
                            style="Modern.Vertical.TScrollbar")
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        if horizontal:
            hsb = ttk.Scrollbar(
                wrap,
                orient="horizontal",
                command=tree.xview,
                style="Modern.Horizontal.TScrollbar",
            )
            tree.configure(xscrollcommand=hsb.set)
            hsb.pack(side="bottom", fill="x")
        if fixed_columns:
            tree.bind("<Button-1>", self._block_tree_column_resize, add="+")
        tree.pack(side="left", fill="both", expand=True)
        return wrap, tree

    @staticmethod
    def _block_tree_column_resize(event):
        if event.widget.identify_region(event.x, event.y) == "separator":
            return "break"

    def _text_with_scroll(self, parent, bg, fg, font, padx=14, pady=12):
        wrap = tk.Frame(parent, bg=bg)
        text = tk.Text(wrap, wrap="word", font=font, bg=bg, fg=fg, insertbackground=fg,
                       borderwidth=0, highlightthickness=0, padx=padx, pady=pady,
                       selectbackground=STEEL, selectforeground=WHITE)
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=text.yview,
                            style="Modern.Vertical.TScrollbar")
        text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        return wrap, text

    # -------------------------------------------------------------- UI ----

    def _build_widgets(self):
        shell = tk.Frame(self.root, bg=BG)
        shell.pack(fill="both", expand=True)
        self._build_sidebar(shell)

        main = tk.Frame(shell, bg=BG)
        main.pack(side="left", fill="both", expand=True)

        # Top bar --------------------------------------------------------
        topbar = tk.Frame(main, bg=BG)
        topbar.pack(fill="x", padx=28, pady=(22, 14))
        self.page_title_var = tk.StringVar()
        self.page_sub_var = tk.StringVar()
        tk.Label(topbar, textvariable=self.page_title_var, bg=BG, fg=NAVY,
                 font=(FONT, 20, "bold"), anchor="w").pack(anchor="w")
        tk.Label(topbar, textvariable=self.page_sub_var, bg=BG, fg=MUTED,
                 font=(FONT, 10), anchor="w").pack(anchor="w", pady=(2, 0))

        self.page_container = tk.Frame(main, bg=BG)
        self.page_container.pack(fill="both", expand=True, padx=28, pady=(0, 22))
        self.creation_page = tk.Frame(self.page_container, bg=BG)
        self.log_page = tk.Frame(self.page_container, bg=BG)
        self.results_page = tk.Frame(self.page_container, bg=BG)

        self._build_creation_page()
        self._build_log_page()
        self._build_results_page()
        self._show_page("creation")
        self._update_model_chip()

    # ---- Sidebar ---------------------------------------------------------

    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=NAVY, width=236)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        brand = tk.Frame(side, bg=NAVY)
        brand.pack(fill="x", padx=20, pady=(26, 26))
        tk.Label(brand, text="OBE", bg=ACCENT, fg=NAVY, font=(FONT, 11, "bold"),
                 padx=9, pady=5).pack(anchor="w")
        tk.Label(brand, text="Syllabus Generator", bg=NAVY, fg=WHITE,
                 font=(FONT, 14, "bold")).pack(anchor="w", pady=(12, 0))
        tk.Label(brand, text="CCS · Structured outcomes studio", bg=NAVY, fg=PERIWINKLE,
                 font=(FONT, 9)).pack(anchor="w", pady=(2, 0))

        tk.Label(side, text="WORKSPACE", bg=NAVY, fg=STEEL,
                 font=(FONT, 8, "bold")).pack(anchor="w", padx=22, pady=(0, 6))
        self.nav_creation = NavItem(side, "1", "Create batch", lambda: self._show_page("creation"))
        self.nav_creation.pack(fill="x")
        self.nav_log = NavItem(side, "2", "Generation log", lambda: self._show_page("log"))
        self.nav_log.pack(fill="x")
        self.nav_results = NavItem(side, "3", "Results", lambda: self._show_page("results"))
        self.nav_results.pack(fill="x")

        # Model status chip (bottom)
        chip = tk.Frame(side, bg=NAVY_SOFT)
        chip.pack(side="bottom", fill="x", padx=16, pady=18)
        inner = tk.Frame(chip, bg=NAVY_SOFT)
        inner.pack(fill="x", padx=14, pady=12)
        tk.Label(inner, text="ACTIVE MODEL", bg=NAVY_SOFT, fg=PERIWINKLE,
                 font=(FONT, 8, "bold")).pack(anchor="w")
        row = tk.Frame(inner, bg=NAVY_SOFT)
        row.pack(fill="x", pady=(6, 0))
        self.chip_dot = tk.Canvas(row, width=10, height=10, bg=NAVY_SOFT, highlightthickness=0)
        self.chip_dot.pack(side="left", padx=(0, 8))
        self._chip_dot_id = self.chip_dot.create_oval(1, 1, 9, 9, fill=PERIWINKLE, outline="")
        self.chip_text = tk.Label(row, text="Detecting…", bg=NAVY_SOFT, fg=WHITE,
                                  font=(FONT, 10), anchor="w")
        self.chip_text.pack(side="left", fill="x", expand=True)

    def _update_model_chip(self, *_):
        if not hasattr(self, "chip_text"):
            return
        if self.mock_var.get():
            color, text = WARN_AMBER, "Offline mock mode"
        else:
            name = self.model_var.get()
            if name == "Detecting…":
                color, text = PERIWINKLE, "Detecting…"
            elif name in ("", "(none found)"):
                color, text = FAIL_RED, "No model found"
            else:
                color, text = OK_GREEN, name
        self.chip_dot.itemconfigure(self._chip_dot_id, fill=color)
        self.chip_text.configure(text=text)

    def _sync_preset_retry_default(self, *_):
        if not hasattr(self, "retries_var"):
            return
        defaults = {"Fast": 1, "Balanced": 2, "Quality": 3}
        self.retries_var.set(defaults.get(self.preset_var.get(), 2))

    # ---- Page switching --------------------------------------------------

    def _show_page(self, page_name: str):
        self.creation_page.pack_forget()
        self.log_page.pack_forget()
        self.results_page.pack_forget()
        self.nav_creation.set_active(page_name == "creation")
        self.nav_log.set_active(page_name == "log")
        self.nav_results.set_active(page_name == "results")
        if page_name == "results":
            self.results_page.pack(fill="both", expand=True)
            self.page_title_var.set("Results")
            self.page_sub_var.set("Review validated Course Outcomes, grouped by Knowledge, Skills and Attitude.")
        elif page_name == "log":
            self.log_page.pack(fill="both", expand=True)
            self.page_title_var.set("Generation log")
            self.page_sub_var.set("Watch the model work in real time: raw output, validation and auto-repair diffs.")
        else:
            self.creation_page.pack(fill="both", expand=True)
            self.page_title_var.set("Create batch")
            self.page_sub_var.set("Add courses and choose how to generate. The log opens when you run the batch.")

    # ---- Creation page ---------------------------------------------------

    def _build_creation_page(self):
        page = self.creation_page
        page.columnconfigure(0, weight=5, minsize=480)
        page.columnconfigure(1, weight=7)
        page.rowconfigure(0, weight=1)

        left_scroll = ScrollableFrame(page, bg=BG)
        left_scroll.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left = left_scroll.inner

        right = tk.Frame(page, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")

        # Courses live in the scrollable left column; the settings card sits at
        # the top of the right column so it is always visible without scrolling.
        self._build_courses_card(left)
        self._build_settings_card(right)
        self._build_run_card(right)

    def _build_log_page(self):
        page = self.log_page
        self._build_progress_card(page)
        self._build_log_card(page)

    def _build_courses_card(self, parent):
        card, body = self._card(
            parent, "Courses", "Build the batch queue from built-in courses or your own.", step=1)
        card.pack(fill="x", pady=(0, 14), padx=(0, 6))

        # Built-in
        self._caption(body, "Built-in course").pack(fill="x")
        row = tk.Frame(body, bg=SURFACE)
        row.pack(fill="x", pady=(4, 0))
        built_in_labels = [f"{m.course_code} — {m.course_title}" for m in COURSE_CATALOG]
        self.builtin_subject_var = tk.StringVar(value=built_in_labels[0])
        self.builtin_subject_combo = ttk.Combobox(
            row, textvariable=self.builtin_subject_var, values=built_in_labels, state="readonly")
        self.builtin_subject_combo.pack(side="left", fill="x", expand=True)
        FlatButton(row, "Add", self._add_builtin_to_queue, kind="secondary").pack(side="left", padx=(8, 0))

        self.regenerate_description_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(body, text="Regenerate course description with AI",
                        variable=self.regenerate_description_var,
                        style="Card.TCheckbutton").pack(anchor="w", pady=(8, 0))

        self._divider(body)

        # Custom
        self._caption(body, "Custom course").pack(fill="x")
        grid = tk.Frame(body, bg=SURFACE)
        grid.pack(fill="x", pady=(4, 0))
        grid.columnconfigure(0, weight=1)
        self.custom_title_var = tk.StringVar()
        self.custom_code_var = tk.StringVar()
        self.custom_title_entry = ttk.Entry(grid, textvariable=self.custom_title_var)
        self.custom_title_entry.grid(row=0, column=0, sticky="ew")
        code_entry = ttk.Entry(grid, textvariable=self.custom_code_var, width=11)
        code_entry.grid(row=0, column=1, padx=(8, 0))
        tk.Label(grid, text="Course title, e.g. Animal Welfare", bg=SURFACE, fg=PERIWINKLE,
                 font=(FONT, 8), anchor="w").grid(row=1, column=0, sticky="w", pady=(3, 0))
        tk.Label(grid, text="Code (auto-assigned if blank)", bg=SURFACE, fg=PERIWINKLE,
                 font=(FONT, 8), anchor="w").grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(3, 0))
        self.custom_title_entry.bind("<Return>", lambda _e: self._add_custom_to_queue())
        code_entry.bind("<Return>", lambda _e: self._add_custom_to_queue())
        self._caption(body, "Course description (optional)").pack(fill="x", pady=(12, 0))
        description_wrap = tk.Frame(body, bg=SURFACE)
        description_wrap.pack(fill="x", pady=(4, 0))
        self.custom_description_text = tk.Text(
            description_wrap,
            height=4,
            wrap="word",
            font=self.font_body,
            bg=WHITE,
            fg=NAVY,
            insertbackground=NAVY,
            borderwidth=1,
            relief="solid",
            padx=10,
            pady=8,
        )
        self.custom_description_text.pack(fill="x", expand=True)
        self.improve_description_var = tk.BooleanVar(value=False)
        self.improve_description_check = ttk.Checkbutton(
            body,
            text="Improve this description with AI",
            variable=self.improve_description_var,
            style="Card.TCheckbutton",
            state="disabled",
        )
        self.improve_description_check.pack(anchor="w", pady=(8, 0))
        self.custom_description_text.bind(
            "<KeyRelease>",
            self._sync_improve_description_check,
        )
        FlatButton(body, "+  Add custom course", self._add_custom_to_queue,
                   kind="secondary").pack(anchor="w", pady=(10, 0))

        self._divider(body)

        # Queue
        qhead = tk.Frame(body, bg=SURFACE)
        qhead.pack(fill="x")
        tk.Label(qhead, text="Course queue", bg=SURFACE, fg=NAVY,
                 font=(FONT, 10, "bold")).pack(side="left")
        self.queue_count_var = tk.StringVar(value="0 courses loaded")
        self.queue_chip = tk.Label(qhead, textvariable=self.queue_count_var, bg=SKY, fg=STEEL,
                                   font=(FONT, 8, "bold"), padx=8, pady=2)
        self.queue_chip.pack(side="left", padx=(8, 0))
        FlatButton(qhead, "Clear", self._clear_subject_queue, kind="ghost", size="small").pack(side="right")
        FlatButton(qhead, "Remove selected", self._remove_selected_queue_subject,
                   kind="ghost", size="small").pack(side="right", padx=(0, 4))

        tree_wrap, self.subject_queue_tree = self._tree_with_scroll(
            body, ("code", "title", "description"), height=5)
        tree_wrap.pack(fill="x", pady=(8, 0))
        t = self.subject_queue_tree
        t.heading("code", text="Code")
        t.heading("title", text="Course")
        t.heading("description", text="Description")
        t.column("code", width=80, stretch=False)
        t.column("title", width=220, stretch=True)
        t.column("description", width=110, stretch=False)
        self.queue_empty_lbl = tk.Label(
            tree_wrap, text="No courses yet — add one above", bg=WHITE, fg=MUTED, font=self.font_small)
        self.queue_empty_lbl.place(relx=0.5, rely=0.6, anchor="center")

    def _build_settings_card(self, parent):
        card, body = self._card(
            parent, "Generation settings", "Pick a model and how thorough the run should be.", step=2)
        card.pack(fill="x", pady=(0, 14))

        self._caption(body, "Model").pack(fill="x")
        row = tk.Frame(body, bg=SURFACE)
        row.pack(fill="x", pady=(4, 0))
        self.model_var = tk.StringVar(value="Detecting…")
        self.model_combo = ttk.Combobox(row, textvariable=self.model_var, state="readonly")
        self.model_combo.pack(side="left", fill="x", expand=True)
        self.redetect_btn = FlatButton(row, "Re-detect", self._detect_model_async, kind="secondary")
        self.redetect_btn.pack(side="left", padx=(8, 0))

        self._caption(body, "Mode").pack(fill="x", pady=(14, 0))
        self.preset_var = tk.StringVar(value="Balanced")
        self.retries_var = tk.IntVar(value=2)
        Segmented(body, self.preset_var,
                  [("Fast", "Fast"), ("Balanced", "Balanced"), ("Quality", "Quality")]
                  ).pack(anchor="w", pady=(4, 0))
        self.preset_var.trace_add("write", self._sync_preset_retry_default)
        tk.Label(body, text="Fast = quickest demo · Balanced = recommended · Quality = deepest audit",
                 bg=SURFACE, fg=PERIWINKLE, font=(FONT, 8), anchor="w").pack(fill="x", pady=(4, 0))

        nums = tk.Frame(body, bg=SURFACE)
        nums.pack(fill="x", pady=(14, 0))
        for column in range(3):
            nums.columnconfigure(column, weight=1, uniform="n")
        self._caption(nums, "Weeks per course").grid(row=0, column=0, sticky="w")
        self._caption(nums, "Retries").grid(row=0, column=1, sticky="w", padx=(10, 0))
        self._caption(nums, "Timeout (seconds)").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.week_count_var = tk.IntVar(value=8)
        ttk.Spinbox(nums, from_=1, to=12, textvariable=self.week_count_var, width=5).grid(
            row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Spinbox(nums, from_=0, to=10, textvariable=self.retries_var, width=6).grid(
            row=1, column=1, sticky="ew", padx=(10, 0), pady=(4, 0))
        self.timeout_var = tk.IntVar(value=300)
        ttk.Spinbox(nums, from_=60, to=900, increment=30, textvariable=self.timeout_var, width=6).grid(
            row=1, column=2, sticky="ew", padx=(10, 0), pady=(4, 0))
        tk.Label(
            body,
            text="More weeks increase generation time. Choose 1–12.",
            bg=SURFACE,
            fg=MUTED,
            font=(FONT, 8),
            anchor="w",
        ).pack(fill="x", pady=(4, 0))

        tk.Frame(body, bg=BORDER, height=1).pack(fill="x", pady=(10, 8))

        checks = tk.Frame(body, bg=SURFACE)
        checks.pack(fill="x")
        self.debug_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(checks, text="Show raw model output (debug)",
                        variable=self.debug_var, style="Card.TCheckbutton").pack(side="left")
        self.mock_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(checks, text="Offline mock mode",
                        variable=self.mock_var, style="Card.TCheckbutton").pack(side="left", padx=(18, 0))

        self.model_var.trace_add("write", self._update_model_chip)
        self.mock_var.trace_add("write", self._update_model_chip)

    def _build_run_card(self, parent):
        card, body = self._card(parent)
        card.pack(fill="x", pady=(0, 14))

        top = tk.Frame(body, bg=SURFACE)
        top.pack(fill="x")
        text_col = tk.Frame(top, bg=SURFACE)
        text_col.pack(side="left", fill="x", expand=True)
        tk.Label(text_col, text="Run batch", bg=SURFACE, fg=NAVY,
                 font=(FONT, 12, "bold"), anchor="w").pack(anchor="w")
        self.status_var = tk.StringVar(value="Load courses to begin.")
        tk.Label(text_col, textvariable=self.status_var, bg=SURFACE, fg=MUTED,
                 font=self.font_small, anchor="w", justify="left", wraplength=420).pack(anchor="w", pady=(2, 0))

        self.run_btn = FlatButton(top, "Generate outcomes", self._on_run_clicked,
                                  kind="primary", size="large")
        self.run_btn.configure(state="disabled")
        self.run_btn.pack(side="right")

    def _build_progress_card(self, parent):
        card, body = self._card(parent)
        card.pack(fill="x", pady=(0, 14))

        top = tk.Frame(body, bg=SURFACE)
        top.pack(fill="x")
        tk.Label(top, textvariable=self.status_var, bg=SURFACE, fg=NAVY,
                 font=(FONT, 10, "bold"), anchor="w", justify="left",
                 wraplength=700).pack(side="left", fill="x", expand=True)
        self.cancel_btn = FlatButton(top, "Cancel", self._cancel_run, kind="danger")
        self.cancel_btn.configure(state="disabled")
        self.cancel_btn.pack(side="right")

        prog = tk.Frame(body, bg=SURFACE)
        prog.pack(fill="x", pady=(14, 0))
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=len(COURSE_CATALOG),
                                        style="Modern.Horizontal.TProgressbar")
        self.progress.pack(side="left", fill="x", expand=True)
        self.progress_label_var = tk.StringVar(value="0 / 0")
        tk.Label(prog, textvariable=self.progress_label_var, bg=SURFACE, fg=STEEL,
                 font=(FONT, 9, "bold"), width=8, anchor="e").pack(side="left", padx=(10, 0))

    def _build_log_card(self, parent):
        card, body = self._card(parent, "Live generation log",
                                "Raw model output, validation and auto-repair diffs appear here as they happen.")
        card.pack(fill="both", expand=True)
        FlatButton(self._card_header_right, "Clear log", self._clear_log, kind="ghost",
                   size="small").pack()

        wrap = tk.Frame(body, bg=NAVY_DEEP, highlightbackground=NAVY_DEEP, highlightthickness=1)
        wrap.pack(fill="both", expand=True)
        log_frame, self.log_text = self._text_with_scroll(
            wrap, NAVY_DEEP, "#D5DDEA", self.font_mono_small, padx=14, pady=12)
        log_frame.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")
        self.log_text.tag_configure("ok", foreground="#7FD6AE")
        self.log_text.tag_configure("fail", foreground="#F08A86")
        self.log_text.tag_configure("info", foreground=ACCENT)
        self.log_text.tag_configure("raw", foreground=PERIWINKLE)
        self._log_empty_lbl = tk.Label(
            wrap, text="Nothing running yet.\nLogs will stream in here.", bg=NAVY_DEEP, fg=STEEL,
            font=self.font_small, justify="center")
        self._log_empty_lbl.place(relx=0.5, rely=0.5, anchor="center")

    # ---- Results page ----------------------------------------------------

    def _build_results_page(self):
        page = self.results_page

        # KPI row --------------------------------------------------------
        kpis = tk.Frame(page, bg=BG)
        kpis.pack(fill="x", pady=(0, 14))
        self.kpi_vars = {}
        for i, (key, caption, color) in enumerate([
            ("total", "COURSES", NAVY),
            ("ok", "VALIDATED", OK_GREEN),
            ("fail", "FAILED", FAIL_RED),
            ("cancelled", "CANCELLED", WARN_AMBER),
            ("retries", "TOTAL RETRIES", STEEL),
        ]):
            kpis.columnconfigure(i, weight=1, uniform="kpi")
            var = tk.StringVar(value="0")
            self.kpi_vars[key] = var
            box = tk.Frame(kpis, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
            box.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 10, 0))
            tk.Frame(box, bg=color, width=4).pack(side="left", fill="y")
            inner = tk.Frame(box, bg=SURFACE)
            inner.pack(side="left", fill="x", expand=True, padx=16, pady=12)
            tk.Label(inner, text=caption, bg=SURFACE, fg=MUTED,
                     font=(FONT, 8, "bold"), anchor="w").pack(anchor="w")
            tk.Label(inner, textvariable=var, bg=SURFACE, fg=color,
                     font=(FONT, 22, "bold"), anchor="w").pack(anchor="w")

        # Footer actions (packed first so it stays at the bottom) ---------
        footer = tk.Frame(page, bg=BG)
        footer.pack(side="bottom", fill="x", pady=(14, 0))
        self.open_file_btn = FlatButton(footer, "Open output JSON", self._open_output_file, kind="secondary")
        self.open_file_btn.configure(state="disabled")
        self.open_file_btn.pack(side="left")
        self.open_folder_btn = FlatButton(footer, "Open output folder", self._open_output_folder, kind="secondary")
        self.open_folder_btn.configure(state="disabled")
        self.open_folder_btn.pack(side="left", padx=(8, 0))
        self.export_summary_btn = FlatButton(
            footer,
            "Export selected summary PDF",
            self._export_selected_summary_pdf,
            kind="secondary",
        )
        self.export_summary_btn.configure(state="disabled")
        self.export_summary_btn.pack(side="left", padx=(8, 0))
        self.summary_var = tk.StringVar(value="")
        tk.Label(footer, textvariable=self.summary_var, bg=BG, fg=NAVY,
                 font=(FONT, 10, "bold")).pack(side="right")

        # Body ----------------------------------------------------------
        body = tk.Frame(page, bg=BG)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=4, minsize=360)
        body.columnconfigure(1, weight=7)
        body.rowconfigure(0, weight=1)

        # Course list
        list_card, list_body = self._card(body, "Generated courses")
        self.delete_result_btn = FlatButton(
            self._card_header_right,
            "Delete selected",
            self._delete_selected_result,
            kind="danger",
            size="small",
        )
        self.delete_result_btn.configure(state="disabled")
        self.delete_result_btn.pack()
        list_card.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        tree_wrap, self.tree = self._tree_with_scroll(
            list_body,
            ("course", "status", "ksa", "retries"),
            height=10,
            horizontal=True,
            fixed_columns=True,
        )
        tree_wrap.pack(fill="both", expand=True)
        self.tree.heading("course", text="Course")
        self.tree.heading("status", text="Status")
        self.tree.heading("ksa", text="K / S / A")
        self.tree.heading("retries", text="Retries")
        self.tree.column("course", width=240, stretch=False)
        self.tree.column("status", width=96, anchor="center", stretch=False)
        self.tree.column("ksa", width=96, anchor="center", stretch=False)
        self.tree.column("retries", width=76, anchor="center", stretch=False)
        self.tree.bind("<<TreeviewSelect>>", self._on_row_selected)
        self.tree.tag_configure("ok", foreground=OK_GREEN)
        self.tree.tag_configure("fail", foreground=FAIL_RED)
        self.tree.tag_configure("cancelled", foreground=WARN_AMBER)
        self.results_empty_lbl = tk.Label(
            tree_wrap, text="No results yet.\nRun a batch to see courses here.", bg=WHITE,
            fg=MUTED, font=self.font_small, justify="center")
        self.results_empty_lbl.place(relx=0.5, rely=0.45, anchor="center")

        # Output viewer
        out_card, out_body = self._card(body, "Output")
        out_card.grid(row=0, column=1, sticky="nsew")
        self.result_view_var = tk.StringVar(value="Cleaned")
        Segmented(self._card_header_right, self.result_view_var, [
            ("Cleaned", "Summary"),
            ("Selected Course JSON", "Course JSON"),
            ("Full Batch JSON", "Batch JSON"),
        ]).pack()
        detail_wrap = tk.Frame(out_body, bg=WHITE, highlightbackground=BORDER, highlightthickness=1)
        detail_wrap.pack(fill="both", expand=True)
        dframe, self.detail_text = self._text_with_scroll(
            detail_wrap, WHITE, NAVY, self.font_body, padx=18, pady=14)
        dframe.pack(fill="both", expand=True)
        self._configure_detail_tags()
        self._set_detail_placeholder()
        self.result_view_var.trace_add("write", lambda *_a: self._refresh_selected_result())

    def _configure_detail_tags(self):
        t = self.detail_text
        t.tag_configure("code", font=(FONT, 9, "bold"), foreground=ACCENT_INK, spacing1=0, spacing3=0)
        t.tag_configure("title", font=(FONT, 17, "bold"), foreground=NAVY, spacing3=4)
        t.tag_configure("meta", font=(FONT, 9), foreground=MUTED, spacing3=2)
        t.tag_configure("h2", font=(FONT, 9, "bold"), foreground=STEEL, spacing1=18, spacing3=6)
        t.tag_configure("body", font=(FONT, 10), foreground=NAVY, spacing2=3, lmargin1=0, lmargin2=0)
        t.tag_configure("clo", font=(FONT, 10, "bold"), foreground=NAVY, spacing1=10, lmargin1=12, lmargin2=12)
        t.tag_configure("bloom", font=(FONT, 9), foreground=MUTED)
        t.tag_configure("clo_body", font=(FONT, 10), foreground=NAVY, spacing2=3, lmargin1=12, lmargin2=12)
        t.tag_configure("pos", font=(FONT, 9), foreground=STEEL, lmargin1=12, lmargin2=12, spacing3=4)
        t.tag_configure("count", font=(FONT, 9), foreground=MUTED)
        t.tag_configure("empty", font=(FONT, 10, "italic"), foreground=PERIWINKLE, lmargin1=12)
        t.tag_configure("json", font=self.font_mono_small, foreground=NAVY)
        t.tag_configure("placeholder", font=(FONT, 11), foreground=MUTED, spacing1=60, justify="center")
        t.tag_configure("fail_msg", font=(FONT, 10), foreground=FAIL_RED)
        t.tag_configure("cancelled_msg", font=(FONT, 10), foreground=WARN_AMBER)
        for domain, (bg, fg) in DOMAIN_COLORS.items():
            t.tag_configure(f"badge_{domain}", font=(FONT, 9, "bold"), background=bg, foreground=fg)

    def _set_detail_placeholder(self):
        t = self.detail_text
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.insert("1.0", "Select a course on the left to preview its outcomes.", "placeholder")
        t.configure(state="disabled")

    def _update_empty_states(self):
        if self.subject_queue:
            self.queue_empty_lbl.place_forget()
        else:
            self.queue_empty_lbl.place(relx=0.5, rely=0.6, anchor="center")
        if self.tree.get_children():
            self.results_empty_lbl.place_forget()
        else:
            self.results_empty_lbl.place(relx=0.5, rely=0.45, anchor="center")

    def _update_kpis(self):
        for key, var in self.kpi_vars.items():
            var.set(str(self._stats[key]))

    # ----------------------------------------------------- model detection

    def _detect_model_async(self):
        self.model_combo["values"] = []
        self.model_var.set("Detecting…")
        self.redetect_btn.configure(state="disabled")
        threading.Thread(target=self._detect_model_worker, daemon=True).start()

    def _detect_model_worker(self):
        try:
            models = list_installed_models()
            self.msg_queue.put(("models_detected", models))
        except Exception as exc:
            self.msg_queue.put(("model_detect_failed", str(exc)))

    # -------------------------------------------------------- subject queue

    def _refresh_subject_queue(self):
        for item in self.subject_queue_tree.get_children():
            self.subject_queue_tree.delete(item)
        for index, meta in enumerate(self.subject_queue):
            description_source = {
                "manual": "User-written",
                "ai_improved": "AI-improved",
                "generated": "AI generated",
            }.get(meta.description_source, "AI generated")
            self.subject_queue_tree.insert(
                "", "end", iid=str(index),
                values=(meta.course_code, meta.course_title, description_source),
            )
        count = len(self.subject_queue)
        self.queue_count_var.set(f"{count} course{'s' if count != 1 else ''} loaded")
        self.queue_chip.configure(bg=OK_BG if count else SKY, fg=OK_GREEN if count else STEEL)
        self.run_btn.configure(state="normal" if count else "disabled")
        self.progress["maximum"] = max(count, 1)
        self.progress_label_var.set(f"0 / {count}")
        self.status_var.set(
            f"{count} course{'s' if count != 1 else ''} loaded; ready to run."
            if count else "No courses loaded."
        )
        self._update_empty_states()

    def _add_builtin_to_queue(self):
        label = self.builtin_subject_var.get()
        code = label.split(" — ", 1)[0]
        selected = find_builtin_subject(code)
        if selected is None:
            messagebox.showwarning("Course unavailable", "Could not find the selected built-in course.")
            return
        if self.regenerate_description_var.get():
            selected = selected.model_copy(update={"course_description": None})
        if any(meta.course_code == selected.course_code for meta in self.subject_queue):
            messagebox.showinfo("Already loaded", f"{selected.course_code} is already in the queue.")
            return
        self.subject_queue.append(selected)
        self._refresh_subject_queue()

    def _add_custom_to_queue(self):
        title = self.custom_title_var.get().strip()
        if not title:
            messagebox.showwarning("Course title required", "Enter a custom course title.")
            return
        try:
            course_code = self.custom_code_var.get().strip()
            if not course_code:
                course_code = next_custom_course_code(
                    meta.course_code for meta in self.subject_queue
                )
            custom_description = self.custom_description_text.get("1.0", "end").strip()
            improve_description = bool(
                custom_description
                and self.improve_description_var.get()
            )
            subject = build_custom_subject(
                title=title,
                course_code=course_code,
                course_description=custom_description or None,
            )
            subject = subject.model_copy(update={
                "description_source": (
                    "ai_improved"
                    if improve_description
                    else "manual"
                    if subject.course_description
                    else "generated"
                ),
                "improve_description": improve_description,
            })
        except Exception as exc:
            messagebox.showwarning("Invalid course", str(exc))
            return
        self.subject_queue.append(subject)
        self.custom_title_var.set("")
        self.custom_code_var.set("")
        self.custom_description_text.delete("1.0", "end")
        self.improve_description_var.set(False)
        self.improve_description_check.configure(state="disabled")
        self._refresh_subject_queue()

    def _remove_selected_queue_subject(self):
        selection = self.subject_queue_tree.selection()
        if not selection:
            messagebox.showinfo("No course selected", "Select a loaded course to remove.")
            return
        for index in sorted((int(item) for item in selection), reverse=True):
            if 0 <= index < len(self.subject_queue):
                del self.subject_queue[index]
        self._refresh_subject_queue()

    def _clear_subject_queue(self):
        self.subject_queue.clear()
        self._refresh_subject_queue()

    def _sync_improve_description_check(self, _event=None):
        if not hasattr(self, "improve_description_check"):
            return
        has_description = bool(
            self.custom_description_text.get("1.0", "end").strip()
        )
        self.improve_description_check.configure(
            state="normal" if has_description else "disabled"
        )
        if not has_description:
            self.improve_description_var.set(False)

    # ------------------------------------------------------------ run flow

    def _on_run_clicked(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return
        if not self.subject_queue:
            messagebox.showwarning("No courses loaded", "Add at least one built-in or custom course before running.")
            return

        if not self.mock_var.get() and (not self.model_var.get() or self.model_var.get() in ("Detecting…", "(none found)")):
            messagebox.showwarning(
                "No model selected",
                "No local Ollama model was detected. Install Ollama and run "
                "`ollama pull qwen2.5:7b` (or similar), then click Re-detect — "
                "or tick 'Offline mock mode' to try the tool without Ollama.",
            )
            return

        try:
            mock_mode = bool(self.mock_var.get())
            preset = self.preset_var.get()
            retries = self.retries_var.get()
            timeout = float(self.timeout_var.get())
            week_count = self.week_count_var.get()
            if timeout <= 0:
                raise ValueError("Timeout must be greater than zero.")
            model_name = self.model_var.get()
            cfg = apply_generation_preset(
                preset,
                "mock-scripted" if mock_mode else model_name,
                retries=retries,
                week_count=week_count,
            )
            cfg.request_timeout_seconds = timeout
            debug = bool(self.debug_var.get())
        except (ValueError, tk.TclError) as exc:
            messagebox.showwarning("Check generation settings", str(exc))
            return

        self._clear_for_new_run()
        catalog_snapshot = tuple(meta.model_copy(deep=True) for meta in self.subject_queue)
        subject_count = len(catalog_snapshot)
        self.progress["maximum"] = subject_count
        self.progress_label_var.set(f"0 / {subject_count}")
        self.cancel_event.clear()
        self._run_active = True
        self.run_btn.configure(state="disabled", text="Generating…")
        self.cancel_btn.configure(state="normal")
        self.status_var.set(
            f"Generating {subject_count} course{'s' if subject_count != 1 else ''}. "
            "The live log shows progress."
        )
        self.worker_thread = threading.Thread(
            target=self._run_worker,
            args=(catalog_snapshot, cfg, model_name, mock_mode, debug),
            daemon=True,
        )
        self.worker_thread.start()
        self._show_page("log")

    def _cancel_run(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.cancel_event.set()
            self.cancel_btn.configure(state="disabled")
            self.status_var.set("Cancelling — stopping after the current provider call…")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        if hasattr(self, "_log_empty_lbl"):
            self._log_empty_lbl.place(relx=0.5, rely=0.5, anchor="center")

    def _clear_for_new_run(self):
        self.tree.delete(*self.tree.get_children())
        self.result_items.clear()
        self.selected_result_id = None
        self._result_sequence = 0
        self.last_result = None
        self.output_path = None
        self.delete_result_btn.configure(state="disabled")
        self.export_summary_btn.configure(state="disabled")
        self._stats = {"total": 0, "ok": 0, "fail": 0, "cancelled": 0, "retries": 0}
        self._update_kpis()
        self.progress["value"] = 0
        self.summary_var.set("")
        self.open_file_btn.configure(state="disabled")
        self.open_folder_btn.configure(state="disabled")
        self._clear_log()
        self._set_detail_placeholder()
        self._update_empty_states()

    def _run_worker(self, catalog_snapshot, cfg, model_name, mock_mode, debug):
        emit = lambda line: self.msg_queue.put(("log", line))
        on_course_done = lambda meta, outcome: self.msg_queue.put(("course_done", meta, outcome))

        try:
            if mock_mode:
                emit("[*] Running in OFFLINE MOCK mode (no Ollama daemon required).")
                provider: LLMProvider = _build_mock_provider()
            else:
                emit(f"[*] Connecting to local Ollama daemon, model='{model_name}'...")
                provider = OllamaProvider(
                    model_name,
                    cfg.request_timeout_seconds,
                    keep_alive=cfg.keep_alive,
                    num_predict=cfg.num_predict,
                )

            result = run_batch(
                provider,
                cfg,
                catalog=catalog_snapshot,
                debug=debug,
                emit=emit,
                on_course_done=on_course_done,
                is_cancelled=self.cancel_event.is_set,
            )

            out_path = os.path.join(os.getcwd(), OUTPUT_FILENAME)
            write_json_atomic(out_path, result.model_dump_json(indent=2))
            self.msg_queue.put(("batch_done", result, out_path))
        except Exception as exc:
            self.msg_queue.put(("batch_error", str(exc)))

    # --------------------------------------------------------- queue poll

    def _poll_queue(self):
        try:
            while True:
                item = self.msg_queue.get_nowait()
                kind = item[0]

                if kind == "models_detected":
                    models = item[1]
                    self.model_combo["values"] = models
                    self.model_var.set(models[0] if models else "(none found)")
                    self.redetect_btn.configure(state="normal")

                elif kind == "model_detect_failed":
                    self.model_combo["values"] = []
                    self.model_var.set("(none found)")
                    self.redetect_btn.configure(state="normal")
                    self._append_log(f"[!] Could not auto-detect an Ollama model: {item[1]}", tag="fail")

                elif kind == "log":
                    self._append_log(item[1])

                elif kind == "course_done":
                    meta, outcome = item[1], item[2]
                    self._add_result_row(meta, outcome)
                    self.progress["value"] = self.progress["value"] + 1
                    done, total = int(self.progress["value"]), int(self.progress["maximum"])
                    self.progress_label_var.set(f"{done} / {total}")
                    self.status_var.set(f"Finished {done}/{total}: {meta.course_code} {meta.course_title}")

                elif kind == "batch_done":
                    result: BatchGenerationResult = item[1]
                    out_path = item[2]
                    self.last_result = result
                    self.output_path = out_path
                    self._run_active = False
                    self.summary_var.set(
                        f"{self._stats['ok']} validated, {self._stats['fail']} failed, "
                        f"{self._stats['cancelled']} cancelled / "
                        f"{result.total_courses_attempted} attempted")
                    cancelled = self.cancel_event.is_set()
                    self.status_var.set("Batch cancelled." if cancelled else "Batch complete.")
                    self.open_file_btn.configure(state="normal")
                    self.open_folder_btn.configure(state="normal")
                    self.run_btn.configure(state="normal", text="Generate outcomes")
                    self.cancel_btn.configure(state="disabled")
                    self._append_log(
                        f"\n[+] {'Batch cancelled.' if cancelled else 'Batch complete.'} Output saved to: {out_path}",
                        tag="ok")
                    self._show_page("results")

                elif kind == "batch_error":
                    self._run_active = False
                    self.run_btn.configure(state="normal", text="Generate outcomes")
                    self.cancel_btn.configure(state="disabled")
                    self.status_var.set("Batch failed — see log.")
                    self._append_log(f"[!] Batch error: {item[1]}", tag="fail")
                    messagebox.showerror("Batch failed", item[1])

        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll_queue)

    # ------------------------------------------------------------- helpers

    def _append_log(self, text: str, tag: Optional[str] = None):
        self._log_empty_lbl.place_forget()
        if tag is None:
            stripped = text.lstrip()
            if stripped.startswith("[!"):
                tag = "fail"
            elif stripped.startswith("[+"):
                tag = "ok"
            elif stripped.startswith("[*"):
                tag = "info"
            elif stripped.startswith("-----"):
                tag = "raw"
        self.log_text.configure(state="normal")
        if tag:
            self.log_text.insert("end", text + "\n", tag)
        else:
            self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _add_result_row(self, meta, outcome):
        self._stats["total"] += 1
        self._stats["retries"] += int(outcome.audit.retries_used)
        self._result_sequence += 1
        result_id = f"result-{self._result_sequence}"
        cancelled = outcome.error == "Cancelled by user."
        if cancelled:
            ksa_str, status, tag = "—", "CANCELLED", "cancelled"
            self._stats["cancelled"] += 1
            payload = None
            error = "Generation was stopped before this course completed."
        elif outcome.success and outcome.payload is not None:
            ksa = outcome.payload.ksa_summary()
            ksa_str = f"K{ksa['K']}  S{ksa['S']}  A{ksa['A']}"
            status, tag = "✓ OK", "ok"
            self._stats["ok"] += 1
            payload = outcome.payload.model_dump()
            error = None
        else:
            ksa_str, status, tag = "—", "✕ FAILED", "fail"
            self._stats["fail"] += 1
            payload = None
            error = outcome.error or "Generation failed without an error message."

        self.result_items[result_id] = {
            "course_code": meta.course_code,
            "course_title": meta.course_title,
            "payload": payload,
            "audit": outcome.audit.model_dump(),
            "error": error,
            "cancelled": cancelled,
        }

        self.tree.insert(
            "", "end", iid=result_id,
            values=(f"{meta.course_code}  {meta.course_title}", status, ksa_str, outcome.audit.retries_used),
            tags=(tag,),
        )
        self._update_kpis()
        self._update_empty_states()

    def _on_row_selected(self, _event=None):
        selection = self.tree.selection()
        if not selection:
            self.selected_result_id = None
            self._update_delete_button_state()
            self._update_export_button_state()
            return
        self.selected_result_id = selection[0]
        self._update_delete_button_state()
        self._update_export_button_state()
        self._refresh_selected_result()

    def _update_delete_button_state(self):
        if not hasattr(self, "delete_result_btn"):
            return
        can_delete = (
            bool(self.tree.selection())
            and self.last_result is not None
            and not self._run_active
        )
        self.delete_result_btn.configure(state="normal" if can_delete else "disabled")

    def _update_export_button_state(self):
        if not hasattr(self, "export_summary_btn"):
            return
        selection = self.tree.selection()
        result_item = self.result_items.get(selection[0]) if selection else None
        can_export = (
            result_item is not None
            and result_item["payload"] is not None
            and not self._run_active
        )
        self.export_summary_btn.configure(state="normal" if can_export else "disabled")

    def _export_selected_summary_pdf(self):
        selection = self.tree.selection()
        result_item = self.result_items.get(selection[0]) if selection else None
        if result_item is None or result_item["payload"] is None or self._run_active:
            return

        course_code = str(result_item["course_code"])
        safe_code = "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in course_code
        )
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export course summary as PDF",
            initialfile=f"{safe_code}_summary.pdf",
            defaultextension=".pdf",
            filetypes=(("PDF documents", "*.pdf"),),
        )
        if not path:
            return
        try:
            write_course_summary_pdf(path, result_item["payload"])
        except Exception as exc:
            messagebox.showerror("Could not export PDF", str(exc), parent=self.root)
            return
        messagebox.showinfo("PDF exported", f"Summary saved to:\n{path}", parent=self.root)

    def _delete_selected_result(self):
        selection = self.tree.selection()
        if not selection or not self.last_result or self._run_active:
            return
        result_id = selection[0]
        result_item = self.result_items.get(result_id)
        if result_item is None:
            return
        if not messagebox.askyesno(
            "Delete generated course",
            f"Delete {result_item['course_code']} — {result_item['course_title']} "
            "from this batch and update the saved JSON?",
            parent=self.root,
        ):
            return

        records = list(self.last_result.records)
        failures = [dict(failure) for failure in self.last_result.failures]
        if result_item["payload"] is not None:
            match_index = next(
                (
                    index for index, record in enumerate(records)
                    if record.payload.course_code == result_item["course_code"]
                    and record.payload.course_title == result_item["course_title"]
                    and record.payload.model_dump() == result_item["payload"]
                    and record.audit.generated_at == result_item["audit"].get("generated_at")
                ),
                None,
            )
            if match_index is None:
                messagebox.showerror(
                    "Could not delete course",
                    "The selected result could not be matched to the saved batch.",
                    parent=self.root,
                )
                return
            records.pop(match_index)
        elif not result_item["cancelled"]:
            match_index = next(
                (
                    index for index, failure in enumerate(failures)
                    if failure.get("course_code") == result_item["course_code"]
                    and failure.get("course_title") == result_item["course_title"]
                    and (failure.get("error") or "Generation failed without an error message.")
                    == result_item["error"]
                ),
                None,
            )
            if match_index is None:
                messagebox.showerror(
                    "Could not delete course",
                    "The selected failure could not be matched to the saved batch.",
                    parent=self.root,
                )
                return
            failures.pop(match_index)

        updated_result = self.last_result.model_copy(
            update={
                "records": records,
                "failures": failures,
                "total_courses_attempted": max(self.last_result.total_courses_attempted - 1, 0),
                "total_courses_validated": len(records),
            },
            deep=True,
        )
        try:
            output_path = self.output_path or os.path.join(os.getcwd(), OUTPUT_FILENAME)
            write_json_atomic(output_path, updated_result.model_dump_json(indent=2))
        except Exception as exc:
            messagebox.showerror(
                "Could not update saved JSON",
                f"The course was not deleted because the output file could not be updated:\n{exc}",
                parent=self.root,
            )
            return

        self.last_result = updated_result
        self.output_path = output_path
        self.tree.delete(result_id)
        self.result_items.pop(result_id, None)
        self._stats["total"] = max(self._stats["total"] - 1, 0)
        self._stats["retries"] = max(
            self._stats["retries"] - int(result_item["audit"].get("retries_used", 0)),
            0,
        )
        if result_item["cancelled"]:
            self._stats["cancelled"] = max(self._stats["cancelled"] - 1, 0)
        elif result_item["payload"] is not None:
            self._stats["ok"] = max(self._stats["ok"] - 1, 0)
        else:
            self._stats["fail"] = max(self._stats["fail"] - 1, 0)

        self.selected_result_id = None
        self._update_kpis()
        self.summary_var.set(
            f"{self._stats['ok']} validated, {self._stats['fail']} failed, "
            f"{self._stats['cancelled']} cancelled / "
            f"{updated_result.total_courses_attempted} attempted"
        )
        self._update_empty_states()
        self._update_delete_button_state()
        self._update_export_button_state()
        self._refresh_selected_result()

    def _refresh_selected_result(self):
        view = self.result_view_var.get()
        result_id = self.selected_result_id
        if not result_id:
            selection = self.tree.selection()
            result_id = selection[0] if selection else None

        t = self.detail_text
        t.configure(state="normal")
        t.delete("1.0", "end")

        if view == "Full Batch JSON":
            if self.last_result:
                t.insert("1.0", self.last_result.model_dump_json(indent=2), "json")
            else:
                t.insert("1.0", "No full batch output is available yet.", "placeholder")
        elif not result_id:
            t.insert("1.0", "Select a course on the left to preview its outcomes.", "placeholder")
        else:
            result_item = self.result_items.get(result_id)
            if result_item is None:
                t.insert("1.0", "This result is no longer available.", "placeholder")
            elif result_item["error"]:
                heading = (
                    f"{result_item['course_code']}  |  {result_item['course_title']}\n\n"
                    + ("Generation cancelled\n" if result_item["cancelled"] else "Generation failed\n")
                )
                tag = "cancelled_msg" if result_item["cancelled"] else "fail_msg"
                t.insert("1.0", heading, tag)
                t.insert("end", result_item["error"], tag)
            elif view == "Selected Course JSON":
                t.insert("1.0", json.dumps(result_item["payload"], indent=2), "json")
            else:
                self._render_clean(result_item["payload"], result_item["audit"])
        t.configure(state="disabled")

    def _render_clean(self, payload: dict, audit: Optional[dict]):
        t = self.detail_text
        t.insert("end", f"{payload.get('course_code', '—')}\n", "code")
        t.insert("end", f"{payload.get('course_title', '—')}\n", "title")
        pos = ", ".join(f"PO{po}" for po in payload.get("target_pos", [])) or "—"
        t.insert(
            "end",
            f"Year {payload.get('year_level', '—')}  ·  Semester {payload.get('semester', '—')}"
            f"  ·  Target POs: {pos}\n",
            "meta",
        )

        t.insert("end", "COURSE DESCRIPTION\n", "h2")
        t.insert("end", f"{payload.get('course_description', '—')}\n", "body")

        t.insert("end", "COURSE OUTCOMES\n", "h2")
        outcomes = payload.get("course_outcomes", [])
        labels = {"Knowledge": "KNOWLEDGE  K", "Skills": "SKILLS  S", "Attitude": "ATTITUDE  A"}
        for domain in ("Knowledge", "Skills", "Attitude"):
            items = [o for o in outcomes if o.get("domain") == domain]
            t.insert("end", "\n")
            t.insert("end", f"  {labels[domain]}  ", f"badge_{domain}")
            t.insert("end", f"   {len(items)} outcome{'s' if len(items) != 1 else ''}\n", "count")
            if not items:
                t.insert("end", "No outcomes.\n", "empty")
                continue
            for o in items:
                mapped = "  ·  ".join(f"PO{po}" for po in o.get("mapped_po", [])) or "—"
                t.insert("end", f"CLO {o.get('clo_number', '—')}", "clo")
                t.insert("end", f"   {o.get('bloom_level', '—')}\n", "bloom")
                t.insert("end", f"{o.get('co_description', '—')}\n", "clo_body")
                t.insert("end", f"Mapped to  {mapped}\n", "pos")

        weekly_schedule = payload.get("weekly_schedule", [])
        if weekly_schedule:
            t.insert("end", "WEEKLY LEARNING PLAN\n", "h2")
            for week in weekly_schedule:
                number = week.get("week_number", "—")
                t.insert(
                    "end",
                    f"Week {number}  ·  {week.get('topic', '—')}\n",
                    "clo",
                )
                t.insert(
                    "end",
                    f"Learning outcome: {week.get('learning_outcome', '—')}\n",
                    "clo_body",
                )
                for activity in week.get("activities", []):
                    t.insert("end", f"Activity: {activity}\n", "body")
                mapped = ", ".join(f"CLO {clo}" for clo in week.get("mapped_clos", [])) or "—"
                t.insert(
                    "end",
                    f"Mapped CLOs: {mapped}  ·  Contact hours: {week.get('contact_hours', 3.0)}\n",
                    "pos",
                )

        if audit:
            t.insert("end", "GENERATION DETAILS\n", "h2")
            t.insert(
                "end",
                f"Elapsed {audit.get('elapsed_seconds', 0):.2f}s  ·  "
                f"Provider calls {audit.get('provider_calls', 0)}  ·  "
                f"Retries {audit.get('retries_used', 0)}\n",
                "meta",
            )

    @staticmethod
    def _format_clean_result(payload: dict) -> str:
        """Plain-text version of the cleaned view (kept for compatibility/export)."""
        lines = [
            f"{payload.get('course_code', '—')}  |  {payload.get('course_title', '—')}",
            "=" * 72,
            f"Year / Semester: {payload.get('year_level', '—')} / {payload.get('semester', '—')}",
            f"Target POs: {', '.join(f'PO{po}' for po in payload.get('target_pos', [])) or '—'}",
            "",
            "COURSE DESCRIPTION",
            "-" * 72,
            payload.get("course_description", "—"),
            "",
            "COURSE OUTCOMES",
            "-" * 72,
        ]
        domain_names = {"Knowledge": "KNOWLEDGE (K)", "Skills": "SKILLS (S)", "Attitude": "ATTITUDE (A)"}
        outcomes = payload.get("course_outcomes", [])
        for domain in ("Knowledge", "Skills", "Attitude"):
            lines += ["", domain_names[domain], "-" * 72]
            domain_outcomes = [o for o in outcomes if o.get("domain") == domain]
            if not domain_outcomes:
                lines.append("No outcomes.")
                continue
            for o in domain_outcomes:
                mapped = ", ".join(f"PO{po}" for po in o.get("mapped_po", [])) or "—"
                lines.append(f"CLO {o.get('clo_number', '—')}  —  {o.get('bloom_level', '—')}")
                lines.append(o.get("co_description", "—"))
                lines.append(f"Mapped POs: {mapped}")
                lines.append("")
        weekly_schedule = payload.get("weekly_schedule", [])
        if weekly_schedule:
            lines += ["", "WEEKLY LEARNING PLAN", "-" * 72]
            for week in weekly_schedule:
                lines.append(f"Week {week.get('week_number', '—')}: {week.get('topic', '—')}")
                lines.append(f"Learning outcome: {week.get('learning_outcome', '—')}")
                lines.extend(f"Activity: {activity}" for activity in week.get("activities", []))
                mapped = ", ".join(f"CLO {clo}" for clo in week.get("mapped_clos", [])) or "—"
                lines.append(f"Mapped CLOs: {mapped}")
                lines.append(f"Contact hours: {week.get('contact_hours', 3.0)}")
                lines.append("")
        return "\n".join(lines).rstrip()

    def _open_output_file(self):
        self._open_path(os.path.join(os.getcwd(), OUTPUT_FILENAME))

    def _open_output_folder(self):
        self._open_path(os.getcwd())

    @staticmethod
    def _open_path(path: str):
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            messagebox.showerror("Could not open", str(exc))


def main():
    # Crisp text on high-DPI Windows displays.
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
    except Exception:
        pass
    root = tk.Tk()
    OBEGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
