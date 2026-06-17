# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""A flat, modern look for the AUSAXS GUI, built on the ttk 'clam' theme.

The palette and fonts are exposed as module-level dicts so the hand-drawn widgets
(range slider, log pane, file fields) can match the styled ttk widgets. Call
apply_theme(root) once before building the interface.
"""

import tkinter.font as tkfont
from tkinter import ttk

# Light palette tuned for a calm, modern scientific tool.
PALETTE = {
    "bg":           "#f4f6f9",  # window background
    "surface":      "#ffffff",  # cards, inputs, plots
    "surface_alt":  "#eef1f5",  # subtle raised areas / hover on light controls
    "border":       "#d8dee6",  # hairline borders
    "text":         "#1f2733",  # primary text
    "muted":        "#6b7785",  # secondary text, labels
    "accent":       "#2f7de0",  # primary accent (buttons, selection, fill)
    "accent_hover": "#2569c0",
    "accent_soft":  "#e4eefb",  # accent tint
    "ok":           "#e8f4ec",  # valid input tint
    "ok_border":    "#7fbf95",
    "bad":          "#fdeceb",  # invalid input tint
    "bad_border":   "#e0a4a0",
    "track":        "#cdd5df",  # slider track
    "console_bg":   "#1b1f27",  # log pane
    "console_fg":   "#d7dde5",
}

# ANSI SGR colour code → hex, for the backend's coloured output. Tuned to read
# clearly on the dark console background; codes not listed (0 = reset, bold, ...)
# fall back to the default console foreground. Bright variants (90-96) included.
ANSI_COLORS: dict[int, str] = {
    31: "#ff6b6b",  # red          – warnings
    32: "#7ee787",  # green        – accepted steps
    33: "#e3b341",  # yellow/amber – cautions
    34: "#79c0ff",  # blue
    35: "#d2a8ff",  # magenta
    36: "#56d4dd",  # cyan
    37: "#b1bac4",  # light grey
    90: "#8b949e",  # dark grey
    91: "#ffa198",  # bright red
    92: "#56d364",  # bright green
    93: "#e3b341",  # bright yellow/amber
    94: "#a5d6ff",  # bright blue
    95: "#e2c5ff",  # bright magenta
    96: "#b3f0ff",  # bright cyan
}

# Sequencer-script syntax colours, tuned for the light editor surface. The scope
# list assigns a colour to each nesting depth, cycling once deeper than its length.
SYNTAX = {
    "operation":  "#1565c0",  # blue   – line operations (first token)
    "keyword":    "#8e24aa",  # purple – argument keywords
    "comment":    "#2e7d32",  # green  – # comments
    "error":      "#c62828",  # red    – unrecognised tokens (bold)
    "error_bg":   "#fbe0df",  # soft red tint behind a line with an error
    "scope":      ["#1565c0", "#cc6600", "#990099", "#cc0000"],
    "bracket_bg": "#e4eefb",  # accent tint behind a matched scope pair
}

# Filled in by apply_theme() once a Tk root exists. Each value is a font tuple.
FONTS: dict[str, tuple] = {
    "base":    ("TkDefaultFont", 10),
    "heading": ("TkDefaultFont", 10, "bold"),
    "title":   ("TkDefaultFont", 13, "bold"),
    "small":   ("TkDefaultFont", 8),
    "mono":    ("TkFixedFont", 9),
}


def _resolve_fonts():
    """Derive fonts from the system defaults so we inherit a clean sans wherever we run."""
    sans = tkfont.nametofont("TkDefaultFont").actual("family")
    mono = tkfont.nametofont("TkFixedFont").actual("family")
    FONTS["base"]    = (sans, 10)
    FONTS["heading"] = (sans, 10, "bold")
    FONTS["title"]   = (sans, 14, "bold")
    FONTS["small"]   = (sans, 8)
    FONTS["mono"]    = (mono, 9)


def apply_theme(root):
    """Apply the modern theme to a Tk root and return the ttk.Style object."""
    _resolve_fonts()
    p = PALETTE
    base, heading, title, small = (FONTS["base"], FONTS["heading"], FONTS["title"], FONTS["small"])

    style = ttk.Style(root)
    style.theme_use("clam")

    root.configure(background=p["bg"])
    # classic (non-ttk) widgets such as Text/Entry pick these up
    root.option_add("*Font", base)
    root.option_add("*background", p["bg"])

    style.configure(".",
        background=p["bg"], foreground=p["text"], font=base,
        bordercolor=p["border"], lightcolor=p["bg"], darkcolor=p["bg"],
        troughcolor=p["surface_alt"], focuscolor=p["accent"],
    )

    style.configure("TFrame", background=p["bg"])
    style.configure("Card.TFrame", background=p["surface"])

    style.configure("TLabel", background=p["bg"], foreground=p["text"])
    style.configure("Muted.TLabel", background=p["bg"], foreground=p["muted"])
    style.configure("Heading.TLabel", background=p["bg"], foreground=p["muted"], font=heading)
    style.configure("Title.TLabel", background=p["bg"], foreground=p["text"], font=title)
    style.configure("Footer.TLabel", background=p["bg"], foreground=p["muted"], font=small)

    # labelframes become flat bordered cards with a quiet uppercase-feeling header
    style.configure("TLabelframe",
        background=p["bg"], bordercolor=p["border"], relief="solid", borderwidth=1)
    style.configure("TLabelframe.Label",
        background=p["bg"], foreground=p["muted"], font=heading)

    # default buttons: flat, hairline border, accent on hover
    style.configure("TButton",
        background=p["surface"], foreground=p["text"], font=base,
        relief="flat", borderwidth=1, bordercolor=p["border"], padding=(14, 7))
    style.map("TButton",
        background=[("active", p["surface_alt"]), ("pressed", p["surface_alt"]),
                    ("disabled", p["bg"])],
        bordercolor=[("active", p["accent"]), ("focus", p["accent"])],
        foreground=[("disabled", p["muted"])])

    # primary action button: solid accent
    style.configure("Accent.TButton",
        background=p["accent"], foreground="#ffffff", font=heading,
        relief="flat", borderwidth=0, padding=(18, 8))
    style.map("Accent.TButton",
        background=[("active", p["accent_hover"]), ("pressed", p["accent_hover"]),
                    ("disabled", p["border"])],
        foreground=[("disabled", p["muted"])])

    # icon-sized browse button
    style.configure("Icon.TButton", padding=(8, 6))

    style.configure("TNotebook",
        background=p["bg"], borderwidth=0, tabmargins=(6, 6, 6, 0))
    style.configure("TNotebook.Tab",
        background=p["bg"], foreground=p["muted"], font=base,
        padding=(20, 9), borderwidth=0)
    style.map("TNotebook.Tab",
        background=[("selected", p["bg"]), ("active", p["surface_alt"])],
        foreground=[("selected", p["accent"]), ("active", p["text"])],
        font=[("selected", heading)])

    style.configure("TCheckbutton",
        background=p["bg"], foreground=p["text"], focuscolor=p["bg"])
    style.map("TCheckbutton",
        background=[("active", p["bg"])],
        indicatorcolor=[("selected", p["accent"]), ("!selected", p["surface"])])

    style.configure("TCombobox",
        fieldbackground=p["surface"], background=p["surface"], foreground=p["text"],
        bordercolor=p["border"], arrowcolor=p["muted"], relief="flat", padding=5)
    style.map("TCombobox",
        fieldbackground=[("readonly", p["surface"]), ("disabled", p["bg"])],
        bordercolor=[("focus", p["accent"]), ("active", p["accent"])],
        arrowcolor=[("disabled", p["border"])])
    # the dropdown list is a classic Tk listbox
    root.option_add("*TCombobox*Listbox.background", p["surface"])
    root.option_add("*TCombobox*Listbox.foreground", p["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", p["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    root.option_add("*TCombobox*Listbox.borderWidth", 0)

    style.configure("TEntry",
        fieldbackground=p["surface"], foreground=p["text"],
        bordercolor=p["border"], relief="flat", padding=5)
    style.map("TEntry", bordercolor=[("focus", p["accent"])])

    style.configure("TProgressbar",
        background=p["accent"], troughcolor=p["surface_alt"],
        bordercolor=p["surface_alt"], borderwidth=0, thickness=6)

    style.configure("TScrollbar",
        background=p["track"], troughcolor=p["bg"], bordercolor=p["bg"],
        arrowcolor=p["muted"], relief="flat", borderwidth=0)
    style.map("TScrollbar", background=[("active", p["muted"])])

    style.configure("TPanedwindow", background=p["bg"])
    style.configure("Sash", sashthickness=8, background=p["border"], bordercolor=p["border"])

    return style
