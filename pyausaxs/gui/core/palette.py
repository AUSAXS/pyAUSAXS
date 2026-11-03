# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""The GUI's colour palette, as plain data so both the plots and a toolkit theme can match it."""

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
    "ready":        "#3fa360",  # action button: enough info to click (typed text or a usable selection)
    "ready_hover":  "#358a52",
    "bad":          "#fdeceb",  # invalid input tint
    "bad_border":   "#e0a4a0",
    "danger":       "#cc3b3b",  # destructive-action icon (the reset cross)
    "danger_hover": "#e53935",
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


def blend(a: str, b: str, t: float) -> str:
    """Mix two #rrggbb colours; t=0 gives a, t=1 gives b. Used for fades, where a toolkit offers no opacity."""
    return "#%02x%02x%02x" % tuple(round(int(a[i:i + 2], 16) * (1 - t) + int(b[i:i + 2], 16) * t) for i in (1, 3, 5))
