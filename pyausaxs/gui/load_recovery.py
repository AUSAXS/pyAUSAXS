# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""The tkinter side of recovering from structure loads the backend refuses: a modal offering the relaxations in core.loads, which holds
the grants and the reasoning behind them."""

import tkinter as tk
from tkinter import ttk
from typing import Optional

from .core.loads import LOAD_OPTIONS, RELAXED_LOADS, LoadOption, structure_load_error
from .theme import PALETTE, FONTS


class LoadRecoveryDialog(tk.Toplevel):
    """Modal shown when a structure won't load: the backend's own message, then one button per offered retry.

    Takes a list of options so adding a second one later is a call-site change only."""

    def __init__(self, parent, path: str, message: str, options: list[LoadOption]):
        super().__init__(parent)
        self.title("Could not read structure")
        self.configure(background=PALETTE["bg"])
        self.transient(parent.winfo_toplevel())
        self.chosen: Optional[LoadOption] = None

        body = ttk.Frame(self, padding=14)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=path.rsplit("/", 1)[-1], style="Heading.TLabel").pack(anchor="w")
        ttk.Label(body, text="The backend refused this structure:", style="Muted.TLabel").pack(anchor="w", pady=(6, 4))

        # the backend's message verbatim, in the mono/surface treatment used for script text elsewhere — it is the only account of what is
        # actually wrong, and paraphrasing it would mean guessing at the cause
        readout = tk.Text(body, wrap="word", font=FONTS["mono"], relief="flat", borderwidth=0, height=4, width=64,
                          background=PALETTE["surface"], foreground=PALETTE["text"], padx=8, pady=6)
        readout.insert("1.0", message.strip() or "Unknown error.")
        readout.configure(state="disabled")
        readout.pack(fill="x")

        for option in options:
            ttk.Label(body, text=option.consequence, style="Muted.TLabel", wraplength=460, justify="left").pack(anchor="w", pady=(12, 4))

        buttons = ttk.Frame(self, padding=(14, 0, 14, 14))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right")
        for option in reversed(options):
            ttk.Button(buttons, text=option.label, style="Accent.TButton",
                       command=lambda o=option: self._choose(o)).pack(side="right", padx=(0, 8))

        self.resizable(False, False)
        self.grab_set()
        self.wait_window(self)

    def _choose(self, option: LoadOption):
        self.chosen = option
        self.destroy()


def offer_relaxed_load(parent, path: str, message: str, options: Optional[list[LoadOption]] = None) -> Optional[LoadOption]:
    """Show the recovery dialog for a failed load; returns the chosen option, or None if the user cancelled."""
    return LoadRecoveryDialog(parent, path, message, options or LOAD_OPTIONS).chosen


def ensure_structure_loads(parent, path: str, console=None) -> bool:
    """Whether `path` can be read by the backend right now, offering a relaxed-load retry through RELAXED_LOADS if it can't (see
    core.loads.structure_load_error for how the file is checked). RigidbodyPane checks this itself, built around its own live
    rigidbody-script preview cache."""
    message = structure_load_error(path)
    if message is None:
        return True

    option = offer_relaxed_load(parent, path, message)
    if option is None:
        if console is not None:
            console.append(f"Could not read structure: {message}\n", tag="error")
        return False
    RELAXED_LOADS.grant(path, option)

    message = structure_load_error(path)
    if message is None:
        return True
    if console is not None:
        console.append(f"Still could not read structure: {message}\n", tag="error")
    return False
