# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Recovery from structure loads the backend refuses.

Some structures only load once a backend setting is relaxed. Relaxing one silently is not an option: every relaxation trades away accuracy
somewhere, and the user would have no way to know their scattering weights are approximate. So a failed load offers the relaxation as an
explicit retry, and a granted one is remembered *per file* rather than left switched on — the settings are process-wide statics, so flipping
one for a file loaded here would otherwise silently change how every other pane reads every other file.

Currently one relaxation is offered; see temp/rigidbody-load-recovery-plan.md in the ausaxs repo for the intended extensions.
"""

import re
import tkinter as tk
from tkinter import ttk
from typing import Optional

from .theme import PALETTE, FONTS


class LoadOption:
    """One offered way to retry a failed load: a button label, a sentence on what it costs, and the settings it overrides."""

    def __init__(self, label: str, consequence: str, settings: dict[str, str]):
        self.label = label
        self.consequence = consequence
        self.settings = settings


# Skip implicit hydrogens for residues missing from the residue table, instead of refusing the file. The rest of the molecule is unaffected,
# so the result is a mixed model: correct hydrogen content everywhere except those residues, which come out systematically light.
ALLOW_UNKNOWN_RESIDUES = LoadOption(
    "Retry, allowing unknown residues",
    "Residues the backend doesn't recognise keep their explicit atoms but get no implicit hydrogens, so they will scatter slightly too "
    "weakly. The rest of the structure is unaffected.",
    {"allow_unknown_residues": "1"},
)


class RelaxedLoads:
    """Which files the user has granted a relaxed load, and the machinery to apply it around a load.

    Grants are keyed by path and last for the session. Nothing is applied globally: `apply` records the previous values so `restore` can put
    them back, keeping the relaxation confined to the load it was granted for."""

    def __init__(self):
        self._granted: dict[str, dict[str, str]] = {}

    def grant(self, path: str, option: LoadOption):
        self._granted.setdefault(path, {}).update(option.settings)

    def granted(self, path: Optional[str]) -> bool:
        return bool(path) and path in self._granted

    @staticmethod
    def _as_setting_string(value) -> str:
        """`settings.get` returns a typed value but `settings.set` only takes strings, so a value read back for restoring has to be
        re-encoded. Booleans are the only type currently overridden; the rest pass through str()."""
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)

    def apply(self, path: Optional[str]) -> Optional[dict[str, str]]:
        """Set this path's granted overrides, returning the previous values to hand back to `restore` (None if there was nothing to do)."""
        overrides = self._granted.get(path or "")
        if not overrides:
            return None
        from ..wrapper.settings import settings
        previous = {}
        for name, value in overrides.items():
            try:
                previous[name] = self._as_setting_string(settings.get(name))
                settings.set(name, value)
            except Exception:  # an unknown setting name must not take the load down with it
                previous.pop(name, None)
        return previous

    def restore(self, previous: Optional[dict[str, str]]):
        if not previous:
            return
        from ..wrapper.settings import settings
        for name, value in previous.items():
            try:
                settings.set(name, value)
            except Exception:
                pass

    class _Scope:
        def __init__(self, owner: "RelaxedLoads", path: Optional[str]):
            self._owner, self._path, self._previous = owner, path, None

        def __enter__(self):
            self._previous = self._owner.apply(self._path)
            return self

        def __exit__(self, *_exc):
            self._owner.restore(self._previous)
            return False

    def applied(self, path: Optional[str]) -> "_Scope":
        """Context manager applying this path's overrides for the duration of a load, then putting the previous values back."""
        return RelaxedLoads._Scope(self, path)


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
    return LoadRecoveryDialog(parent, path, message, options or [ALLOW_UNKNOWN_RESIDUES]).chosen


# Shared across every pane in the process: a grant made anywhere (rigidbody pane, its structure pane, a fitter pane, ...) applies everywhere
# else too, since it is keyed by file path rather than by whichever pane happened to ask first.
RELAXED_LOADS = RelaxedLoads()


def backend_message(err) -> str:
    """Strip the wrapper _check_error_code adds (`AUSAXS: "fn" failed with error code N: "..."`), leaving just the backend's own message.
    Non-matching exceptions (e.g. library-unavailable) are returned unchanged."""
    match = re.match(r'^AUSAXS: ".*?" failed with error code \d+:\s*"(.*)"\s*$', str(err), re.DOTALL)
    return match.group(1) if match else str(err)


def ensure_structure_loads(parent, path: str, console=None) -> bool:
    """Whether `path` can be read by the backend right now, offering a relaxed-load retry through RELAXED_LOADS if it can't. Loads the bare
    file through Molecule, the same plain (non-rigidbody) structure loader the SAXS fitter itself uses — the rigidbody sequencer's `load`
    element belongs to a different subsystem entirely (it also requires a "saxs" argument that has nothing to do with this check) and must
    not be used here. RigidbodyPane checks this itself, built around its own live rigidbody-script preview cache."""
    from ..wrapper.Molecule import Molecule

    def try_load():
        with RELAXED_LOADS.applied(path):
            Molecule(path)

    try:
        try_load()
        return True
    except Exception as error:
        message = backend_message(error)  # `error` itself is unbound once the except block ends

    option = offer_relaxed_load(parent, path, message)
    if option is None:
        if console is not None:
            console.append(f"Could not read structure: {message}\n", tag="error")
        return False
    RELAXED_LOADS.grant(path, option)

    try:
        try_load()
        return True
    except Exception as error:
        if console is not None:
            console.append(f"Still could not read structure: {backend_message(error)}\n", tag="error")
        return False
