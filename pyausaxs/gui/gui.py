# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""The AUSAXS graphical interface: SAXS and EM fitting in a single window."""

import os
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")

from .em_pane import EmFitterPane
from .rigidbody_pane import RigidbodyPane
from .saxs_pane import SaxsFitterPane
from .session import load_config, update_config
from .theme import PALETTE, apply_theme


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        from .. import __version__

        self.title("AUSAXS")
        self.geometry("1320x880")
        self.minsize(960, 680)
        apply_theme(self)

        header = ttk.Frame(self, padding=(20, 14, 20, 6))
        header.pack(side="top", fill="x")
        ttk.Label(header, text="AUSAXS", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="small-angle scattering fitter", style="Muted.TLabel").pack(side="left", padx=(12, 0), pady=(6, 0))

        config = load_config()

        notebook = ttk.Notebook(self)
        notebook.pack(side="top", fill="both", expand=True, padx=14, pady=(0, 4))
        titles = []
        for pane_cls in (SaxsFitterPane, EmFitterPane, RigidbodyPane):
            pane = pane_cls(notebook)
            notebook.add(pane, text=pane_cls.title)
            titles.append(pane_cls.title)

        # restore the panel that was open when the GUI was last closed
        if config.get("last_panel") in titles:
            notebook.select(titles.index(config["last_panel"]))

        # remember where we were launched from (so the rigid-body pane can resolve relative
        # script paths next boot) and keep the active panel persisted as the user switches
        update_config(last_launch_directory=os.path.abspath(os.getcwd()))
        notebook.bind(
            "<<NotebookTabChanged>>",
            lambda _e: update_config(last_panel=notebook.tab(notebook.select(), "text")),
        )

        footer = ttk.Frame(self, padding=(20, 6, 20, 10))
        footer.pack(side="bottom", fill="x")
        ttk.Label(footer, text=f"pyAUSAXS {__version__}", style="Footer.TLabel").pack(side="left")
        ttk.Label(footer, text="Kristian Lytje & Jan Skov Pedersen", style="Footer.TLabel").pack(side="right")


def main(argv=None) -> int:
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
