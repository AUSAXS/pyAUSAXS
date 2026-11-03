# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Toolkit-independent logic behind the AUSAXS GUI.

Nothing in this package imports tkinter (or any other GUI toolkit), and nothing here selects a matplotlib backend, so it can be shared
between the bundled tkinter interface and other front-ends embedding the same workflow (e.g. SasView's Qt calculators). The modules are
deliberately not re-exported from here: `script` and `syntax` need only the standard library, while `plotting` pulls in matplotlib.

    script   — reading and rewriting rigid-body sequencer scripts (load block, splits, setup elements, signatures)
    syntax   — tokenizing a script for syntax highlighting and scope matching
    files    — recognising structure / SAXS / EM-map files and finding a file's companion input
    loads    — relaxed-load grants for structures the backend refuses
    palette  — the colour palette shared by the plots and the tkinter theme
    plotting — matplotlib figures for fit results and the 3D structure preview
"""
