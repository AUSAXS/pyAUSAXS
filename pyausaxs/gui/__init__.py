# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje


def main(argv=None) -> int:
    # imported on demand: the tkinter interface selects the TkAgg matplotlib backend at import, which must not happen merely because
    # another front-end imported the toolkit-independent pyausaxs.gui.core
    from .gui import main as _main
    return _main(argv)


__all__ = ["main"]
