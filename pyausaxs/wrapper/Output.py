# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

from pyausaxs.integration import OutputCallback
from pyausaxs.signatures import register
from .AUSAXS import AUSAXS

register({
    "set_output_callback": ([OutputCallback], None),
    "reset_output_callback": ([], None),
})

_active_cb: OutputCallback | None = None
def set_output_callback(python_callable) -> None:
    """Redirect ausaxs stdout to *python_callable(line: str)*."""
    global _active_cb

    def _c_cb(text_bytes: bytes, length: int) -> None:
        if text_bytes:
            python_callable(text_bytes[:length].decode("utf-8", errors="replace"))

    _active_cb = OutputCallback(_c_cb)
    AUSAXS.lib().functions.set_output_callback(_active_cb)


def reset_output_callback() -> None:
    """Restore normal ausaxs stdout."""
    global _active_cb
    AUSAXS.lib().functions.reset_output_callback()
    _active_cb = None
