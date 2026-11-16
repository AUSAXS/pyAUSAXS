# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Every entry into the backend starts from the same settings.

The backend's settings are process-wide, and every operation a front-end starts runs in the same process. Left alone, each operation inherits
whatever the previous one left behind: an EM fit turns implicit hydrogens off and changes the atomic radii, a fit flag ticked for one fit stays
on for the next, and a data read widens the q-range. So an operation does not save and restore what it changes. Right before it calls into the
backend, it resets the settings and then applies only the ones it depends on itself (`prepare`).
"""

import os
import tempfile
from typing import Optional

from .loads import RELAXED_LOADS, RelaxedLoads

_defaults: Optional[str] = None  # the settings `reset` returns to, as the text of a settings file


def snapshot_defaults():
    """Capture the backend's current settings as the ones `reset` returns to. A front-end calls this once at start-up, before anything else
    touches the backend; without it, the first `reset` takes the snapshot."""
    global _defaults
    from ...wrapper.settings import settings
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        settings.save(path)
        with open(path, errors="replace") as f:
            _defaults = f.read()
    finally:
        os.unlink(path)


def reset():
    """Return the backend to the settings captured by `snapshot_defaults`."""
    if _defaults is None:
        snapshot_defaults()
        return
    from ...wrapper.settings import settings
    fd, path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(_defaults)
        settings.load(path)
    finally:
        os.unlink(path)


def prepare(*, structure: Optional[str] = None, qrange: Optional[tuple[float, float]] = None, unit: Optional[str] = None,
            relaxed_loads: RelaxedLoads = RELAXED_LOADS):
    """Reset the backend, then apply an operation's own settings: the relaxed load granted for `structure` (see core.loads), the q-range,
    and the q-unit ("A" or "nm"; None keeps the backend's detection). Call right before the operation enters the backend."""
    from ...wrapper.settings import settings
    reset()
    if qrange is not None:
        settings.histogram(qmin=qrange[0], qmax=qrange[1])
    if unit is not None:
        settings.histogram(unit=unit)
    relaxed_loads.apply(structure)
