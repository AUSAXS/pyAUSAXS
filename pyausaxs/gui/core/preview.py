# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""The backend's view of a script's structure, for drawing a preview of it: cached for the static preview while the script is edited, and
polled frame by frame for the live view during a run.

Both hand out preview-structure dicts (see Rigidbody.preview_structure) for core.plotting.draw_structure.
"""

from typing import Optional

from . import script as rbscript
from .loads import RELAXED_LOADS, RelaxedLoads, backend_message


class PreviewCache:
    """The preview structure of the script last asked for, rebuilt through the backend only when its preview signature changes, so edits to
    run-time elements (iterations, print, save, ...) cost nothing.

    A script that can't be built (it is usually mid-edit) gives None rather than raising; the reason is kept in `error`, so a deliberate
    action such as Validate/Run can report what the empty preview hides."""

    def __init__(self, relaxed_loads: RelaxedLoads = RELAXED_LOADS):
        self._relaxed_loads = relaxed_loads
        self._key = None
        self.data: Optional[dict] = None            # the last built preview structure, or None
        self.error: Optional[Exception] = None      # why the last build failed, or None

    def get(self, script: str) -> Optional[dict]:
        sig = rbscript.preview_signature(script)
        if sig != self._key:
            self._key = sig
            self.data, self.error = None, None
            path = rbscript.load_value(script, "pdb")
            if path:
                try:
                    from ...wrapper.Rigidbody import Rigidbody
                    with self._relaxed_loads.applied(path):
                        data = Rigidbody(script).preview_structure()
                    self.data = data if len(data["coords"]) else None
                except Exception as e:
                    self.error = e
        return self.data

    def invalidate(self):
        """Force the next get() to rebuild, e.g. once a relaxed load has been granted, which changes nothing the signature can see."""
        self._key = None


def bare_structure_preview(pdb_path: str, saxs_path: Optional[str],
                           relaxed_loads: RelaxedLoads = RELAXED_LOADS) -> Optional[dict]:
    """Preview-structure data for `pdb_path` alone, with no splits or elements staged — a fallback for when a script's own don't apply to it
    any more, e.g. right after the structure was swapped. Carries over the script's `saxs` path, which the `load` element requires and has
    nothing to do with whatever the pdb itself does; without one, there is nothing this can build. Distinct from
    loads.structure_load_error, which checks whether the bare file loads at all through Molecule and deliberately avoids the
    rigidbody-only `load` element and its unrelated requirements."""
    if not saxs_path:
        return None
    block = (f"load {{\n    pdb {rbscript.quote_script_value(pdb_path)}\n"
             f"    saxs {rbscript.quote_script_value(saxs_path)}\n}}\n")
    try:
        from ...wrapper.Rigidbody import Rigidbody
        with relaxed_loads.applied(pdb_path):
            data = Rigidbody(block).preview_structure()
        return data if len(data["coords"]) else None
    except Exception:
        return None


class LiveStructure:
    """The structure a running refinement publishes through its `update structure` elements, frame by frame.

    begin() before starting the run, then poll() from the front-end's timer while it runs and final() once it is done. Frames reuse the
    preview metadata (bodies, copies, Cα mask) built once by begin(): atom order is fixed after setup, so only the coordinates change."""

    POLL_MS = 200  # a suitable interval for the front-end's poll() timer

    def __init__(self):
        self._meta: Optional[dict] = None
        self._version = 0

    @staticmethod
    def register_consumer():
        """Tell the backend a live-structure consumer exists, so `update` elements actually publish. Call once at start-up."""
        from ...wrapper.Rigidbody import Rigidbody
        Rigidbody.set_live_consumer(True)

    @property
    def active(self) -> bool:
        return self._meta is not None

    def begin(self, script: str, relaxed_loads: RelaxedLoads = RELAXED_LOADS) -> Optional[str]:
        """Prepare to follow a run of `script`, if it publishes its structure (contains an `update` element). Must come before the run is
        started, so the run's parse is the last to reset the live buffer. Returns a message if the live view is unavailable, else None;
        check `active` for whether there is anything to follow.

        The build here can fail for the same reasons the run itself will (a refused structure, a malformed script). It is only here to set
        up the live view, so a failure must not take the run down with it — the run reports the problem through its own console."""
        self._meta = None
        self._version = 0
        if not rbscript.has_live_update(script):
            return None
        try:
            from ...wrapper.Rigidbody import Rigidbody
            with relaxed_loads.applied(rbscript.load_value(script, "pdb")):
                meta = Rigidbody(script).preview_structure()
        except Exception as e:
            return f"Live structure view unavailable: {backend_message(e)}"
        if len(meta["coords"]):
            self._meta = meta
        return None

    def _frame(self, coords) -> Optional[dict]:
        if coords is None or len(coords) != len(self._meta["coords"]):
            return None
        return dict(self._meta, coords=coords)  # reuse the mask, swap in the live coordinates

    def poll(self) -> Optional[dict]:
        """The newest published frame, if one has been published since the last poll; None otherwise."""
        if self._meta is None:
            return None
        from ...wrapper.Rigidbody import Rigidbody
        coords, version = Rigidbody.live_structure()
        if version == self._version:
            return None
        frame = self._frame(coords)
        if frame is not None:
            self._version = version
        return frame

    def final(self) -> Optional[dict]:
        """The last published frame, regardless of whether it was already polled, and stop following the run."""
        frame = None
        if self._meta is not None:
            try:
                from ...wrapper.Rigidbody import Rigidbody
                frame = self._frame(Rigidbody.live_structure()[0])
            except Exception:
                pass
        self._meta = None
        return frame
