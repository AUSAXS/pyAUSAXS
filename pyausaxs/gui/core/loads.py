# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Recovery from structure loads the backend refuses.

Some structures only load once a backend setting is relaxed. Relaxing one silently is not an option: every relaxation trades away accuracy
somewhere, and the user would have no way to know their scattering weights are approximate. So a failed load offers the relaxation as an
explicit retry, and a granted one is remembered *per file*.

This module holds the grants and the load check; asking the user is left to the front-end (see structure_load_error).
"""

import re
from typing import Optional


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

# every relaxation a front-end should offer for a refused structure, in the order to offer them
LOAD_OPTIONS = [ALLOW_UNKNOWN_RESIDUES]


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
        from ...wrapper.settings import settings
        previous = {}
        for name, value in overrides.items():
            previous[name] = self._as_setting_string(settings.get(name))
            settings.set(name, value)
        return previous

    def restore(self, previous: Optional[dict[str, str]]):
        if not previous:
            return
        from ...wrapper.settings import settings
        for name, value in previous.items():
            settings.set(name, value)

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


# Shared across every pane in the process: a grant made anywhere (rigidbody pane, its structure pane, a fitter pane, ...) applies everywhere
# else too, since it is keyed by file path rather than by whichever pane happened to ask first.
RELAXED_LOADS = RelaxedLoads()


def backend_message(err) -> str:
    """Strip the wrapper _check_error_code adds (`AUSAXS: "fn" failed with error code N: "..."`), leaving just the backend's own message.
    Non-matching exceptions (e.g. library-unavailable) are returned unchanged."""
    match = re.match(r'^AUSAXS: ".*?" failed with error code \d+:\s*"(.*)"\s*$', str(err), re.DOTALL)
    return match.group(1) if match else str(err)


def structure_load_error(path: str, relaxed_loads: RelaxedLoads = RELAXED_LOADS) -> Optional[str]:
    """Why the backend refuses to read `path` right now (with any relaxation already granted for it applied), or None if it reads fine.

    Loads the bare file through Molecule, the same plain (non-rigidbody) structure loader the SAXS fitter itself uses — the rigidbody
    sequencer's `load` element belongs to a different subsystem entirely (it also requires a "saxs" argument that has nothing to do with this
    check) and must not be used here. A front-end offering recovery calls this, asks the user to pick one of LOAD_OPTIONS on failure, grants
    it, and calls this again to see whether the retry took."""
    from ...wrapper.Molecule import Molecule
    try:
        with relaxed_loads.applied(path):
            Molecule(path)
        return None
    except Exception as error:
        return backend_message(error)
