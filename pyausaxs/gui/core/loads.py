# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Recovery from structure loads the backend refuses.

Some structures only load once a load-time setting (allow_unknown_residues, ...) is relaxed. Relaxing one silently is not an option: every
relaxation trades away accuracy somewhere. So a failed load offers the relaxation as an explicit retry; a chosen one is added to the pane's
own settings, which it sets on the backend before each operation like any other.

This module holds the options and the load check; asking the user is left to the front-end (see structure_load_error).
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


def backend_message(err) -> str:
    """Strip the wrapper _check_error_code adds (`AUSAXS: "fn" failed with error code N: "..."`), leaving just the backend's own message.
    Non-matching exceptions (e.g. library-unavailable) are returned unchanged."""
    match = re.match(r'^AUSAXS: ".*?" failed with error code \d+:\s*"(.*)"\s*$', str(err), re.DOTALL)
    return match.group(1) if match else str(err)


def structure_load_error(path: str, backend_settings: dict[str, str]) -> Optional[str]:
    """Why the backend refuses to read `path` with `backend_settings` set on top of the defaults, or None if it reads fine.

    Loads the bare file through Molecule, the same plain (non-rigidbody) structure loader the SAXS fitter itself uses — the rigidbody
    sequencer's `load` element belongs to a different subsystem entirely (it also requires a "saxs" argument that has nothing to do with this
    check) and must not be used here. A front-end offering recovery calls this, asks the user to pick one of LOAD_OPTIONS on failure, adds its
    settings to `backend_settings`, and calls this again to see whether the retry took."""
    from ...wrapper.Molecule import Molecule
    from ...wrapper.settings import settings
    settings.refresh(backend_settings)
    try:
        Molecule(path)
        return None
    except Exception as error:
        return backend_message(error)
