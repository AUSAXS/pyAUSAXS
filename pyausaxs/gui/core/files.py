# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Recognising the GUI's input files, and guessing one input from another."""

import os
from pathlib import Path
from typing import Callable, Optional

STRUCTURE_EXTENSIONS = {".pdb", ".ent", ".cif", ".xyz"}
SAXS_EXTENSIONS      = {".dat", ".rsr", ".xvg"}
EM_MAP_EXTENSIONS    = {".map", ".ccp4", ".mrc", ".rec"}

# a directory with more entries than this is not searched for a lone companion file (see find_companion)
_COMPANION_SEARCH_LIMIT = 20


def make_validator(extensions: set[str], lib_check: str) -> Callable[[str], bool]:
    """File validator using the AUSAXS library's Filetypes.<lib_check>."""
    def validate(path: str) -> bool:
        if not os.path.isfile(path):
            return False
        from ...wrapper import Filetypes
        return getattr(Filetypes, lib_check)(path)
    return validate


is_structure_file = make_validator(STRUCTURE_EXTENSIONS, "_is_pdb_file")
is_saxs_file = make_validator(SAXS_EXTENSIONS, "_is_saxs_data_file")
is_em_map_file = make_validator(EM_MAP_EXTENSIONS, "_is_em_map_file")


def find_companion(path: str, extensions: set[str]) -> Optional[str]:
    """Guess the companion input of `path` — e.g. the SAXS data belonging to a structure — among files with the given extensions: a file
    sharing its stem, or else the only such file in its directory, provided the directory is small enough that this is likely deliberate.
    Returns the companion's path, or None if there is no unambiguous candidate."""
    if not path:
        return None

    for ext in extensions:
        candidate = str(Path(path).with_suffix(ext))
        if os.path.isfile(candidate):
            return candidate

    directory = os.path.dirname(os.path.abspath(path))
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return None

    if _COMPANION_SEARCH_LIMIT < len(entries):
        return None
    candidates = [e for e in entries if os.path.splitext(e)[1].lower() in extensions]
    return os.path.join(directory, candidates[0]) if len(candidates) == 1 else None
