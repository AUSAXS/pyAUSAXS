# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Tiny JSON config persisted in the AUSAXS cache to carry a little state across GUI
sessions: the directory the GUI was last launched from (used to resolve relative paths in
the cached rigid-body script) and the panel that was open when it was last closed."""

import json
import os


def _config_path() -> str:
    """Path of the GUI config: <AUSAXS cache>/gui_config.json."""
    from ..architecture import get_cache_dir
    return str(get_cache_dir() / "gui_config.json")


def load_config() -> dict:
    """Return the persisted config, or an empty dict if it is missing or unreadable."""
    path = _config_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, errors="replace") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def update_config(**values) -> None:
    """Merge the given keys into the persisted config, leaving other keys untouched."""
    path = _config_path()
    data = load_config()
    data.update(values)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
