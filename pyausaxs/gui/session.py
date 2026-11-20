# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Persistent JSON config in the AUSAXS cache to carry state across GUI sessions."""

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


def pane_states() -> dict[str, dict]:
    """Each pane's persisted state (see App.save_state), keyed by pane title."""
    states = load_config().get("panes")
    if not isinstance(states, dict):
        return {}
    return {title: state for title, state in states.items() if isinstance(state, dict)}
