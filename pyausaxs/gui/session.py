# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""Persistent JSON config in the AUSAXS cache to carry state across GUI sessions."""

import json
import os
import tempfile


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


def _dump_settings_to_config(key: str) -> None:
    """Save the backend's current settings under `key` in the config, via a throwaway temp file (the backend API only writes to a path)."""
    from ..wrapper.settings import settings as backend_settings
    fd, tmp_path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        backend_settings.save(tmp_path)
        with open(tmp_path, errors="replace") as f:
            update_config(**{key: f.read()})
    finally:
        os.unlink(tmp_path)


def _load_settings_from_config(key: str) -> None:
    """Load the backend's settings from `key` in the config, if present, via a throwaway temp file."""
    text = load_config().get(key)
    if not text:
        return
    from ..wrapper.settings import settings as backend_settings
    fd, tmp_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        backend_settings.load(tmp_path)
    finally:
        os.unlink(tmp_path)


def snapshot_default_settings() -> None:
    """Capture the backend's pristine defaults into the config. Call once per launch, before anything else touches the backend."""
    _dump_settings_to_config("settings_defaults")


def reset_settings_to_defaults() -> None:
    """Reset the backend to the defaults captured by `snapshot_default_settings`."""
    _load_settings_from_config("settings_defaults")


class SettingsBackup:
    """Carries the backend's full settings across GUI restarts: restored once at launch, then autosaved periodically."""

    AUTOSAVE_MS = 10_000
    _KEY = "settings"

    def __init__(self, widget):
        self._widget = widget
        self._job = None

    def restore(self):
        """Load the last session's backend settings from the config, if any are stored there."""
        try:
            _load_settings_from_config(self._KEY)
        except Exception:
            pass  # library unavailable, or a stale/corrupt entry: fall back to defaults

    def start_autosave(self):
        """Begin the periodic autosave; call once, after `restore`."""
        self._autosave()

    def stop(self):
        if self._job is not None:
            self._widget.after_cancel(self._job)
            self._job = None

    def _autosave(self):
        try:
            _dump_settings_to_config(self._KEY)
        except Exception:
            pass
        self._job = self._widget.after(self.AUTOSAVE_MS, self._autosave)
