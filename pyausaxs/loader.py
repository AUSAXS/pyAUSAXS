import os
import sys
import importlib.resources as pkg_resources
from pathlib import Path

from pyausaxs.architecture import get_shared_lib_extension

# environment override: takes precedence over everything, intended for one-off use.
ENV_VAR = "AUSAXS_LIB"


def _config_dir() -> Path:
    """Platform-appropriate user config directory for AUSAXS (dependency-free)."""
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "ausaxs"


def _cache_file() -> Path:
    """File holding the relinked backend path, if any."""
    return _config_dir() / "libpath"


def get_relink_path() -> str | None:
    """Return the cached relink path, or None if none is configured."""
    f = _cache_file()
    if f.is_file():
        path = f.read_text(encoding="utf-8").strip()
        if path:
            return path
    return None


def set_relink_path(path: str) -> Path:
    """
    Persist a custom backend library path. Returns the cache file it was written to.
    Raises FileNotFoundError if the path does not point to an existing file, and
    ValueError if it does not have this platform's shared-library extension.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {p}")

    ext = get_shared_lib_extension()
    if ext and p.suffix != ext:
        raise ValueError(
            f"'{p.name}' is not a shared library for this platform "
            f"(expected a '*{ext}' file)"
        )

    cache = _cache_file()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(str(p), encoding="utf-8")
    return cache


def clear_relink_path() -> bool:
    """Forget any cached relink path. Returns True if one was removed."""
    f = _cache_file()
    if f.is_file():
        f.unlink()
        return True
    return False


def bundled_lib_path() -> str:
    """Path to the shared library shipped inside the package."""
    ext = get_shared_lib_extension()
    lib_file = pkg_resources.files("pyausaxs").joinpath("resources", "libausaxs" + ext)
    with pkg_resources.as_file(lib_file) as p:
        return str(p)


def find_lib_path() -> str:
    """
    Resolve the backend library path, in order of precedence:
      1. the AUSAXS_LIB environment variable,
      2. a path cached via `ausaxs setup --relink`,
      3. the library bundled with the package.
    """
    env = os.environ.get(ENV_VAR)
    if env:
        return env

    cached = get_relink_path()
    if cached:
        return cached

    return bundled_lib_path()
