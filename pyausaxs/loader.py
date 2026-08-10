import importlib.resources as pkg_resources
import os
import sys
from pathlib import Path

from pyausaxs.architecture import OS, get_os, get_shared_lib_extension, get_cache_dir

def _cache_file() -> Path:
    return get_cache_dir() / "libpath"


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
    Raises FileNotFoundError if the path does not point to an existing file, and ValueError if it does not have this platform's shared-library extension.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {p}")

    ext = get_shared_lib_extension()
    if ext and p.suffix != ext:
        raise ValueError(
            f"'{p.name}' is not a valid shared library extension for this platform "
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


def env_var_lib_path() -> str | None:
    """Path set via the AUSAXS_LIB environment variable, or None if unset."""
    return os.environ.get("AUSAXS_LIB") or None


def bundled_lib_path() -> str | None:
    """Path to the shared library shipped inside the package, or None if this installation has none bundled."""
    ext = get_shared_lib_extension()
    lib_file = pkg_resources.files("pyausaxs").joinpath("resources", "libausaxs" + ext)
    try:
        with pkg_resources.as_file(lib_file) as p:
            if p.is_file():
                return str(p)
    except FileNotFoundError:
        pass
    return None


def prefix_lib_path() -> str | None:
    """
    Path to a library installed into the current environment prefix, e.g. by the
    conda libausaxs package, or None if there is none.
    """
    ext = get_shared_lib_extension()
    if get_os() == OS.WIN:
        # conda installs DLLs into Library/bin; the CMake target is named ausaxs,
        # but accept the libausaxs spelling used by the bundled copies as well
        candidates = [
            Path(sys.prefix) / "Library" / "bin" / ("ausaxs" + ext),
            Path(sys.prefix) / "Library" / "bin" / ("libausaxs" + ext),
        ]
    else:
        candidates = [Path(sys.prefix) / "lib" / ("libausaxs" + ext)]

    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def resolve_lib() -> tuple[str, str]:
    """
    Resolve the backend library. Returns (path, origin), where origin is one of
    "relink", "environment", "bundled", or "prefix".
    Raises FileNotFoundError if no library can be located.
    """
    for origin, path in (
        ("relink", get_relink_path()),
        ("environment", env_var_lib_path()),
        ("bundled", bundled_lib_path()),
        ("prefix", prefix_lib_path()),
    ):
        if path:
            return path, origin

    raise FileNotFoundError(
        "could not locate the AUSAXS backend library: this installation has no bundled copy, "
        "no libausaxs is installed in the environment prefix, and neither a relink path nor "
        "the AUSAXS_LIB environment variable is set"
    )


def find_lib_path() -> str:
    """
    Resolve the backend library path.
    """
    return resolve_lib()[0]
