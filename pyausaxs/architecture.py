from enum import Enum
from pathlib import Path
import os
import platform

class OS(Enum):
    WIN = 0
    LINUX = 1
    MAC = 2
    UNKNOWN = 3

def get_os():
    if platform.system() == "Windows":  return OS.WIN
    elif platform.system() == "Linux":  return OS.LINUX
    elif platform.system() == "Darwin": return OS.MAC
    return OS.UNKNOWN

def get_cache_dir() -> Path:
    """Return the AUSAXS cache directory, mirroring the C++ backend's path resolution."""
    _os = get_os()
    if _os == OS.WIN:
        base = os.environ.get("LOCALAPPDATA")
    elif _os == OS.MAC:
        base = os.path.join(os.environ.get("HOME", "~"), "Library", "Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.environ.get("HOME", "~"), ".cache")
    return Path(base or ".") / "ausaxs"

def get_shared_lib_extension():
    """
    Get the shared library extension for the current operating system, including the dot.
    If the operating system is unknown, return an empty string.
    """
    _os = get_os()
    if _os == OS.WIN:
        return ".dll"
    elif _os == OS.LINUX:
        return ".so"
    elif _os == OS.MAC:
        return ".dylib"
    return ""
