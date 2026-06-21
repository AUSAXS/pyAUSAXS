from enum import Enum
from pathlib import Path
import os
import platform
import cpuinfo

class OS(Enum):
    WIN = 0
    LINUX = 1
    MAC = 2
    UNKNOWN = 3

class CPUFeatures:
    """CPU feature detection for AUSAXS library compatibility."""
    
    @staticmethod
    def get_cpu_info():
        try:
            return cpuinfo.get_cpu_info()
        except:
            return {}

    @staticmethod
    def has_sse_support():
        cpu_info = CPUFeatures.get_cpu_info()
        flags = cpu_info.get('flags', [])
        return any('sse' in flag.lower() for flag in flags)

    @staticmethod
    def has_avx_support():
        cpu_info = CPUFeatures.get_cpu_info()
        flags = cpu_info.get('flags', [])
        return 'avx' in [flag.lower() for flag in flags]

    @staticmethod
    def has_avx2_support():
        cpu_info = CPUFeatures.get_cpu_info()
        flags = cpu_info.get('flags', [])
        return 'avx2' in [flag.lower() for flag in flags]

    @staticmethod
    def get_architecture():
        return platform.machine().lower()

    @staticmethod
    def is_compatible_architecture():
        def is_compatible_macos():
            arch = CPUFeatures.get_architecture()
            return arch in ['x86_64', 'amd64', 'arm64']
        
        def is_compatible_linux():
            arch = CPUFeatures.get_architecture()
            return arch in ['x86_64', 'amd64'] and CPUFeatures.has_avx2_support()
        
        def is_compatible_windows():
            arch = CPUFeatures.get_architecture()
            return arch in ['x86_64', 'amd64'] and CPUFeatures.has_avx2_support()

        match get_os():
            case OS.MAC:   return is_compatible_macos()
            case OS.LINUX: return is_compatible_linux()
            case OS.WIN:   return is_compatible_windows()
            case _:        return False

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