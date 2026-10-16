import importlib
import pkgutil

import pyausaxs
from pyausaxs.signatures import _SIGNATURES
from pyausaxs.wrapper.AUSAXS import AUSAXS


def test_every_registered_function_is_exported():
    """
    Every backend function a wrapper registers must resolve in the loaded library. Symbols resolve lazily on first use,
    so without this a function missing from the library (e.g. not marked API, and hence not exported from a Windows
    DLL) only fails when a user first calls it.
    """
    for module in pkgutil.walk_packages(pyausaxs.__path__, prefix="pyausaxs."):
        if any(s in module.name for s in (".__pyinstaller", ".gui")):
            continue
        importlib.import_module(module.name) # each wrapper registers its signatures at import

    cdll = AUSAXS.lib().functions._cdll
    missing = [name for name in sorted(_SIGNATURES) if not hasattr(cdll, name)]
    assert not missing, "Registered but not exported by the AUSAXS library: " + ", ".join(missing)
