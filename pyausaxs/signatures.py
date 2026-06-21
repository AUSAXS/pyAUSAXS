# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""
Lazy ctypes binding for the AUSAXS backend.

Each wrapper module registers the signatures of the backend functions it uses via
``register({...})`` at import time. ``LazyLib`` wraps the loaded ``CDLL`` and applies a
function's ``argtypes``/``restype`` the first time that function is accessed, then caches
it so subsequent calls go straight through.

A signature is therefore only available once its owning module has been imported. That is
safe by construction: a backend function can only be reached through the wrapper that uses
it, and importing that wrapper runs its ``register(...)`` call.
"""

import ctypes as ct

# name -> (argtypes, restype)
_SIGNATURES: dict[str, tuple[list, object]] = {}


def register(signatures: dict[str, tuple[list, object]]) -> None:
    """Register backend function signatures (name -> (argtypes, restype))."""
    _SIGNATURES.update(signatures)


class LazyLib:
    """Wrap a CDLL and apply each function's signature on first access."""

    def __init__(self, cdll: ct.CDLL):
        object.__setattr__(self, "_cdll", cdll)

    def __getattr__(self, name: str):
        try:
            argtypes, restype = _SIGNATURES[name]
        except KeyError:
            raise AttributeError(
                f"AUSAXS: no signature registered for backend function '{name}'"
            )
        func = getattr(self._cdll, name)  # resolves the symbol; raises if missing
        func.argtypes = argtypes
        func.restype = restype
        setattr(self, name, func)  # cache so __getattr__ is not called again
        return func


# Core functions not owned by a single wrapper module: the integration self-test, the
# allocation/error helpers used across wrappers, and the CLI tools dispatched dynamically
# by name from __main__.py and the GUI runner.
register({
    "test_integration":   ([ct.POINTER(ct.c_int)], None),
    "deallocate":         ([ct.c_int, ct.POINTER(ct.c_int)], None),
    "get_last_error_msg": ([ct.POINTER(ct.c_char_p), ct.POINTER(ct.c_int)], None),
    "cli_saxs_fitter":    ([ct.c_int, ct.POINTER(ct.c_char_p)], ct.c_int),
    "cli_em_fitter":      ([ct.c_int, ct.POINTER(ct.c_char_p)], ct.c_int),
    "cli_rigidbody":      ([ct.c_int, ct.POINTER(ct.c_char_p)], ct.c_int),
})
