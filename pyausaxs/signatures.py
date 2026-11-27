# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

"""
Lazy ctypes binding for the AUSAXS backend.

Each wrapper module registers the signatures of the backend functions it uses via ``register({...})`` at import time, with
every argument tagged by its role (input, output array, status, ...), so that ``pyausaxs.wrapper.AUSAXS.call`` can run a
function from its signature alone. ``LazyLib`` wraps the loaded ``CDLL`` and applies a function's ``argtypes``/``restype``
the first time that function is accessed, then caches it so subsequent calls go straight through.
"""

from collections import namedtuple
import ctypes as ct
import numpy as np

class Arg:
    """One argument of a backend function, tagged with its role."""
    __slots__ = ("role", "ctype", "name", "n", "cols")

    def __init__(self, role: str, ctype, name: str | None = None, n: str | int | None = None, cols: int | None = None):
        self.role, self.ctype, self.name, self.n, self.cols = role, ctype, name, n, cols

    @property
    def argtype(self):
        match self.role:
            case "in" | "length": return self.ctype
            case "out_array":     return ct.POINTER(ct.POINTER(self.ctype))
            case _:               return ct.POINTER(self.ctype)

def arg(name: str, ctype) -> Arg:
    """A named scalar input, for a buffer to take its size from. Unnamed inputs are given as their bare ctypes type."""
    return Arg("in", ctype, name)

def array(name: str, ctype=ct.c_double) -> Arg:
    """An input array; any array-like is accepted."""
    return Arg("array", ctype, name)

def length(of: str) -> Arg:
    """The length of the input array `of`, passed automatically."""
    return Arg("length", ct.c_int, n=of)

def out(name: str, ctype) -> Arg:
    """An output scalar. Strings are returned decoded."""
    return Arg("out", ctype, name)

def out_array(name: str, ctype, n: str, cols: int | None = None) -> Arg:
    """An output array allocated by the backend, with as many rows as the output scalar `n`, copied into a numpy array."""
    return Arg("out_array", ctype, name, n, cols)

def buffer(name: str, n: str | int, ctype=ct.c_double) -> Arg:
    """An output array the caller allocates, as long as the named input or of a fixed size."""
    return Arg("buffer", ctype, name, n)

STATUS = Arg("status", ct.c_int, "status") # the status code, checked after the call
HOLDER = object() # restype: the id of a backend object holding the outputs, deallocated once they are copied

class Signature:
    """The role-tagged arguments and the return type of a backend function."""
    __slots__ = ("args", "restype", "holder", "argtypes", "outputs", "result")

    def __init__(self, args: list, restype):
        self.args = [a if isinstance(a, Arg) else Arg("in", a) for a in args]
        self.holder = restype is HOLDER
        self.restype = ct.c_int if self.holder else restype
        self.argtypes = [a.argtype for a in self.args]

        # a call returns the return value, unless it is a holder, and every output that is not just the size of another
        sizes = {a.n for a in self.args if a.role == "out_array"}
        self.outputs = [a for a in self.args if a.role in ("out", "out_array", "buffer") and a.name not in sizes]
        fields = (["result"] if restype not in (None, HOLDER) else []) + [a.name for a in self.outputs]
        self.result = namedtuple("Result", fields) if 1 < len(fields) else None

def copy_array(ptr, n: int, ctype, cols: int | None = None) -> np.ndarray:
    """Copy the `n` rows of a backend-allocated array into an owning numpy array; strings are decoded."""
    count = n*(cols or 1)
    if ctype in (ct.c_char_p, ct.c_char):
        res = np.array([ptr[i].decode("utf-8") for i in range(count)], dtype=np.str_)
    elif count <= 0 or not ptr:
        res = np.empty(0, dtype=np.dtype(ctype))
    else:
        res = np.ctypeslib.as_array(ptr, shape=(count,)).copy()
    return res.reshape(-1, cols) if cols else res

# name -> Signature
_SIGNATURES: dict[str, Signature] = {}

def register(signatures: dict[str, tuple[list, object]]) -> None:
    """Register backend function signatures: name -> (role-tagged arguments, return type or HOLDER)."""
    _SIGNATURES.update({name: Signature(args, restype) for name, (args, restype) in signatures.items()})

class LazyLib:
    """Wrap a CDLL and apply each function's signature on first access."""

    def __init__(self, cdll: ct.CDLL):
        object.__setattr__(self, "_cdll", cdll)

    def __getattr__(self, name: str):
        try:
            signature = _SIGNATURES[name]
        except KeyError:
            raise AttributeError(
                f"AUSAXS: no signature registered for backend function '{name}'"
            )
        func = getattr(self._cdll, name)  # resolves the symbol; raises if missing
        func.argtypes = signature.argtypes
        func.restype = signature.restype
        setattr(self, name, func)  # cache so __getattr__ is not called again
        return func


# Core functions not owned by a single wrapper module: the allocation/error helpers used across wrappers,
# and the backend's `ausaxs` CLI entrypoint used by __main__.py and the GUI runner.
register({
    "deallocate":          ([ct.c_int, STATUS], None),
    "get_last_error_msg":  ([out("msg", ct.c_char_p), STATUS], None),
    "cli_ausaxs":          ([ct.c_int, ct.POINTER(ct.c_char_p)], ct.c_int),
    "cli_ausaxs_add_tool": ([ct.c_char_p, ct.c_char_p], None),
})
