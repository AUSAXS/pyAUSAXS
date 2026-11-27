from pyausaxs.integration import AUSAXSLIB
from pyausaxs.signatures import _SIGNATURES, copy_array
from typing import Union, Optional
import ctypes as ct
import numpy as np
import threading

def _check_array_inputs(*arrays: Union[list, np.ndarray], names: list[str] | None = None) -> None:
    """Check that all input arrays are either lists or numpy arrays."""
    if names is None:
        names = [f"array_{i}" for i in range(len(arrays))]
    
    for name, arr in zip(names, arrays):
        if not isinstance(arr, (list, np.ndarray, tuple)):
            raise TypeError(f"{name} must be a list, tuple, or numpy array, got {type(arr)} instead.")

def _check_similar_length(*arrays: Union[list, np.ndarray], msg: str) -> None:
    """Check that all input arrays have the same length."""
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) != 1:
        names = [f"array_{i}" for i in range(len(arrays))]
        raise ValueError(f"{msg}, but got lengths: {dict(zip(names, lengths))}")

def _as_numpy_f64_arrays(*arrays: Union[list, np.ndarray]) -> list[np.ndarray]:
    """Convert all input arrays to numpy arrays of type float64."""
    np_arrays = []
    for arr in arrays:
        if isinstance(arr, list) or isinstance(arr, tuple):
            np_arr = np.array(arr, dtype=np.float64)
        elif isinstance(arr, np.ndarray):
            np_arr = arr.astype(np.float64)
        else:
            raise TypeError(f"Input must be a list or numpy array, got {type(arr)} instead.")
        np_arrays.append(np_arr)
    return np_arrays

def _check_error_code(status: ct.c_int, function_name: str) -> None:
    """Check the status code returned by AUSAXS functions and raise an error if non-zero."""
    if status.value != 0:
        ausaxs = AUSAXS()
        msg = ct.c_char_p()
        status_msg = ct.c_int()
        ausaxs.lib().functions.get_last_error_msg(ct.byref(msg), ct.byref(status_msg))
        if status_msg.value != 0:
            raise RuntimeError(f"AUSAXS: \"{function_name}\" failed with error code {status.value}.")
        error_message = msg.value.decode('utf-8') if msg.value is not None else "Unknown error"
        raise RuntimeError(f"AUSAXS: \"{function_name}\" failed with error code {status.value}: \n\"{error_message}\"")

class AUSAXS:
    _instance: "AUSAXS" = None # type: ignore[assignment]

    def __new__(cls):
        if cls._instance: return cls._instance
        with threading.Lock():
            if cls._instance is None:
                cls._instance = super(AUSAXS, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._lib = None
        self._ready = False
        self._init_error = None
        try:
            self._lib = AUSAXSLIB()
            self._ready = self._lib.ready()
        except Exception as e:
            self._ready = False
            self._init_error = e
        finally:
            self._initialized = True

    @classmethod
    def reset_singleton(cls):
        """Reset the singleton instance."""
        cls._instance = None

    @classmethod
    def ready(cls) -> bool:
        """Check if the AUSAXS library is ready for use."""
        cls()
        return cls._instance._ready

    @classmethod
    def init_error(cls) -> Optional[Exception]:
        """Return the initialization error, if any."""
        cls()
        return cls._instance._init_error

    @classmethod
    def lib(cls) -> AUSAXSLIB:
        """Get the underlying AUSAXSLIB instance."""
        if not cls.ready():
            raise RuntimeError(f"AUSAXS: library failed to initialize. Reason: {cls.init_error()}")
        return cls._instance._lib

    @classmethod
    def deallocate(cls, object_id: int) -> None:
        """Deallocate an object in the AUSAXS library by its ID."""
        if not cls.ready():
            raise RuntimeError(f"AUSAXS: library failed to initialize. Reason: {cls.init_error()}")
        if not isinstance(object_id, int):
            raise TypeError(f"object_id must be of type int, got {type(object_id)} instead.")
        call("deallocate", object_id)

def call(name: str, *inputs):
    """
    Call the backend function `name` with its inputs (its scalar and array arguments, in order) and check its status.
    Returns its outputs as registered in pyausaxs.signatures: nothing, a single value, or a named tuple of several.
    """
    signature = _SIGNATURES[name]
    params = [a for a in signature.args if a.role in ("in", "array")]
    if len(inputs) != len(params):
        raise TypeError(f"AUSAXS: \"{name}\" takes {len(params)} inputs, got {len(inputs)}.")
    values = {a: np.ascontiguousarray(v, dtype=np.dtype(a.ctype)) if a.role == "array" else v for a, v in zip(params, inputs)}
    named = {a.name: v for a, v in values.items() if a.name}

    def size(n) -> int:
        n = named[n] if isinstance(n, str) else n
        return len(n) if hasattr(n, "__len__") else int(n)

    outputs = {}
    def c_arg(a):
        match a.role:
            case "in":
                value = values[a]
                return value.encode("utf-8") if isinstance(value, str) else value
            case "array":     return values[a].ctypes.data_as(a.argtype)
            case "length":    return size(a.n)
            case "buffer":
                outputs[a.name] = np.zeros(size(a.n), dtype=np.dtype(a.ctype))
                return outputs[a.name].ctypes.data_as(a.argtype)
            case "out_array": outputs[a.name] = ct.POINTER(a.ctype)()
            case _:           outputs[a.name] = a.ctype() # out, status
        return ct.byref(outputs[a.name])

    result = getattr(AUSAXS.lib().functions, name)(*[c_arg(a) for a in signature.args])
    if "status" in outputs:
        _check_error_code(outputs["status"], name)

    def copy(a):
        value = outputs[a.name]
        match a.role:
            case "out":
                value = value.value
                return value.decode("utf-8") if isinstance(value, bytes) else value
            case "out_array": return copy_array(value, outputs[a.n].value, a.ctype, a.cols)
            case _:           return value
    try:
        res = ([] if signature.holder or signature.restype is None else [result]) + [copy(a) for a in signature.outputs]
    finally:
        if signature.holder:
            AUSAXS.deallocate(result)
    return signature.result(*res) if signature.result else (res[0] if res else None)
