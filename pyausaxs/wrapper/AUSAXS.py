from pyausaxs.integration import AUSAXSLIB
from typing import Union, Optional
import ctypes as ct
import numpy as np
import threading

def _check_array_inputs(*arrays: Union[list, np.ndarray], names: list[str] = None) -> None:
    """Check that all input arrays are either lists or numpy arrays."""
    if names is None:
        names = [f"array_{i}" for i in range(len(arrays))]
    
    for name, arr in zip(names, arrays):
        if not isinstance(arr, (list, np.ndarray)):
            raise TypeError(f"{name} must be a list or numpy array, got {type(arr)} instead.")

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
        if isinstance(arr, list):
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
    _instance = None

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
        return cls._instance._ready

    @classmethod
    def init_error(cls) -> Optional[Exception]:
        """Return the initialization error, if any."""
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
        status = ct.c_int()
        cls.lib().functions.deallocate(object_id, ct.byref(status))
        _check_error_code(status, "deallocate")