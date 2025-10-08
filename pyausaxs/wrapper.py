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

class AUSAXS:
    """
    AUSAXS Python wrapper for the C++ library.
    Implemented as a singleton to avoid expensive reinitialization.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
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

    def ready(self) -> bool:
        """Check if the AUSAXS library is ready for use."""
        return self._ready

    def init_error(self) -> Optional[Exception]:
        """Return the initialization error if any."""
        return self._init_error
    
    @classmethod
    def reset_singleton(cls):
        """Reset the singleton instance. Useful for testing or debugging."""
        with cls._lock:
            cls._instance = None

    def debye(
            self, q_vector: Union[list[float], np.ndarray], 
            atom_x: Union[list[float], np.ndarray], atom_y: Union[list[float], np.ndarray], atom_z: Union[list[float], np.ndarray], 
            weights: Union[list[float], np.ndarray]
        ):
        """
        Compute the Debye scattering intensity I(q) for given q values and atomic coordinates.
        No form factors or excluded volume effects are considered; only the pure Debye formula is evaluated. 
        """
        # input validation
        if not self.ready():
            raise RuntimeError(f"AUSAXS: library failed to initialize. Reason: {self.init_error()}")
        _check_array_inputs(q_vector, atom_x, atom_y, atom_z, weights)
        _check_similar_length(atom_x, atom_y, atom_z, weights, msg="Atomic coordinates and weights must have the same length")
        q_vector, atom_x, atom_y, atom_z, weights = _as_numpy_f64_arrays(q_vector, atom_x, atom_y, atom_z, weights)

        # prepare ctypes args
        Iq = (ct.c_double * len(q_vector))()
        nq = ct.c_int(len(q_vector))
        nc = ct.c_int(len(weights))
        q = q_vector.ctypes.data_as(ct.POINTER(ct.c_double))
        x = atom_x.ctypes.data_as(ct.POINTER(ct.c_double))
        y = atom_y.ctypes.data_as(ct.POINTER(ct.c_double))
        z = atom_z.ctypes.data_as(ct.POINTER(ct.c_double))
        w = weights.ctypes.data_as(ct.POINTER(ct.c_double))
        status = ct.c_int()
        self._lib.functions.evaluate_sans_debye(q, x, y, z, w, nq, nc, Iq, ct.byref(status))

        if (status.value == 0):
            # convert ctypes array to numpy array
            arr = np.ctypeslib.as_array(Iq)
            return arr.copy()
        raise RuntimeError(f"AUSAXS: \"debye\" terminated unexpectedly (error code \"{status.value}\").")

    def fit(self, q, I, Ierr, x, y, z, names, resnames, elements) -> np.ndarray:
        """
        Perform automatic SAXS fitting and return the optimal I(q).
        This is the main fitting method most users should use.
        """
        if not self.ready():
            raise RuntimeError(f"AUSAXS: library failed to initialize. Reason: {self.init_error()}")

        _check_array_inputs(
            q, I, Ierr, x, y, z, 
            names=['q', 'I', 'Ierr', 'x', 'y', 'z']
        )
        _check_similar_length(x, y, z, names, resnames, elements, msg="Atomic coordinates and names must have the same length")
        _check_similar_length(q, I, Ierr, msg="q, I, and Ierr must have the same length")
        q, I, Ierr, x, y, z = _as_numpy_f64_arrays(q, I, Ierr, x, y, z)
        
        nq = ct.c_int(len(q))
        nc = ct.c_int(len(x))
        Iq = (ct.c_double * len(q))()
        q_ptr = q.ctypes.data_as(ct.POINTER(ct.c_double))
        I_ptr = I.ctypes.data_as(ct.POINTER(ct.c_double))
        Ierr_ptr = Ierr.ctypes.data_as(ct.POINTER(ct.c_double))
        x_ptr = x.ctypes.data_as(ct.POINTER(ct.c_double))
        y_ptr = y.ctypes.data_as(ct.POINTER(ct.c_double))
        z_ptr = z.ctypes.data_as(ct.POINTER(ct.c_double))
        names_ptr = (ct.c_char_p * len(names))(*[s.encode('utf-8') for s in names])
        resnames_ptr = (ct.c_char_p * len(resnames))(*[s.encode('utf-8') for s in resnames])
        elements_ptr = (ct.c_char_p * len(elements))(*[s.encode('utf-8') for s in elements])
        status = ct.c_int()

        self._lib.functions.fit_saxs(q_ptr, I_ptr, Ierr_ptr, nq, x_ptr, y_ptr, z_ptr, names_ptr, resnames_ptr, elements_ptr, nc, Iq, ct.byref(status))
        if status.value == 0:
            arr = np.ctypeslib.as_array(Iq)
            return arr.copy()
        raise RuntimeError(f"AUSAXS: fit failed (error code {status.value})")

    def manual_fit(self, q, I, Ierr, x, y, z, names, resnames, elements):
        """
        Create a manual fitting instance for iterative control.
        Returns an AUSAXSManualFit object with step() and finish() methods.
        """
        return AUSAXSManualFit(self, q, I, Ierr, x, y, z, names, resnames, elements)

class AUSAXSManualFit:
    """Manual fitting class for step-by-step SAXS fitting control."""
    
    def __init__(self, ausaxs_instance: AUSAXS, q, I, Ierr, x, y, z, names, resnames, elements):
        self.ausaxs = ausaxs_instance

        if not self.ausaxs.ready():
            raise RuntimeError(f"AUSAXS: library failed to initialize. Reason: {self.ausaxs.init_error()}")
        _check_array_inputs(
            q, I, Ierr, x, y, z,
            names=['q', 'I', 'Ierr', 'x', 'y', 'z']
        )
        _check_similar_length(x, y, z, names, resnames, elements, msg="Atomic coordinates, weights, and names must have the same length")
        _check_similar_length(q, I, Ierr, msg="q, I, and Ierr must have the same length")

        q, I, Ierr, x, y, z = _as_numpy_f64_arrays(q, I, Ierr, x, y, z)
        self.nq = len(q)  # number of q points
        self.nc = len(x)  # number of coordinates
        self.nq_c = ct.c_int(self.nq)
        self.nc_c = ct.c_int(self.nc)
        self.q = q.ctypes.data_as(ct.POINTER(ct.c_double))
        self.I = I.ctypes.data_as(ct.POINTER(ct.c_double))
        self.Ierr = Ierr.ctypes.data_as(ct.POINTER(ct.c_double))
        self.Iq = (ct.c_double * self.nq)()  # Output array

        self.x = x.ctypes.data_as(ct.POINTER(ct.c_double))
        self.y = y.ctypes.data_as(ct.POINTER(ct.c_double))
        self.z = z.ctypes.data_as(ct.POINTER(ct.c_double))
        self.names = (ct.c_char_p * len(names))(*[s.encode('utf-8') for s in names])
        self.resnames = (ct.c_char_p * len(resnames))(*[s.encode('utf-8') for s in resnames])
        self.elements = (ct.c_char_p * len(elements))(*[s.encode('utf-8') for s in elements])

        status = ct.c_int()
        self.ausaxs._lib.functions.iterative_fit_start(
            self.q, self.I, self.Ierr, self.nq_c, 
            self.x, self.y, self.z, self.names, self.resnames, self.elements, self.nc_c, 
            ct.byref(status)
        )
        if status.value != 0:
            raise RuntimeError(f"AUSAXS: manual fit initialization failed (error code {status.value})")

    def step(self, params) -> np.ndarray:
        """Perform one fitting iteration and return the current I(q)."""
        _check_array_inputs(params, names=['params'])
        params_ptr = _as_numpy_f64_arrays(params)[0].ctypes.data_as(ct.POINTER(ct.c_double))
        status = ct.c_int()
        self.ausaxs._lib.functions.iterative_fit_step(params_ptr, self.Iq, ct.byref(status))
        if status.value == 0:
            arr = np.ctypeslib.as_array(self.Iq)
            return arr.copy()
        raise RuntimeError(f"AUSAXS: fit step failed (error code {status.value})")

    def finish(self, params) -> np.ndarray:
        """Finalize the fitting process and return the optimal I(q)."""
        _check_array_inputs(params, names=['params'])
        params_ptr = _as_numpy_f64_arrays(params)[0].ctypes.data_as(ct.POINTER(ct.c_double))
        status = ct.c_int()
        self.ausaxs._lib.functions.iterative_fit_finish(params_ptr, self.Iq, ct.byref(status))
        if status.value == 0:
            arr = np.ctypeslib.as_array(self.Iq)
            return arr.copy()
        raise RuntimeError(f"AUSAXS: fit finish failed (error code {status.value})")

def ausaxs() -> Optional[AUSAXS]:
    """Return the AUSAXS singleton instance. If the library fails to initialize, returns None."""
    instance = AUSAXS()
    return instance if instance.ready() else None

def create_ausaxs() -> AUSAXS:
    """Return the AUSAXS singleton instance. If the library fails to initialize, raises an exception."""
    instance = AUSAXS()
    if not instance.ready():
        raise RuntimeError(f"AUSAXS: library failed to initialize. Reason: {instance.init_error()}")
    return instance