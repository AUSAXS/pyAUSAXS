from pyausaxs.wrapper import AUSAXS, _check_array_inputs, _check_similar_length, _as_numpy_f64_arrays
import ctypes as ct
import numpy as np

class AUSAXSManualFit:
    """Manual fitting class for step-by-step SAXS fitting control."""
    
    def __init__(self, q, I, Ierr, x, y, z, names, resnames, elements):
        ausaxs = AUSAXS()
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