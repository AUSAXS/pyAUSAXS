import multiprocessing
import ctypes as ct
from enum import Enum

from pyausaxs.loader import find_lib_path
from pyausaxs.architecture import CPUFeatures

class AUSAXSLIB: 
    class STATE(Enum):
        UNINITIALIZED = 0
        FAILED = 1
        READY = 2

    def __init__(self):
        self.functions = None
        self.state = self.STATE.UNINITIALIZED
        # prefer library bundled inside package resources; fall back to top-level resources/
        self.lib_path = find_lib_path()

        self._check_cpu_compatibility()
        self._attach_hooks()
        self._test_integration()

    def _check_cpu_compatibility(self):
        """Check if the current CPU is compatible with the AUSAXS library."""
        if not CPUFeatures.is_compatible_architecture():
            self.state = self.STATE.FAILED
            raise RuntimeError(f"AUSAXS: Incompatible CPU architecture: {CPUFeatures.get_architecture()}")
        return True

    def _attach_hooks(self):
        # skip if CPU compatibility check already failed
        if self.state == self.STATE.FAILED:
            return

        # see the corresponding API at https://github.com/AUSAXS/AUSAXS/blob/master/include/core/api/sasview.h 
        self.state = self.STATE.READY
        try:
            self.functions = ct.CDLL(str(self.lib_path))

            # test_integration
            self.functions.test_integration.argtypes = [
                ct.POINTER(ct.c_int) # test val
            ]
            self.functions.test_integration.restype = None # returns void

            # evaluate_sans_debye
            self.functions.evaluate_sans_debye.argtypes = [
                ct.POINTER(ct.c_double), # q vector
                ct.POINTER(ct.c_double), # atom x vector
                ct.POINTER(ct.c_double), # atom y vector
                ct.POINTER(ct.c_double), # atom z vector
                ct.POINTER(ct.c_double), # atom weight vector
                ct.c_int,                # nq (number of points in q)
                ct.c_int,                # nc (number of points in x, y, z, w)
                ct.POINTER(ct.c_double), # Iq vector for return value
                ct.POINTER(ct.c_int)     # status (0 = success)
            ]
            self.functions.evaluate_sans_debye.restype = None # returns void

            # fit_saxs
            self.functions.fit_saxs.argtypes = [
                ct.POINTER(ct.c_double), # data q vector
                ct.POINTER(ct.c_double), # data I vector
                ct.POINTER(ct.c_double), # data Ierr vector
                ct.c_int,                # n_data (number of points in q, I, Ierr)
                ct.POINTER(ct.c_double), # pdb x vector
                ct.POINTER(ct.c_double), # pdb y vector
                ct.POINTER(ct.c_double), # pdb z vector
                ct.POINTER(ct.c_char_p), # pdb atom names
                ct.POINTER(ct.c_char_p), # pdb residue names
                ct.POINTER(ct.c_char_p), # pdb elements
                ct.c_int,                # n_pdb (number of atoms)
                ct.POINTER(ct.c_double), # return I vector for return value
                ct.POINTER(ct.c_int)     # return status (0 = success)
            ]
            self.functions.fit_saxs.restype = None # returns void

            # iterative_fit_start
            self.functions.iterative_fit_start.argtypes = [
                ct.POINTER(ct.c_double), # data q vector
                ct.POINTER(ct.c_double), # data I vector
                ct.POINTER(ct.c_double), # data Ierr vector
                ct.c_int,                # n_data (number of points in q, I, Ierr)
                ct.POINTER(ct.c_double), # pdb x vector
                ct.POINTER(ct.c_double), # pdb y vector
                ct.POINTER(ct.c_double), # pdb z vector
                ct.POINTER(ct.c_char_p), # pdb atom names
                ct.POINTER(ct.c_char_p), # pdb residue names
                ct.POINTER(ct.c_char_p), # pdb elements
                ct.c_int,                # n_pdb (number of atoms)
                ct.POINTER(ct.c_int)     # return status (0 = success)
            ]
            self.functions.iterative_fit_start.restype = None # returns void

            # iterative_fit_step
            self.functions.iterative_fit_step.argtypes = [
                ct.POINTER(ct.c_double), # parameters vector
                ct.POINTER(ct.c_double), # return I vector for return value
                ct.POINTER(ct.c_int),    # return status (0 = success)
            ]
            self.functions.iterative_fit_step.restype = None # returns void

            # iterative_fit_finish
            self.functions.iterative_fit_finish.argtypes = [
                ct.POINTER(ct.c_double), # parameters vector
                ct.POINTER(ct.c_double), # return I vector for return value
                ct.POINTER(ct.c_int),    # return status (0 = success)
            ]
            self.functions.iterative_fit_finish.restype = None # returns void

            self.state = self.STATE.READY

        except Exception as e:
            self.state = self.STATE.FAILED
            raise RuntimeError(f"AUSAXS: Unexpected error during library integration: {e}")

    def _test_integration(self):
        """
        Test the integration of the AUSAXS library by running a simple test function in a separate process. 
        This protects the main thread from potential segfaults due to e.g. incompatible architectures. 
        """
        if (self.state != self.STATE.READY):
            return

        try: 
            # we need a queue to access the return value
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=_run, args=(self.lib_path, queue))
            p.start()
            p.join()
            if p.exitcode == 0: # process successfully terminated
                val = queue.get_nowait() # get the return value
                if (val != 6): # test_integration increments the test value by 1
                    raise Exception("AUSAXS integration test failed. Test value was not incremented")
            else:
                raise Exception(f"AUSAXS: External invocation seems to have crashed (exit code \"{p.exitcode}\").")

        except Exception as e:
            self.state = self.STATE.FAILED
            raise RuntimeError(f"AUSAXS: Unexpected integration test failure: \"{e}\".")

    def ready(self):
        return self.state == self.STATE.READY

def _run(lib_path, queue):
    """
    Helper method for AUSAXSLIB._test_integration, which must be defined in global scope to be picklable.
    """
    func = ct.CDLL(str(lib_path))
    func.test_integration.argtypes = [ct.POINTER(ct.c_int)]
    func.test_integration.restype = None
    test_val = ct.c_int(5)
    func.test_integration(ct.byref(test_val))
    queue.put(test_val.value)