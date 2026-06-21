import multiprocessing
import ctypes as ct
from enum import Enum

from pyausaxs.config import architecture_runtime_validation
from pyausaxs.loader import bundled_lib_path, get_relink_path
from pyausaxs.architecture import CPUFeatures
from pyausaxs.signatures import LazyLib

OutputCallback = ct.CFUNCTYPE(None, ct.c_char_p, ct.c_int)
class AUSAXSLIB:
    class STATE(Enum):
        UNINITIALIZED = 0
        FAILED = 1
        READY = 2

    def __init__(self):
        self.functions: LazyLib = None # type: ignore[assignment]
        self.state = self.STATE.UNINITIALIZED
        self.lib_path = bundled_lib_path()

        self._check_cpu_compatibility()
        self._attach_hooks()

        relink = get_relink_path()
        if relink:
            self.lib_path = relink
            self._attach_hooks()

        self._test_integration()

    def _check_cpu_compatibility(self):
        """Check if the current CPU is compatible with the AUSAXS library."""
        if architecture_runtime_validation and not CPUFeatures.is_compatible_architecture():
            self.state = self.STATE.FAILED
            raise RuntimeError(f"AUSAXS: Incompatible CPU architecture: {CPUFeatures.get_architecture()}")
        return True

    def _attach_hooks(self):
        """
        Load the backend library. Function signatures are not configured here; each is
        applied lazily on first use by the LazyLib proxy (see pyausaxs.signatures), and the
        signatures themselves are registered by the wrapper modules that use them.
        """
        # skip if CPU compatibility check already failed
        if self.state == self.STATE.FAILED:
            return

        try:
            self.functions = LazyLib(ct.CDLL(str(self.lib_path)))
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
