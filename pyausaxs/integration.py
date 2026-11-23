import ctypes as ct
from enum import Enum

from pyausaxs.loader import resolve_lib
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
        self.lib_path, _ = resolve_lib()
        self._attach_hooks()

    def _attach_hooks(self):
        """
        Load the backend library. Function signatures are not configured here; each is
        applied lazily on first use by the LazyLib proxy (see pyausaxs.signatures), and the
        signatures themselves are registered by the wrapper modules that use them.

        Whether the bundled library runs on this system is decided at install time by the wheel's platform tag.
        """
        try:
            self.functions = LazyLib(ct.CDLL(str(self.lib_path)))
            self.state = self.STATE.READY

        except Exception as e:
            self.state = self.STATE.FAILED
            raise RuntimeError(f"AUSAXS: Unexpected error during library integration: {e}")

    def ready(self):
        return self.state == self.STATE.READY
