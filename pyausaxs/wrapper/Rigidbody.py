from .AUSAXS import AUSAXS, _check_error_code

import ctypes as ct

def run_rigidbody_script(path: str) -> None:
    """Run the rigidbody script at the given path."""
    ausaxs = AUSAXS()
    status = ct.c_int()
    path_ptr = ct.c_char_p(path.encode('utf-8'))
    ausaxs.lib().functions.rigidbody_config_run(path_ptr, ct.byref(status))
    _check_error_code(status, "run_rigidbody_script")