from .AUSAXS import AUSAXS, _check_error_code
from pyausaxs.signatures import register
import ctypes as ct

register({
    "io_is_pdb": ([ct.c_char_p, ct.POINTER(ct.c_int)], bool),
    "io_is_saxs_data": ([ct.c_char_p, ct.POINTER(ct.c_int)], bool),
    "io_is_em_map": ([ct.c_char_p, ct.POINTER(ct.c_int)], bool),
    "io_is_rigidbody_config": ([ct.c_char_p, ct.POINTER(ct.c_int)], bool),
})

def _is_pdb_file(filename: str) -> bool:
    ausaxs = AUSAXS()
    filename_c = filename.encode('utf-8')
    status = ct.c_int()
    result = ausaxs.lib().functions.io_is_pdb(
        filename_c,
        ct.byref(status)
    )
    _check_error_code(status, "is_pdb_file")
    return bool(result)

def _is_saxs_data_file(filename: str) -> bool:
    ausaxs = AUSAXS()
    filename_c = filename.encode('utf-8')
    status = ct.c_int()
    result = ausaxs.lib().functions.io_is_saxs_data(
        filename_c,
        ct.byref(status)
    )
    _check_error_code(status, "is_data_file")
    return bool(result)

def _is_em_map_file(filename: str) -> bool:
    ausaxs = AUSAXS()
    filename_c = filename.encode('utf-8')
    status = ct.c_int()
    result = ausaxs.lib().functions.io_is_em_map(
        filename_c,
        ct.byref(status)
    )
    _check_error_code(status, "is_em_map_file")
    return bool(result)

def _is_rigidbody_config_file(filename: str) -> bool:
    ausaxs = AUSAXS()
    filename_c = filename.encode('utf-8')
    status = ct.c_int()
    result = ausaxs.lib().functions.io_is_rigidbody_config(
        filename_c,
        ct.byref(status)
    )
    _check_error_code(status, "is_rigidbody_config_file")
    return bool(result)