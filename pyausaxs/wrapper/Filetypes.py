from .AUSAXS import AUSAXS, _check_error_code
import ctypes as ct

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
    result = ausaxs.lib().functions.io_is_data(
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