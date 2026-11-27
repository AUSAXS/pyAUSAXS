from .AUSAXS import call
from pyausaxs.signatures import register, STATUS
import ctypes as ct

register({
    "io_is_pdb":              ([ct.c_char_p, STATUS], ct.c_bool),
    "io_is_saxs_data":        ([ct.c_char_p, STATUS], ct.c_bool),
    "io_is_em_map":           ([ct.c_char_p, STATUS], ct.c_bool),
    "io_is_rigidbody_config": ([ct.c_char_p, STATUS], ct.c_bool),
})

def _is_pdb_file(filename: str) -> bool:
    return bool(call("io_is_pdb", filename))

def _is_saxs_data_file(filename: str) -> bool:
    return bool(call("io_is_saxs_data", filename))

def _is_em_map_file(filename: str) -> bool:
    return bool(call("io_is_em_map", filename))

def _is_rigidbody_config_file(filename: str) -> bool:
    return bool(call("io_is_rigidbody_config", filename))
