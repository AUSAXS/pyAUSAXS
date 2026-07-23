from .AUSAXS import AUSAXS, _check_error_code, _ptr_to_array, _ptr_to_str_list
from .BackendObject import BackendObject
from pyausaxs.signatures import register

import ctypes as ct
import numpy as np

register({
    "rigidbody_load_script": (
        [
            ct.c_char_p,         # script
            ct.POINTER(ct.c_int) # status (0 = success)
        ],
        ct.c_int                 # return rigidbody id
    ),
    "rigidbody_validate": (
        [
            ct.c_int,            # rigidbody id
            ct.POINTER(ct.c_int) # status (0 = success)
        ],
        None
    ),
    "rigidbody_get_preview_structure": (
        [
            ct.c_int,                            # rigidbody id
            ct.POINTER(ct.POINTER(ct.c_double)), # x vector (output)
            ct.POINTER(ct.POINTER(ct.c_double)), # y vector (output)
            ct.POINTER(ct.POINTER(ct.c_double)), # z vector (output)
            ct.POINTER(ct.POINTER(ct.c_int)),    # body_index vector (output)
            ct.POINTER(ct.POINTER(ct.c_int)),    # copy_index vector (output)
            ct.POINTER(ct.POINTER(ct.c_int)),    # residue_seq vector (output)
            ct.POINTER(ct.POINTER(ct.c_int)),    # is_ca vector (output)
            ct.POINTER(ct.c_int),                # n_atoms (output)
            ct.POINTER(ct.POINTER(ct.c_int)),    # constraint_data vector, flat [idx1,idx2,type]* (output)
            ct.POINTER(ct.c_int),                # n_constraints (output)
            ct.POINTER(ct.c_int)                 # status (0 = success)
        ],
        ct.c_int                                 # return data id
    ),
    "rigidbody_get_live_structure": (
        [
            ct.POINTER(ct.POINTER(ct.c_double)), # x vector (output)
            ct.POINTER(ct.POINTER(ct.c_double)), # y vector (output)
            ct.POINTER(ct.POINTER(ct.c_double)), # z vector (output)
            ct.POINTER(ct.c_int),                # n_atoms (output)
            ct.POINTER(ct.c_int),                # version (output)
            ct.POINTER(ct.c_int)                 # status (0 = success)
        ],
        ct.c_int                                 # return data id
    ),
    "rigidbody_register_live_consumer": (
        [
            ct.c_bool,           # connected
            ct.POINTER(ct.c_int) # status (0 = success)
        ],
        None
    ),
    "rigidbody_run": (
        [
            ct.c_int,                            # rigidbody id
            ct.POINTER(ct.POINTER(ct.c_double)), # q vector (output)
            ct.POINTER(ct.POINTER(ct.c_double)), # I vector (output)
            ct.POINTER(ct.POINTER(ct.c_double)), # Ierr vector (output)
            ct.POINTER(ct.POINTER(ct.c_double)), # Iinterp vector (output)
            ct.POINTER(ct.c_int),                # n_points (output)
            ct.POINTER(ct.c_int)                 # status (0 = success)
        ],
        ct.c_int                                 # return data id
    ),
    "rigidbody_get_valid_elements": (
        [
            ct.POINTER(ct.POINTER(ct.c_char_p)), # elements (output)
            ct.POINTER(ct.c_int),                # size (output)
            ct.POINTER(ct.c_int)                 # status (0 = success)
        ],
        None
    ),
    "rigidbody_get_valid_arguments": (
        [
            ct.c_char_p,                         # element name
            ct.POINTER(ct.POINTER(ct.c_char_p)), # arguments (output)
            ct.POINTER(ct.c_int),                # size (output)
            ct.POINTER(ct.c_int)                 # status (0 = success)
        ],
        None
    ),
    "rigidbody_get_body_names": (
        [
            ct.c_int,                            # rigidbody id
            ct.POINTER(ct.POINTER(ct.c_char_p)), # names (output)
            ct.POINTER(ct.c_int),                # size (output)
            ct.POINTER(ct.c_int)                 # status (0 = success)
        ],
        None
    ),
    "rigidbody_get_symmetry_layout": (
        [
            ct.c_int,                            # rigidbody id
            ct.POINTER(ct.POINTER(ct.c_int)),    # body vector (output)
            ct.POINTER(ct.POINTER(ct.c_int)),    # copy vector (output)
            ct.POINTER(ct.POINTER(ct.c_int)),    # symmetry vector (output)
            ct.POINTER(ct.POINTER(ct.c_int)),    # replica vector (output)
            ct.POINTER(ct.POINTER(ct.c_char_p)), # type vector (output)
            ct.POINTER(ct.POINTER(ct.c_char_p)), # name vector (output)
            ct.POINTER(ct.c_int),                # n_replicas (output)
            ct.POINTER(ct.c_int)                 # status (0 = success)
        ],
        ct.c_int                                 # return data id
    ),
})

class Rigidbody(BackendObject):
    def __init__(self, script: str):
        super().__init__()
        self._read_script(script)

    def _read_script(self, script: str) -> None:
        ausaxs = AUSAXS()
        script_c = script.encode('utf-8')
        status = ct.c_int()
        self._set_id(ausaxs.lib().functions.rigidbody_load_script(
            script_c,
            ct.byref(status)
        ))
        _check_error_code(status, "rigidbody_load_script")

    def preview_structure(self) -> dict:
        """Get the explicit structure (symmetries realized) annotated with the per-atom metadata needed to draw a preview. 
        Returns a dict with:
            coords      : (N, 3) float array of atom positions
            body        : (N,) int   — index of the body each atom belongs to
            copy        : (N,) int   — symmetry copy index (0 = original)
            residue_seq : (N,) int   — residue number (-1 if unknown)
            is_ca       : (N,) bool  — whether the atom is a Cα
            constraints : (M, 3) int — one row per constraint, [atom_index_1, atom_index_2, type]
                          with type 0=backbone, 1=centre-of-mass, 2=attractor, 3=repulsor.
                          The atom indices point into the rows of `coords` (copy 0 of each body)."""
        ausaxs = AUSAXS()
        x = ct.POINTER(ct.c_double)()
        y = ct.POINTER(ct.c_double)()
        z = ct.POINTER(ct.c_double)()
        body = ct.POINTER(ct.c_int)()
        copy = ct.POINTER(ct.c_int)()
        residue = ct.POINTER(ct.c_int)()
        is_ca = ct.POINTER(ct.c_int)()
        n_atoms = ct.c_int()
        constraints = ct.POINTER(ct.c_int)()
        n_constraints = ct.c_int()
        status = ct.c_int()
        temp_id = ausaxs.lib().functions.rigidbody_get_preview_structure(
            self._get_id(),
            ct.byref(x), ct.byref(y), ct.byref(z),
            ct.byref(body), ct.byref(copy), ct.byref(residue), ct.byref(is_ca),
            ct.byref(n_atoms), ct.byref(constraints), ct.byref(n_constraints), ct.byref(status)
        )
        _check_error_code(status, "rigidbody_get_preview_structure")
        n = n_atoms.value
        nc = n_constraints.value
        # copy out before deallocating the backend-owned buffers (constraint_data is null when nc == 0)
        result = {
            "coords": np.column_stack((
                _ptr_to_array(x, n),
                _ptr_to_array(y, n),
                _ptr_to_array(z, n),
            )),
            "body": _ptr_to_array(body, n, dtype=int),
            "copy": _ptr_to_array(copy, n, dtype=int),
            "residue_seq": _ptr_to_array(residue, n, dtype=int),
            "is_ca": _ptr_to_array(is_ca, n, dtype=bool),
            "constraints": (_ptr_to_array(constraints, nc * 3, dtype=int).reshape(-1, 3)
                            if nc else np.empty((0, 3), dtype=int)),
        }
        ausaxs.deallocate(temp_id)
        return result

    def body_names(self) -> list[str]:
        """Names of the bodies currently in the setup, after any merge/delete/convert_to_symmetry
        elements have collapsed or removed bodies. The order matches the body indices reported by
        preview_structure(), so body index i corresponds to body_names()[i]."""
        ausaxs = AUSAXS()
        names = ct.POINTER(ct.c_char_p)()
        size = ct.c_int()
        status = ct.c_int()
        ausaxs.lib().functions.rigidbody_get_body_names(
            self._get_id(), ct.byref(names), ct.byref(size), ct.byref(status)
        )
        _check_error_code(status, "rigidbody_get_body_names")
        return _ptr_to_str_list(names, size.value)

    def symmetry_layout(self) -> dict:
        """Get the symmetry-replica layout of the current structure, one row per replica (copy > 0),
        keyed to preview_structure()'s (body, copy) pairs so the GUI never has to guess the
        copy -> symmetry mapping. Returns a dict with:
            body     : (R,) int       — body index (matches preview_structure's body)
            copy     : (R,) int       — symmetry copy index (matches preview_structure's copy)
            symmetry : (R,) int       — 0-based symmetry index within the body
            replica  : (R,) int       — replica index within the symmetry
            type     : (R,) list[str] — symmetry type (e.g. "c4", "p2")
            name     : (R,) list[str] — current addressable name, including a rename alias when present (e.g. "b1s1r1" or "my_replica")"""
        ausaxs = AUSAXS()
        body = ct.POINTER(ct.c_int)()
        copy = ct.POINTER(ct.c_int)()
        symmetry = ct.POINTER(ct.c_int)()
        replica = ct.POINTER(ct.c_int)()
        type_ = ct.POINTER(ct.c_char_p)()
        name = ct.POINTER(ct.c_char_p)()
        n_replicas = ct.c_int()
        status = ct.c_int()
        temp_id = ausaxs.lib().functions.rigidbody_get_symmetry_layout(
            self._get_id(),
            ct.byref(body), ct.byref(copy), ct.byref(symmetry), ct.byref(replica),
            ct.byref(type_), ct.byref(name),
            ct.byref(n_replicas), ct.byref(status)
        )
        _check_error_code(status, "rigidbody_get_symmetry_layout")
        n = n_replicas.value
        result = {
            "body": _ptr_to_array(body, n, dtype=int),
            "copy": _ptr_to_array(copy, n, dtype=int),
            "symmetry": _ptr_to_array(symmetry, n, dtype=int),
            "replica": _ptr_to_array(replica, n, dtype=int),
            "type": _ptr_to_str_list(type_, n),
            "name": _ptr_to_str_list(name, n),
        }
        ausaxs.deallocate(temp_id)
        return result

    @staticmethod
    def set_live_consumer(connected: bool) -> None:
        """Register (or unregister) as a consumer of the live structure. While no consumer is registered, an `update` element does 
        nothing and warns once at parse time, so it wastes no resources. A GUI that polls live_structure() should register on startup."""
        ausaxs = AUSAXS()
        status = ct.c_int()
        ausaxs.lib().functions.rigidbody_register_live_consumer(ct.c_bool(connected), ct.byref(status))
        _check_error_code(status, "rigidbody_register_live_consumer")

    @staticmethod
    def live_structure() -> tuple:
        """Poll the structure most recently published by an `update structure` element during a run. Returns (coords, version), where
        coords is an (N, 3) float array (or None if nothing has been published yet) and version is an int that increments on each 
        publish — the GUI compares it against the previous value to detect new frames. The atom order matches preview_structure(), 
        so a backbone mask computed there can be reused here. Thread-safe."""
        ausaxs = AUSAXS()
        x = ct.POINTER(ct.c_double)()
        y = ct.POINTER(ct.c_double)()
        z = ct.POINTER(ct.c_double)()
        n_atoms = ct.c_int()
        version = ct.c_int()
        status = ct.c_int()
        temp_id = ausaxs.lib().functions.rigidbody_get_live_structure(
            ct.byref(x), ct.byref(y), ct.byref(z),
            ct.byref(n_atoms), ct.byref(version), ct.byref(status)
        )
        _check_error_code(status, "rigidbody_get_live_structure")
        n = n_atoms.value
        coords = None
        if n:
            coords = np.column_stack((
                _ptr_to_array(x, n),
                _ptr_to_array(y, n),
                _ptr_to_array(z, n),
            ))
        ausaxs.deallocate(temp_id)
        return coords, version.value

    def validate(self) -> None:
        """Validate the script and raise an error if it is invalid."""
        ausaxs = AUSAXS()
        status = ct.c_int()
        ausaxs.lib().functions.rigidbody_validate(
            self._get_id(),
            ct.byref(status)
        )
        _check_error_code(status, "rigidbody_validate")

    def run(self) -> np.ndarray:
        """Run the rigid-body refinement script."""
        ausaxs = AUSAXS()
        q = ct.POINTER(ct.c_double)()
        I = ct.POINTER(ct.c_double)()
        I_err = ct.POINTER(ct.c_double)()
        I_interp = ct.POINTER(ct.c_double)()
        n_points = ct.c_int()
        status = ct.c_int()
        ausaxs.lib().functions.rigidbody_run(
            self._get_id(),
            ct.byref(q),
            ct.byref(I),
            ct.byref(I_err),
            ct.byref(I_interp),
            ct.byref(n_points),
            ct.byref(status)
        )
        _check_error_code(status, "rigidbody_run")
        n = n_points.value
        result = np.column_stack((
            _ptr_to_array(q, n),
            _ptr_to_array(I, n),
            _ptr_to_array(I_err, n),
            _ptr_to_array(I_interp, n),
        ))
        return result

    @staticmethod
    def get_valid_elements() -> list[str]:
        """Get a list of valid elements for rigid-body refinement."""
        ausaxs = AUSAXS()
        elements_ptr = ct.POINTER(ct.c_char_p)()
        size = ct.c_int()
        status = ct.c_int()
        ausaxs.lib().functions.rigidbody_get_valid_elements(
            ct.byref(elements_ptr),
            ct.byref(size),
            ct.byref(status)
        )
        _check_error_code(status, "rigidbody_get_valid_elements")
        elements = _ptr_to_str_list(elements_ptr, size.value)
        return elements

    @staticmethod
    def get_valid_arguments(element: str) -> list[str]:
        """Get a list of valid arguments for a given element."""
        ausaxs = AUSAXS()
        element_c = element.encode('utf-8')
        arguments_ptr = ct.POINTER(ct.c_char_p)()
        size = ct.c_int()
        status = ct.c_int()
        ausaxs.lib().functions.rigidbody_get_valid_arguments(
            element_c,
            ct.byref(arguments_ptr),
            ct.byref(size),
            ct.byref(status)
        )
        _check_error_code(status, "rigidbody_get_valid_arguments")
        arguments = _ptr_to_str_list(arguments_ptr, size.value)
        return arguments

    @staticmethod
    def get_valid_elements_and_arguments() -> dict[str, list[str]]:
        """Get a dictionary of valid elements and their corresponding arguments."""
        if hasattr(Rigidbody.get_valid_elements_and_arguments, "map"):
            return Rigidbody.get_valid_elements_and_arguments.map

        elements = Rigidbody.get_valid_elements()
        element_arguments = {}
        for element in elements:
            arguments = Rigidbody.get_valid_arguments(element)
            element_arguments[element] = arguments
        setattr(Rigidbody.get_valid_elements_and_arguments, "map", element_arguments)
        return element_arguments

def prepare_rigidbody_refinement(script: str) -> Rigidbody:
    """
    Prepare a rigid-body refinement by loading the refinement script.
    """
    rb = Rigidbody(script)
    return rb