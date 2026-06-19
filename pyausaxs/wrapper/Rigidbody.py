from .AUSAXS import AUSAXS, _check_error_code
from .BackendObject import BackendObject

import ctypes as ct
import numpy as np

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
            is_ca       : (N,) bool  — whether the atom is a Cα"""
        ausaxs = AUSAXS()
        x = ct.POINTER(ct.c_double)()
        y = ct.POINTER(ct.c_double)()
        z = ct.POINTER(ct.c_double)()
        body = ct.POINTER(ct.c_int)()
        copy = ct.POINTER(ct.c_int)()
        residue = ct.POINTER(ct.c_int)()
        is_ca = ct.POINTER(ct.c_int)()
        n_atoms = ct.c_int()
        status = ct.c_int()
        temp_id = ausaxs.lib().functions.rigidbody_get_preview_structure(
            self._get_id(),
            ct.byref(x), ct.byref(y), ct.byref(z),
            ct.byref(body), ct.byref(copy), ct.byref(residue), ct.byref(is_ca),
            ct.byref(n_atoms), ct.byref(status)
        )
        _check_error_code(status, "rigidbody_get_preview_structure")
        n = n_atoms.value
        shape = (n,)
        # copy out before deallocating the backend-owned buffers
        result = {
            "coords": np.column_stack((
                np.ctypeslib.as_array(x, shape=shape),
                np.ctypeslib.as_array(y, shape=shape),
                np.ctypeslib.as_array(z, shape=shape),
            )).astype(float),
            "body": np.ctypeslib.as_array(body, shape=shape).astype(int).copy(),
            "copy": np.ctypeslib.as_array(copy, shape=shape).astype(int).copy(),
            "residue_seq": np.ctypeslib.as_array(residue, shape=shape).astype(int).copy(),
            "is_ca": np.ctypeslib.as_array(is_ca, shape=shape).astype(bool).copy(),
        }
        ausaxs.deallocate(temp_id)
        return result

    @staticmethod
    def set_live_consumer(connected: bool) -> None:
        """Register (or unregister) as a consumer of the live structure. While no consumer is registered, an `update` element does 
        nothing and warns once at parse time, so it wastes no resources. A GUI that polls live_structure() should register on startup."""
        ausaxs = AUSAXS()
        status = ct.c_int()
        ausaxs.lib().functions.rigidbody_set_live_consumer(ct.c_bool(connected), ct.byref(status))
        _check_error_code(status, "rigidbody_set_live_consumer")

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
                np.ctypeslib.as_array(x, shape=(n,)),
                np.ctypeslib.as_array(y, shape=(n,)),
                np.ctypeslib.as_array(z, shape=(n,)),
            )).astype(float)
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
        q_array = np.ctypeslib.as_array(q, shape=(n_points.value,))
        I_array = np.ctypeslib.as_array(I, shape=(n_points.value,))
        I_err_array = np.ctypeslib.as_array(I_err, shape=(n_points.value,))
        I_interp_array = np.ctypeslib.as_array(I_interp, shape=(n_points.value,))
        result = np.column_stack((q_array, I_array, I_err_array, I_interp_array))
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
        elements = [elements_ptr[i].decode('utf-8') for i in range(size.value)]
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
        arguments = [arguments_ptr[i].decode('utf-8') for i in range(size.value)]
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