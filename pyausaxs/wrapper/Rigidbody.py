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
        status = ct.c_int()
        ausaxs.lib().functions.rigidbody_run(
            self._get_id(),
            ct.byref(status)
        )
        _check_error_code(status, "rigidbody_run")

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
    def get_valid_element_arguments() -> dict[str, list[str]]:
        """Get a dictionary of valid elements and their corresponding arguments."""
        if hasattr(Rigidbody.get_valid_element_arguments, "map"):
            return Rigidbody.get_valid_element_arguments.map

        elements = Rigidbody.get_valid_elements()
        element_arguments = {}
        for element in elements:
            arguments = Rigidbody.get_valid_arguments(element)
            element_arguments[element] = arguments
        setattr(Rigidbody.get_valid_element_arguments, "map", element_arguments)
        return element_arguments

def prepare_rigidbody_refinement(script: str) -> Rigidbody:
    """
    Prepare a rigid-body refinement by loading the refinement script.
    """
    rb = Rigidbody(script)
    return rb