import pytest
import numpy as np

from pyausaxs import Rigidbody

def test_rigidbody_valid_elements():
    elements = Rigidbody.get_valid_elements()
    assert isinstance(elements, list), "Valid elements should be returned as a list"
    assert len(elements) > 0, "There should be at least one valid element for rigid-body refinement"

def test_rigidbody_valid_arguments():
    arguments = Rigidbody.get_valid_arguments("parameter")
    assert isinstance(arguments, list), "Valid arguments should be returned as a list"
    assert len(arguments) > 0, f"There should be at least one valid argument for element \"parameter\""