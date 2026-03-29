import pytest
import numpy as np

from pyausaxs import form_factor

def test_valid_types():
    types = form_factor.valid_types()
    assert isinstance(types, np.ndarray), "Form factor types should be returned as a list"
    assert len(types) > 0, "There should be at least one form factor type available"
    print("Available form factor types:", types)

def test_get_five_gaussian_coefficients():
    element = form_factor.valid_types()[0]
    a, b, c = form_factor.get_five_gaussian_coefficients(element)
    assert isinstance(a, np.ndarray) and a.shape == (5,), "Coefficients a should be a 5-element array"
    assert isinstance(b, np.ndarray) and b.shape == (5,), "Coefficients b should be a 5-element array"
    assert isinstance(c, float), "Coefficient c should be a float"

def test_get_current_exv_volume():
    element = form_factor.valid_types()[1]
    volume = form_factor.get_current_exv_volume(element)
    assert isinstance(volume, float), "Excluded volume should be a float"
    assert volume > 0, "Excluded volume should be positive"