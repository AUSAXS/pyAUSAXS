"""
PyInstaller integration for pyausaxs package.

This module provides the entry point functions for PyInstaller hook discovery.
"""

import os
from pathlib import Path


def get_hook_dirs():
    """
    Return the directory containing pyausaxs PyInstaller hooks.
    
    This function is called by PyInstaller's entry point system to discover
    hook directories for this package.
    
    Returns:
        list: List containing the absolute path to the hooks directory
    """
    # Get the directory containing this module
    this_dir = Path(__file__).parent
    hooks_dir = this_dir / "hooks"
    
    # Return as absolute path string in a list
    if hooks_dir.exists():
        return [str(hooks_dir.absolute())]
    return []


def get_PyInstaller_tests():
    """
    Return directories containing PyInstaller tests for this package.
    
    Currently, we don't provide any specific PyInstaller tests.
    
    Returns:
        list: Empty list (no tests provided)
    """
    return []