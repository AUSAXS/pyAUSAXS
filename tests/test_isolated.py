"""Runs each isolated_*.py file in a child process of its own; see isolated_gui_core_runner.py for why they can't share this one."""

import glob
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize("cases", sorted(glob.glob(os.path.join(HERE, "isolated_*.py"))), ids=os.path.basename)
def test_in_isolation(cases):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", cases],
        capture_output=True, text=True, cwd=os.path.dirname(HERE),
    )
    assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-2000:]
