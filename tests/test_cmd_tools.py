import pytest
import os

def test_cmd_tool_rigidbody():
    cmd = "ausaxs rigidbody --help"
    result = os.system(cmd)
    assert result == 0, "RigidBody command-line tool should run without error"

def test_cmd_tool_fit():
    cmd = "ausaxs fit --help"
    result = os.system(cmd)
    assert result == 0, "Fit command-line tool should run without error"

def test_cmd_tool_em():
    cmd = "ausaxs em --help"
    result = os.system(cmd)
    assert result == 0, "EM command-line tool should run without error"

def test_cmd_tool_plot(): 
    cmd = "ausaxs plot --help"
    result = os.system(cmd)
    assert result == 0, "Plot command-line tool should run without error"