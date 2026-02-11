import argparse
import sys
import ctypes as ct

from .wrapper.Filetypes import _is_pdb_file, _is_saxs_data_file, _is_rigidbody_config_file
from .wrapper.Rigidbody import run_rigidbody_script
from .integration import AUSAXSLIB

from .__init__ import __version__


def main(argv=None):
    # if no args provided, show help
    if argv is None:
        argv = sys.argv[1:]
    
    if not argv:
        print(f"PyAUSAXS {__version__}")
        print("\nUsage: ausaxs <tool> [options]")
        print("\nAvailable tools:")
        print("  fit        - Fit SAXS data to a structure (uses saxs_fitter)")
        print("  em         - Fit EM map to SAXS data (uses em_fitter)")
        print("  rigidbody  - Rigid-body optimization (uses rigidbody_optimizer)")
        print("\nOr provide a rigidbody config file:")
        print("  ausaxs <config.txt>")
        print("\nFor tool-specific help:")
        print("  ausaxs <tool> --help")
        return 0

    # check if first arg is a tool selector
    tool = argv[0].lower()
    
    # legacy support: check if it's a rigidbody config file
    if len(argv) == 1 and _is_rigidbody_config_file(argv[0]):
        try:
            run_rigidbody_script(argv[0])
            return 0
        except Exception as e:
            print(f"Error running rigidbody script: {e}", file=sys.stderr)
            return 1

    # route to appropriate CLI tool
    lib = AUSAXSLIB()
    if not lib.ready():
        print("Error: AUSAXS library not ready", file=sys.stderr)
        return 1

    # prepare argv for C function (needs program name as argv[0])
    if tool == "fit":
        c_argv = ["ausaxs_fit"] + argv[1:]
        return _call_cli(lib.functions.cli_saxs_fitter, c_argv)
    elif tool == "em":
        c_argv = ["ausaxs_em"] + argv[1:]
        return _call_cli(lib.functions.cli_em_fitter, c_argv)
    elif tool == "rigidbody":
        c_argv = ["ausaxs_rigidbody"] + argv[1:]
        return _call_cli(lib.functions.cli_rigidbody, c_argv)
    else:
        print(f"Unknown tool: {tool}", file=sys.stderr)
        print("Use 'ausaxs' without arguments to see available tools", file=sys.stderr)
        return 2


def _call_cli(cli_func, args):
    """
    Helper function to call a C CLI function with proper argument conversion.
    
    Args:
        cli_func: The C function to call
        args: List of string arguments (including program name as argv[0])
    
    Returns:
        Exit code from the CLI function
    """
    # convert Python strings to C char* array
    argc = len(args)
    
    # encode strings to bytes and create ctypes array
    c_args = [arg.encode('utf-8') for arg in args]
    argv = (ct.c_char_p * argc)(*c_args)
    
    # call the C function
    return cli_func(argc, argv)


def saxs_fitter():
    """Entry point for saxs_fitter CLI tool."""
    lib = AUSAXSLIB()
    if not lib.ready():
        print("Error: AUSAXS library not ready", file=sys.stderr)
        return 1
    c_argv = ["saxs_fitter"] + sys.argv[1:]
    return _call_cli(lib.functions.cli_saxs_fitter, c_argv)


def em_fitter():
    """Entry point for em_fitter CLI tool."""
    lib = AUSAXSLIB()
    if not lib.ready():
        print("Error: AUSAXS library not ready", file=sys.stderr)
        return 1
    c_argv = ["em_fitter"] + sys.argv[1:]
    return _call_cli(lib.functions.cli_em_fitter, c_argv)


def rigidbody_optimizer():
    """Entry point for rigidbody_optimizer CLI tool."""
    lib = AUSAXSLIB()
    if not lib.ready():
        print("Error: AUSAXS library not ready", file=sys.stderr)
        return 1
    c_argv = ["rigidbody_optimizer"] + sys.argv[1:]
    return _call_cli(lib.functions.cli_rigidbody, c_argv)


if __name__ == "__main__":
    raise SystemExit(main())