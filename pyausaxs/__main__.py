import sys
import ctypes as ct

from .wrapper.AUSAXS import AUSAXS
from .__init__ import __version__
from .plot.plot import main as plot_main


def main(argv=None):
    # if no args provided, show help
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print("\nUsage: ausaxs <tool> [options]")
        print("\nAvailable tools:")
        print("  fit        - Fit SAXS data to a structure")
        print("  em         - Fit EM map to SAXS data")
        print("  rigidbody  - Rigid-body optimization")
        print("  plot       - Plotting utility")
        print("  gui        - Graphical interface") 
        print("\nFor tool-specific help:")
        print("  ausaxs <tool> --help")
        return 0
    
    if argv[0] in ("-v", "--version"):
        print(f"pyAUSAXS version {__version__}")
        return 0

    # check if first arg is a tool selector
    tool = argv[0].lower()

    # the gui handles library initialization itself
    if tool == "gui":
        print("Warning: The Python GUI is highly experimental. Use at your own risk.", file=sys.stderr)
        from .gui import main as gui_main
        return gui_main(argv[1:])

    # route to appropriate CLI tool
    lib = AUSAXS().lib()
    if not lib.ready():
        print("Error: AUSAXS library not ready", file=sys.stderr)
        return 1

    # prepare argv for C function (needs program name as argv[0])
    match tool:
        case "fit":
            c_argv = ["ausaxs_fit"] + argv[1:]
            return _call_cli(lib.functions.cli_saxs_fitter, c_argv)
        case "em":
            c_argv = ["ausaxs_em"] + argv[1:]
            return _call_cli(lib.functions.cli_em_fitter, c_argv)
        case "rigidbody":
            c_argv = ["ausaxs_rigidbody"] + argv[1:]
            return _call_cli(lib.functions.cli_rigidbody, c_argv)
        case "plot":
            return _run_plot_tool(argv[1:])
        case _:
            print(f"Unknown tool: {tool}", file=sys.stderr)
            print("Available tools: fit, em, rigidbody, plot, gui", file=sys.stderr)
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


def _run_plot_tool(args):
    """Run the plotting tool using plot_main."""
    return plot_main(args)


def saxs_fitter():
    """Entry point for saxs_fitter CLI tool."""
    lib = AUSAXS().lib()
    if not lib.ready():
        print("Error: AUSAXS library not ready", file=sys.stderr)
        return 1
    c_argv = ["saxs_fitter"] + sys.argv[1:]
    return _call_cli(lib.functions.cli_saxs_fitter, c_argv)


def em_fitter():
    """Entry point for em_fitter CLI tool."""
    lib = AUSAXS().lib()
    if not lib.ready():
        print("Error: AUSAXS library not ready", file=sys.stderr)
        return 1
    c_argv = ["em_fitter"] + sys.argv[1:]
    return _call_cli(lib.functions.cli_em_fitter, c_argv)


def rigidbody_optimizer():
    """Entry point for rigidbody_optimizer CLI tool."""
    lib = AUSAXS().lib()
    if not lib.ready():
        print("Error: AUSAXS library not ready", file=sys.stderr)
        return 1
    c_argv = ["rigidbody_optimizer"] + sys.argv[1:]
    return _call_cli(lib.functions.cli_rigidbody, c_argv)


if __name__ == "__main__":
    raise SystemExit(main())