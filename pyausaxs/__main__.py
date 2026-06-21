import sys
import ctypes as ct

from .wrapper.AUSAXS import AUSAXS
from .__init__ import __version__
from .plot.plot import main as plot_main

# tool name -> (backend C function, program name passed as argv[0])
_CLI_TOOLS = {
    "fit":       ("cli_saxs_fitter", "ausaxs_fit"),
    "em":        ("cli_em_fitter",   "ausaxs_em"),
    "rigidbody": ("cli_rigidbody",   "ausaxs_rigidbody"),
}

_USAGE = """\
Usage: ausaxs <tool> [options]

Available tools:
  fit        - Fit SAXS data to a structure
  em         - Fit EM map to SAXS data
  rigidbody  - Rigid-body optimization
  plot       - Plotting utility
  gui        - Graphical interface

For tool-specific help:
  ausaxs <tool> --help"""


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE)
        return 0

    if argv[0] in ("-v", "--version"):
        print(f"pyAUSAXS version {__version__}")
        return 0

    tool, rest = argv[0].lower(), argv[1:]

    match tool:
        case "setup":
            return _run_setup(rest)
        case "gui":
            print("Warning: The Python GUI is highly experimental. Use at your own risk.", file=sys.stderr)
            from .gui import main as gui_main
            return gui_main(rest)
        case "plot":
            return plot_main(rest)
        case t if t in _CLI_TOOLS:
            return _run_cli_tool(t, rest)
        case _:
            print(f"Unknown tool: {tool}\n", file=sys.stderr)
            print(_USAGE, file=sys.stderr)
            return 2


def _run_cli_tool(tool, args):
    """Invoke one of the backend's C CLI entry points."""
    func_name, prog = _CLI_TOOLS[tool]
    lib = AUSAXS().lib()
    if not lib.ready():
        print("Error: AUSAXS library not ready", file=sys.stderr)
        return 1
    return _call_cli(getattr(lib.functions, func_name), [prog, *args])


def _call_cli(cli_func, args):
    """Call a C CLI function, converting argv (incl. program name) to a char* array."""
    c_args = [arg.encode("utf-8") for arg in args]
    c_argv = (ct.c_char_p * len(c_args))(*c_args)
    return cli_func(len(c_args), c_argv)


def _run_setup(args):
    """Configure which backend shared library AUSAXS uses (unlisted tool)."""
    import argparse
    from pathlib import Path
    from . import loader

    parser = argparse.ArgumentParser(
        prog="ausaxs setup",
        description="Configure the AUSAXS backend shared library.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--relink", metavar="PATH",
        help="use the backend library at PATH instead of the bundled one",
    )
    group.add_argument(
        "--reset", action="store_true",
        help="forget any relinked path and use the bundled library",
    )
    group.add_argument(
        "--show", action="store_true",
        help="show which backend library is currently in use (default)",
    )
    ns = parser.parse_args(args)

    if ns.relink:
        AUSAXS()  # ensure backend is loaded so settings.get("cache") works
        try:
            cache = loader.set_relink_path(ns.relink)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        linked = loader.get_relink_path()
        if "ausaxs" not in Path(linked).stem.lower():
            print(
                f"Warning: '{Path(linked).name}' does not look like an AUSAXS "
                "library (its name should contain 'ausaxs').",
                file=sys.stderr,
            )
        print(f"Relinked AUSAXS backend to: {linked}")
        print(f"(stored in {cache})")
        return 0

    if ns.reset:
        AUSAXS()  # needed to resolve cache dir for clear_relink_path()
        if loader.clear_relink_path():
            print("Reset: now using the bundled AUSAXS backend.")
        else:
            print("No relink configured; already using the bundled backend.")
        return 0

    # default action: --show
    AUSAXS()  # needed to resolve cache dir for get_relink_path()
    print(f"AUSAXS backend in use: {loader.find_lib_path()}")
    if loader.get_relink_path():
        print("  (relinked; reset with: ausaxs setup --reset)")
    else:
        print("  (bundled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
