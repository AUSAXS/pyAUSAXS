import argparse
import sys

from .__init__ import __version__

# tool name -> (backend C function name, program name passed as argv[0])
_CLI_TOOLS = {
    "fit":       ("cli_saxs_fitter", "ausaxs_fit"),
    "em":        ("cli_em_fitter",   "ausaxs_em"),
    "rigidbody": ("cli_rigidbody",   "ausaxs_rigidbody"),
}


class _Formatter(argparse.HelpFormatter):
    def _format_action(self, action):
        result = super()._format_action(action)
        if action.nargs == argparse.PARSER:
            lines = result.splitlines(keepends=True)
            return ''.join(
                line for line in lines[1:] if argparse.SUPPRESS not in line
            )
        return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ausaxs",
        description="An efficient small-angle Debye scattering calculator.",
        formatter_class=_Formatter,
    )
    parser.add_argument("-v", "--version", action="version", version=f"pyAUSAXS {__version__}")
    parser.add_argument("--license", action="store_true", help="show license information and exit")
    sub = parser.add_subparsers(dest="tool", title="tools")

    # C CLI tools — argparse does not consume their flags; everything after the subcommand name is passed through to the backend as-is.
    for name, desc in [
        ("fit",       "validate structures against SAXS data"),
        ("em",        "validate EM maps against SAXS data"),
        ("rigidbody", "perform rigid-body refinements"),
        ("plot",      "plot results from other tools"),
        ("gui",       "start the graphical interface"),
    ]:
        sub.add_parser(name, help=desc, add_help=False)

    # 'setup' is unlisted and has its own structured flags.
    setup = sub.add_parser(
        "setup", help=argparse.SUPPRESS,
        description="Configure the AUSAXS backend.",
    )
    grp = setup.add_mutually_exclusive_group()
    grp.add_argument(
        "--relink", metavar="PATH",
        help="use the backend library at PATH instead of the bundled one",
    )
    grp.add_argument(
        "--reset", action="store_true",
        help="forget any relinked path and use the bundled library",
    )
    grp.add_argument(
        "--show", action="store_true",
        help="show which backend library is currently in use (default)",
    )

    # # Put the tools section above the options section in help output.
    by_title = {g.title: g for g in parser._action_groups}
    parser._action_groups[:] = [by_title["tools"], by_title["options"]]

    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = _build_parser()
    ns, remaining = parser.parse_known_args(argv)

    if ns.license:
        _print_licence()
        return 0

    if ns.tool is None:
        parser.print_help()
        return 0

    match ns.tool:
        case "setup":
            return _run_setup(ns)
        case "gui":
            print("Warning: The Python GUI is highly experimental. Use at your own risk.", file=sys.stderr)
            from .gui import main as gui_main
            return gui_main(remaining)
        case "plot":
            from .plot.plot import main as plot_main
            return plot_main(remaining)
        case t if t in _CLI_TOOLS:
            return _run_cli_tool(t, remaining)


def _run_cli_tool(tool, args):
    from .wrapper.AUSAXS import AUSAXS
    func_name, prog = _CLI_TOOLS[tool]
    lib = AUSAXS().lib()
    if not lib.ready():
        print("Error: AUSAXS library not ready", file=sys.stderr)
        return 1
    return _call_cli(getattr(lib.functions, func_name), [prog, *args])


def _call_cli(cli_func, args):
    import ctypes as ct
    c_args = [arg.encode("utf-8") for arg in args]
    c_argv = (ct.c_char_p * len(c_args))(*c_args)
    return cli_func(len(c_args), c_argv)


def _run_setup(ns):
    from pathlib import Path
    from . import loader
    from .wrapper.AUSAXS import AUSAXS

    if ns.relink:
        AUSAXS()
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
        AUSAXS()
        if loader.clear_relink_path():
            print("Reset: now using the bundled AUSAXS backend.")
        else:
            print("No relink configured; already using the bundled backend.")
        return 0

    # default: --show
    AUSAXS()
    print(f"AUSAXS backend in use: {loader.find_lib_path()}")
    if loader.get_relink_path():
        print("  (relinked; reset with: ausaxs setup --reset)")
    else:
        print("  (bundled)")
    return 0


def _print_licence():
    import importlib.resources as pkg_resources
    try:
        ref = pkg_resources.files("pyausaxs").joinpath("../../LICENSE")
        with pkg_resources.as_file(ref) as p:
            print(p.read_text(encoding="utf-8"))
    except Exception:
        print("LGPL-3.0-or-later — see https://github.com/AUSAXS/pyAUSAXS/blob/master/LICENSE")


if __name__ == "__main__":
    raise SystemExit(main())
