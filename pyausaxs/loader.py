import importlib.resources as pkg_resources

from pyausaxs.architecture import get_shared_lib_extension

def find_lib_path():
    ext = get_shared_lib_extension()
    lib_file = pkg_resources.files("pyausaxs").joinpath("resources", "libausaxs" + ext)
    with pkg_resources.as_file(lib_file) as p:
        return str(p)
    raise FileNotFoundError(f"AUSAXS: could not find library at expected path: {lib_file}")