from PyInstaller.utils.hooks import collect_data_files
import os
import platform

datas = collect_data_files('pyausaxs')
hiddenimports = []
binaries = []
def get_expected_library_name():
    system = platform.system().lower()
    if system == 'windows':
        return 'ausaxs.dll'
    elif system == 'darwin':
        return 'libausaxs.dylib'
    else:
        return 'libausaxs.so'

try:
    import pyausaxs
    from pyausaxs.loader import find_lib_path
    
    try:
        lib_path = find_lib_path()
        if os.path.exists(lib_path):
            datas.append((lib_path, 'pyausaxs/resources'))
            binaries.append((lib_path, '.'))
            
    except Exception:
        import importlib.resources as pkg_resources
        expected_lib = get_expected_library_name()
        try:
            lib_file = pkg_resources.files("pyausaxs").joinpath("resources", expected_lib)
            with pkg_resources.as_file(lib_file) as p:
                if p.exists():
                    datas.append((str(p), 'pyausaxs/resources'))
                    binaries.append((str(p), '.'))
        except:
            pass

except ImportError:
    pass

