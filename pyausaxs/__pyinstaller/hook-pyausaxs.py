from PyInstaller.utils.hooks import collect_data_files
from pyausaxs.loader import find_lib_path
import os

datas = collect_data_files('pyausaxs')
hiddenimports = []
binaries = []

lib_path = find_lib_path()
if os.path.exists(lib_path):
    datas.append((lib_path, 'pyausaxs/resources'))
    binaries.append((lib_path, '.'))
else:
    raise FileNotFoundError(f"Library not found at expected path: {lib_path}")