#!/usr/bin/env python3
import argparse
import os
import sys


def detect_library_path() -> str:
	plat = sys.platform
	if plat.startswith("linux"):
		return "resources/libausaxs.so"
	if plat == "darwin":
		return "resources/libausaxs.dylib"
	if plat.startswith("win") or plat == "cygwin":
		return "resources/libausaxs.dll"
	else:
		raise RuntimeError(f"Unsupported platform: {plat}")


lib = detect_library_path()
lib_relative = lib

template = "pyproject.template.toml"
if not os.path.exists(template):
    raise FileNotFoundError(f"Template not found!")

with open(template, "r", encoding="utf-8") as fh:
    template = fh.read()

content = template.replace("LIBRARY_PLACEHOLDER", lib)

with open("pyproject.toml", "w", encoding="utf-8") as fh:
    fh.write(content)