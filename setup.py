"""Build the native polynomial evaluator for the centered search."""

import sys

from setuptools import Extension, setup

compile_args = ["/O2", "/std:c++17"] if sys.platform == "win32" else ["-O3", "-std=c++17"]

setup(
    ext_modules=[
        Extension(
            "quantum_splitting._iterated",
            ["src/quantum_splitting/_iterated.cpp"],
            language="c++",
            extra_compile_args=compile_args,
        )
    ]
)
