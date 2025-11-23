from setuptools import setup, Extension
import importlib.util
import os
import pybind11
import jaxlib


# def get_nvidia_path(subpath):
#     spec = importlib.util.find_spec("nvidia")
#     base = spec.submodule_search_locations[0]
#     return os.path.join(base, subpath)


def get_cuda_path(subpath):
    base = "/usr/local/cuda"
    return os.path.join(base, subpath)


def get_jaxlib_path(subpath):
    base = os.path.dirname(jaxlib.__file__)
    return os.path.join(base, subpath)


ext_modules = [
    Extension(
        name="xolky._xolky",
        sources=["xolky/_xolky.cpp"],
        include_dirs=[
            pybind11.get_include(),
            get_jaxlib_path("include"),
            get_cuda_path("include"),
            "xolky",
        ],
        library_dirs=[get_cuda_path("lib64")],
        libraries=["cudart", "cudss"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17", "-fPIC"],
    ),
]

setup(
    ext_modules=ext_modules,
)
