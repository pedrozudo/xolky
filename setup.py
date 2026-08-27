import os
from pathlib import Path
import sysconfig

import jaxlib
import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


CUDA_ROOT = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
_configured_cudss_root = os.environ.get("CUDSS_ROOT")
CUDSS_ROOT = Path(_configured_cudss_root) if _configured_cudss_root else None


def get_cuda_path(subpath):
    return os.fspath(CUDA_ROOT / subpath)


def get_jaxlib_path(subpath):
    base = os.path.dirname(jaxlib.__file__)
    return os.path.join(base, subpath)


def _cudss_candidates():
    if CUDSS_ROOT is not None:
        return (
            (CUDSS_ROOT / "include" / "cudss.h",),
            (
                CUDSS_ROOT / "lib64" / "libcudss.so",
                CUDSS_ROOT / "lib" / "libcudss.so",
            ),
        )

    prefixes = (CUDA_ROOT, Path("/usr/local/cuda"))
    headers = [prefix / "include" / "cudss.h" for prefix in prefixes]
    headers.append(Path("/usr/include/cudss.h"))

    libraries = []
    for prefix in prefixes:
        libraries.extend(
            (
                prefix / "lib64" / "libcudss.so",
                prefix / "lib" / "libcudss.so",
            )
        )

    multiarch = sysconfig.get_config_var("MULTIARCH")
    if multiarch:
        libraries.extend(
            (
                Path("/usr/lib") / multiarch / "libcudss.so",
                Path("/lib") / multiarch / "libcudss.so",
            )
        )
    libraries.extend(
        (
            Path("/usr/lib64/libcudss.so"),
            Path("/usr/lib/libcudss.so"),
            Path("/usr/local/lib/libcudss.so"),
        )
    )
    return tuple(headers), tuple(libraries)


CUDSS_HEADER_CANDIDATES, CUDSS_LIBRARY_CANDIDATES = _cudss_candidates()
CUDSS_HEADER = next(
    (path for path in CUDSS_HEADER_CANDIDATES if path.is_file()),
    None,
)
CUDSS_LIBRARY = next(
    (path for path in CUDSS_LIBRARY_CANDIDATES if path.is_file()),
    None,
)


class XolkyBuildExt(build_ext):
    def run(self):
        if CUDSS_HEADER is None or CUDSS_LIBRARY is None:
            searched = "\n".join(
                f"  - {path}"
                for path in (
                    *CUDSS_HEADER_CANDIDATES,
                    *CUDSS_LIBRARY_CANDIDATES,
                )
            )
            raise RuntimeError(
                "cuDSS development files were not found. Searched:\n"
                f"{searched}\n"
                "Install cuDSS or set CUDSS_ROOT to its installation prefix."
            )
        super().run()


ext_modules = [
    Extension(
        name="xolky._xolky",
        sources=["xolky/_xolky.cpp"],
        include_dirs=[
            pybind11.get_include(),
            get_jaxlib_path("include"),
            get_cuda_path("include"),
            os.fspath(CUDSS_HEADER.parent) if CUDSS_HEADER else "",
            "xolky",
        ],
        library_dirs=[
            get_cuda_path("lib64"),
            os.fspath(CUDSS_LIBRARY.parent) if CUDSS_LIBRARY else "",
        ],
        libraries=["cudart", "cudss"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17", "-fPIC"],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": XolkyBuildExt},
)
