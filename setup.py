import os
from pathlib import Path
import shutil
import subprocess
import sysconfig

import jaxlib
import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


CUDA_ROOT = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
_configured_cudss_root = os.environ.get("CUDSS_ROOT")
CUDSS_ROOT = Path(_configured_cudss_root) if _configured_cudss_root else None
_configured_cholmod_root = os.environ.get("CHOLMOD_ROOT")
CHOLMOD_ROOT = Path(_configured_cholmod_root) if _configured_cholmod_root else None


def _build_mode(variable):
    value = os.environ.get(variable, "auto").lower()
    value = {"1": "on", "true": "on", "0": "off", "false": "off"}.get(
        value, value
    )
    if value not in {"auto", "on", "off"}:
        raise RuntimeError(f"{variable} must be one of auto, on, or off")
    return value


CUDA_BUILD = _build_mode("XOLKY_BUILD_CUDA")
CHOLMOD_BUILD = _build_mode("XOLKY_BUILD_CHOLMOD")


def get_cuda_path(subpath):
    return os.fspath(CUDA_ROOT / subpath)


def get_jaxlib_path(subpath):
    base = os.path.dirname(jaxlib.__file__)
    return os.path.join(base, subpath)


def _pkg_config_directory(package, variable):
    executable = shutil.which("pkg-config")
    if executable is None:
        return None
    result = subprocess.run(
        [executable, "--variable", variable, package],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return Path(result.stdout.strip())


def _multiarch_library_directories():
    multiarch = sysconfig.get_config_var("MULTIARCH")
    if not multiarch:
        return ()
    return (Path("/usr/lib") / multiarch, Path("/lib") / multiarch)


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


def _cholmod_candidates():
    if CHOLMOD_ROOT is not None:
        return (
            (
                CHOLMOD_ROOT / "include" / "suitesparse" / "cholmod.h",
                CHOLMOD_ROOT / "include" / "cholmod.h",
            ),
            (
                CHOLMOD_ROOT / "lib64" / "libcholmod.so",
                CHOLMOD_ROOT / "lib" / "libcholmod.so",
            ),
        )

    include_root = _pkg_config_directory("cholmod", "includedir")
    library_root = _pkg_config_directory("cholmod", "libdir")
    headers = []
    libraries = []
    if include_root is not None:
        headers.extend(
            (
                include_root / "suitesparse" / "cholmod.h",
                include_root / "cholmod.h",
            )
        )
    if library_root is not None:
        libraries.append(library_root / "libcholmod.so")

    headers.extend(
        (
            Path("/usr/include/suitesparse/cholmod.h"),
            Path("/usr/local/include/suitesparse/cholmod.h"),
            Path("/usr/local/include/cholmod.h"),
        )
    )
    libraries.extend(
        path / "libcholmod.so" for path in _multiarch_library_directories()
    )
    libraries.extend(
        (
            Path("/usr/lib64/libcholmod.so"),
            Path("/usr/lib/libcholmod.so"),
            Path("/usr/local/lib/libcholmod.so"),
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
CHOLMOD_HEADER_CANDIDATES, CHOLMOD_LIBRARY_CANDIDATES = _cholmod_candidates()
CHOLMOD_HEADER = next(
    (path for path in CHOLMOD_HEADER_CANDIDATES if path.is_file()),
    None,
)
CHOLMOD_LIBRARY = next(
    (path for path in CHOLMOD_LIBRARY_CANDIDATES if path.is_file()),
    None,
)

CUDA_REQUESTED = CUDA_BUILD == "on" or CUDSS_ROOT is not None
CHOLMOD_REQUESTED = CHOLMOD_BUILD == "on" or CHOLMOD_ROOT is not None
BUILD_CUDA = bool(
    CUDA_BUILD != "off" and CUDSS_HEADER is not None and CUDSS_LIBRARY is not None
)
BUILD_CHOLMOD = bool(
    CHOLMOD_BUILD != "off"
    and CHOLMOD_HEADER is not None
    and CHOLMOD_LIBRARY is not None
)


class XolkyBuildExt(build_ext):
    def run(self):
        if CUDA_REQUESTED and (CUDSS_HEADER is None or CUDSS_LIBRARY is None):
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
        if CHOLMOD_REQUESTED and (
            CHOLMOD_HEADER is None or CHOLMOD_LIBRARY is None
        ):
            searched = "\n".join(
                f"  - {path}"
                for path in (
                    *CHOLMOD_HEADER_CANDIDATES,
                    *CHOLMOD_LIBRARY_CANDIDATES,
                )
            )
            raise RuntimeError(
                "CHOLMOD development files were not found. Searched:\n"
                f"{searched}\n"
                "Install SuiteSparse development files or set CHOLMOD_ROOT "
                "to their installation prefix."
            )
        super().run()


common_include_dirs = [pybind11.get_include(), get_jaxlib_path("include")]
common_compile_args = ["-O3", "-std=c++17", "-fPIC"]
ext_modules = []

if BUILD_CUDA:
    ext_modules.append(
        Extension(
            name="xolky._xolky_cuda",
            sources=["xolky/_xolky_cudss.cpp"],
            include_dirs=[
                *common_include_dirs,
                get_cuda_path("include"),
                os.fspath(CUDSS_HEADER.parent),
                "xolky",
            ],
            library_dirs=[
                get_cuda_path("lib64"),
                os.fspath(CUDSS_LIBRARY.parent),
            ],
            libraries=["cudart", "cudss"],
            language="c++",
            extra_compile_args=common_compile_args,
        )
    )

if BUILD_CHOLMOD:
    ext_modules.append(
        Extension(
            name="xolky._xolky_cholmod",
            sources=["xolky/_xolky_cholmod.cpp"],
            include_dirs=[*common_include_dirs, os.fspath(CHOLMOD_HEADER.parent)],
            library_dirs=[os.fspath(CHOLMOD_LIBRARY.parent)],
            libraries=["cholmod"],
            language="c++",
            extra_compile_args=common_compile_args,
        )
    )

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": XolkyBuildExt},
)
