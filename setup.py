# setup.py (Linux-only, minimal + fixed)
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import setuptools
import importlib.util
import pybind11
import os
import shutil
import jaxlib


def get_nvidia_path(lib_name):
    spec = importlib.util.find_spec("nvidia")
    lib_path = spec.submodule_search_locations[0]
    return lib_path + lib_name


def get_jaxlib_path():
    return os.path.dirname(jaxlib.__file__)


def locate_cuda():
    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    if not shutil.which(nvcc):
        raise RuntimeError("nvcc not found. Install CUDA Toolkit or set CUDA_HOME.")
    return {
        "home": cuda_home,
        "nvcc": nvcc,
        "include": os.path.join(cuda_home, "include"),
        "lib": os.path.join(cuda_home, "lib64"),
    }


CUDA = locate_cuda()


class BuildExtNVCC(build_ext):
    def build_extensions(self):
        comp = self.compiler
        if ".cu" not in comp.src_extensions:
            comp.src_extensions.append(".cu")

        default_compile = comp.compile

        def compile_with_nvcc(
            sources,
            output_dir=None,
            macros=None,
            include_dirs=None,
            debug=0,
            extra_preargs=None,
            extra_postargs=None,
            depends=None,
        ):
            import os

            include_dirs = include_dirs or []
            extra_preargs = extra_preargs or []
            extra_postargs = extra_postargs or []

            cu_sources = [s for s in sources if s.endswith(".cu")]
            cpp_sources = [s for s in sources if not s.endswith(".cu")]

            cu_objects = []
            if cu_sources:
                archs = os.environ.get("CUDAARCHS")
                nvcc_flags = [
                    "-O3",
                    "-std=c++17",
                    "-Xcompiler",
                    "-fPIC",
                    "--expt-relaxed-constexpr",
                ]
                if archs:
                    for a in archs.replace(",", ";").split(";"):
                        a = a.strip()
                        if a:
                            nvcc_flags += [f"-gencode=arch=compute_{a},code=sm_{a}"]
                    nums = [
                        aa for aa in archs.replace(",", ";").split(";") if aa.isdigit()
                    ]
                    if nums:
                        top = sorted(nums, key=int)[-1]
                        nvcc_flags += [
                            f"-gencode=arch=compute_{top},code=compute_{top}"
                        ]
                else:
                    for a in ("75", "80", "86", "89", "90"):
                        nvcc_flags += [f"-gencode=arch=compute_{a},code=sm_{a}"]
                    nvcc_flags += ["-gencode=arch=compute_90,code=compute_90"]

                inc = sum((["-I", d] for d in include_dirs), [])
                for src in cu_sources:
                    obj = comp.object_filenames([src], output_dir=output_dir)[0]
                    os.makedirs(os.path.dirname(obj), exist_ok=True)
                    self.spawn([CUDA["nvcc"], "-c", src, "-o", obj] + inc + nvcc_flags)
                    cu_objects.append(obj)

            cpp_objects = []
            if cpp_sources:
                cxx_flags = ["-O3", "-std=c++17", "-fPIC"]
                cpp_objects = default_compile(
                    cpp_sources,
                    output_dir=output_dir,
                    macros=macros,
                    include_dirs=include_dirs,
                    debug=debug,
                    extra_preargs=extra_preargs,
                    extra_postargs=extra_postargs + cxx_flags,
                    depends=depends,
                )

            return cu_objects + cpp_objects

        comp.compile = compile_with_nvcc
        super().build_extensions()


ext_modules = [
    Extension(
        name="xolky._xolky",
        sources=[
            "xolky/_xolky.cpp",
            "xolky/_cast.cu",
        ],
        include_dirs=[
            pybind11.get_include(),
            get_nvidia_path("cudart") + "/include",
            get_jaxlib_path() + "/include",
            "xolky",
            CUDA["include"],
        ],
        libraries=["cudart", "cudss"],
        library_dirs=[
            get_nvidia_path("cudart") + "/lib",
            CUDA["lib"],
        ],
        language="c++",
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtNVCC},
)
