import functools

import jax
import jax.numpy as jnp
from jax import ffi as jffi


from . import _xolky


jffi.register_ffi_target(
    "xolky_init_structure", _xolky.init_structure(), platform="CUDA"
)
jffi.register_ffi_target("xolky_reorder", _xolky.reorder(), platform="CUDA")
jffi.register_ffi_target("xolky_analyze", _xolky.analyze(), platform="CUDA")
jffi.register_ffi_target("xolky_factorize", _xolky.factorize(), platform="CUDA")
jffi.register_ffi_target("xolky_refactorize", _xolky.refactorize(), platform="CUDA")
jffi.register_ffi_target("xolky_solve", _xolky.solve(), platform="CUDA")


class SparseCholesky:
    def __init__(self, ncols, nnz, csr_indices, csr_indptr):
        assert csr_indices.dtype == jnp.int32, "csr_indices must be int32 type"
        assert csr_indptr.dtype == jnp.int32, "csr_intptr must be int32 type"

        self.ncols = ncols
        self.nnz = nnz
        self.csr_indcs = csr_indices
        self.csr_ptrs = csr_indptr

        self._solver = _xolky.CuDssSparseCholesky()

        _init_structure = jffi.ffi_call(
            "xolky_init_structure",
            result_shape_dtypes=[],
            has_side_effect=True,
        )
        self._init_structure = functools.partial(
            _init_structure, address=self.address()
        )

        _reorder = jffi.ffi_call(
            "xolky_reorder",
            result_shape_dtypes=[],
            has_side_effect=True,
        )
        self._reorder = functools.partial(_reorder, address=self.address())

        _analyze = jffi.ffi_call(
            "xolky_analyze",
            result_shape_dtypes=[],
            has_side_effect=True,
        )
        self._analyze = functools.partial(_analyze, address=self.address())

        _factorize = jffi.ffi_call(
            "xolky_factorize",
            result_shape_dtypes=[],
            has_side_effect=True,
        )
        self._factorize = functools.partial(_factorize, address=self.address())

        _refactorize = jffi.ffi_call(
            "xolky_refactorize",
            result_shape_dtypes=[],
            has_side_effect=True,
        )
        self._refactorize = functools.partial(_refactorize, address=self.address())

        _solve = jffi.ffi_call(
            "xolky_solve",
            jax.ShapeDtypeStruct((ncols,), jnp.float64),
        )
        self._solve = functools.partial(_solve, address=self.address())

        self._init_structure(
            self.csr_indcs,
            self.csr_ptrs,
            address=self.address(),
            ncols=self.ncols,
            nnz=self.nnz,
        )

    def address(self):
        return self._solver.address()

    def reorder(self):
        self._reorder()

    def analyze(self):
        self._analyze()

    def factorize(self, csr_data):
        with jax.enable_x64():
            self._factorize(csr_data.astype(jnp.float64))

    def refactorize(self, csr_data):
        with jax.enable_x64():
            self._factorize(csr_data.astype(jnp.float64))

    def solve(self, b):
        with jax.enable_x64():
            x = self._solve(b.astype(jnp.float64))
        return x
