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
    def __init__(self, csr_indices, csr_indptr):
        if csr_indices.dtype != jnp.int32:
            raise TypeError("csr_indices must have dtype int32")
        if csr_indptr.dtype != jnp.int32:
            raise TypeError("csr_indptr must have dtype int32")

        self.n = csr_indptr.shape[0] - 1
        self.nnz = csr_indices.shape[0]
        self.csr_indices = csr_indices
        self.csr_indptr = csr_indptr

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
            jffi.abstract_token,
            has_side_effect=True,
        )
        self._factorize = functools.partial(_factorize, address=self.address())

        _refactorize = jffi.ffi_call(
            "xolky_refactorize",
            jffi.abstract_token,
            has_side_effect=True,
        )
        self._refactorize = functools.partial(_refactorize, address=self.address())

        _solve = jffi.ffi_call(
            "xolky_solve",
            (
                jffi.abstract_token,
                jax.ShapeDtypeStruct((self.n,), jnp.float64),
            ),
            has_side_effect=True,
        )
        self._solve = functools.partial(_solve, address=self.address())

        self._init_structure(
            self.csr_indices,
            self.csr_indptr,
            ncols=self.n,
            nnz=self.nnz,
        )

    def address(self):
        return self._solver.address()

    def reorder(self):
        self._reorder()

    def analyze(self):
        self._analyze()

    def factorize(self, token, csr_data):
        with jax.enable_x64():
            token = self._factorize(token, csr_data.astype(jnp.float64))
        return token

    def refactorize(self, token, csr_data):
        with jax.enable_x64():
            token = self._refactorize(token, csr_data.astype(jnp.float64))
        return token

    def solve(self, token, b):
        with jax.enable_x64():
            token, x = self._solve(token, b.astype(jnp.float64))
        return token, x.astype(b.dtype)
