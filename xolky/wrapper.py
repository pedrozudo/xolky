from __future__ import annotations

import atexit
import dataclasses
import functools
from dataclasses import field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi as jffi

from . import _xolky


jffi.register_ffi_target("xolky_setup", _xolky.setup(), platform="CUDA")
jffi.register_ffi_target("xolky_refactor", _xolky.refactor(), platform="CUDA")
jffi.register_ffi_target("xolky_solve", _xolky.solve(), platform="CUDA")

atexit.register(_xolky.shutdown)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SparseCholesky:
    """A linear handle to a native cuDSS sparse Cholesky factorization.
    solver_id is a dynamic pinned-host identifier. sequence is an internal JAX
    dependency value,
    which injects linear dependencies in the computation graph
    of multiple solver refactor/solve calls.
    The remaining fields are immutable shape/device metadata
    and therefore participate in JIT caching. Previous values of a solver must
    not be reused after calling refactor or solve.
    """

    solver_id: jax.Array
    sequence: jax.Array
    n: int = field(metadata={"static": True})
    nnz: int = field(metadata={"static": True})
    device_ordinal: int = field(metadata={"static": True})

    def close(self) -> None:
        """Release all native CUDA and cuDSS resources owned by this solver."""
        _xolky.destroy_solver(_concrete_solver_id(self))

    def __enter__(self) -> SparseCholesky:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def _concrete_solver_id(solver: SparseCholesky) -> int:
    try:
        return int(np.asarray(solver.solver_id).item())
    except (TypeError, ValueError) as error:
        raise TypeError(
            "a solver can only be closed outside JAX transformations"
        ) from error


def _host_sharding(device_ordinal: int) -> jax.sharding.SingleDeviceSharding:
    devices = jax.devices("gpu")
    try:
        device = devices[device_ordinal]
    except IndexError as error:
        raise ValueError(f"CUDA device {device_ordinal} is not available") from error
    return jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host")


@functools.cache
def _setup_call():
    return jffi.ffi_call(
        "xolky_setup",
        jax.ShapeDtypeStruct((), np.dtype(np.uint8)),
        has_side_effect=True,
        input_output_aliases={1: 0},
    )


@functools.cache
def _refactor_call():
    return jffi.ffi_call(
        "xolky_refactor",
        jax.ShapeDtypeStruct((), np.dtype(np.uint8)),
        has_side_effect=True,
        input_output_aliases={1: 0},
    )


@functools.cache
def _solve_call(n: int):
    return jffi.ffi_call(
        "xolky_solve",
        (
            jax.ShapeDtypeStruct((), np.dtype(np.uint8)),
            jax.ShapeDtypeStruct((n,), np.dtype(np.float64)),
        ),
        has_side_effect=True,
        input_output_aliases={1: 0},
    )

def _vmap_not_supported(operation: str) -> None:
    raise NotImplementedError(
        f"xolky.{operation} does not support jax.vmap; use an explicitly "
        "constructed native cuDSS batched solver instead"
    )


@jax.custom_batching.custom_vmap
def _refactor_ffi(
    solver_id: jax.Array,
    sequence: jax.Array,
    values: jax.Array,
) -> jax.Array:
    return _refactor_call()(solver_id, sequence, values)


@_refactor_ffi.def_vmap
def _refactor_ffi_vmap(axis_size, in_batched, solver_id, sequence, values):
    del axis_size, in_batched, solver_id, sequence, values
    _vmap_not_supported("refactor")


@functools.cache
def _solve_ffi(n: int):
    @jax.custom_batching.custom_vmap
    def call(
        solver_id: jax.Array,
        sequence: jax.Array,
        right_hand_side: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return _solve_call(n)(solver_id, sequence, right_hand_side)

    @call.def_vmap
    def call_vmap(
        axis_size, in_batched, solver_id, sequence, right_hand_side
    ):
        del axis_size, in_batched, solver_id, sequence, right_hand_side
        _vmap_not_supported("solve")

    return call

def _as_structure_array(value: Any, name: str) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype != jnp.int32:
        raise TypeError(f"{name} must have dtype int32")
    return array


def _array_device(array: jax.Array) -> jax.Device:
    if isinstance(array, jax.core.Tracer) or not hasattr(array, "devices"):
        raise TypeError(
            "xolky.setup is a host resource operation and cannot run under "
            "JAX transformations"
        )
    devices = array.devices()
    if len(devices) != 1:
        raise ValueError("xolky arrays must be placed on exactly one CUDA device")
    device = next(iter(devices))
    if device.platform != "gpu":
        raise ValueError("xolky requires arrays placed on a CUDA device")
    return device


def _validate_solver(solver: SparseCholesky) -> None:
    if not isinstance(solver, SparseCholesky):
        raise TypeError("solver must be a SparseCholesky")
    solver_id_type = jax.typeof(solver.solver_id)
    if solver_id_type.shape != ():
        raise ValueError("solver_id must be scalar")
    if solver_id_type.dtype != np.dtype(np.uint64):
        raise TypeError("solver_id must have dtype uint64")
    if getattr(solver_id_type.memory_space, "name", None) != "Host":
        raise ValueError("solver_id must be stored in pinned host memory")


def setup(csr_indices: Any, csr_indptr: Any) -> SparseCholesky:
    """Create, reorder, and symbolically analyze an SPD CSR matrix structure.

    Setup is a host-side resource operation and must be called outside jit.
    The CSR structure is copied into solver-owned device storage.
    """

    indices = _as_structure_array(csr_indices, "csr_indices")
    indptr = _as_structure_array(csr_indptr, "csr_indptr")
    if indptr.shape[0] < 2:
        raise ValueError("csr_indptr must describe a non-empty square matrix")

    indices_device = _array_device(indices)
    indptr_device = _array_device(indptr)
    if indices_device != indptr_device:
        raise ValueError("csr_indices and csr_indptr must be on the same device")

    n = indptr.shape[0] - 1
    nnz = indices.shape[0]
    native_id = _xolky.create_solver(n, nnz, indices_device.id)

    try:
        with jax.enable_x64():
            solver_id = jax.device_put(
                np.uint64(native_id),
                _host_sharding(indices_device.id),
            )
            sequence = _setup_call()(
                solver_id,
                jnp.asarray(0, dtype=jnp.uint8),
                indices,
                indptr,
            )
        jax.block_until_ready(sequence)
    except Exception:
        _xolky.destroy_solver(native_id)
        raise

    return SparseCholesky(
        solver_id=solver_id,
        sequence=sequence,
        n=n,
        nnz=nnz,
        device_ordinal=indices_device.id,
    )


def refactor(solver: SparseCholesky, csr_values: Any) -> SparseCholesky:
    """Factorize new values using the solver's fixed CSR structure."""

    _validate_solver(solver)
    values = jnp.asarray(csr_values)
    if values.ndim != 1 or values.shape[0] != solver.nnz:
        raise ValueError(f"csr_values must have shape ({solver.nnz},)")
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        raise TypeError("csr_values must have a floating-point dtype")

    with jax.enable_x64():
        sequence = _refactor_ffi(
            solver.solver_id,
            solver.sequence,
            values.astype(jnp.float64),
        )
    return dataclasses.replace(solver, sequence=sequence)


def solve(
    solver: SparseCholesky,
    right_hand_side: Any,
) -> tuple[SparseCholesky, jax.Array]:
    """Solve with the current numeric factorization."""

    _validate_solver(solver)
    rhs = jnp.asarray(right_hand_side)
    if rhs.ndim != 1 or rhs.shape[0] != solver.n:
        raise ValueError(f"right_hand_side must have shape ({solver.n},)")
    if not jnp.issubdtype(rhs.dtype, jnp.inexact):
        raise TypeError("right_hand_side must have a floating-point dtype")

    with jax.enable_x64():
        sequence, result = _solve_ffi(solver.n)(
            solver.solver_id,
            solver.sequence,
            rhs.astype(jnp.float64),
        )
    return dataclasses.replace(solver, sequence=sequence), result.astype(rhs.dtype)
