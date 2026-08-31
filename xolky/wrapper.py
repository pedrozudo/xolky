from __future__ import annotations

import atexit
import dataclasses
import functools
from dataclasses import field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi as jffi


CholmodOrdering = Literal["auto", "amd", "metis", "nesdis"]
CholmodFactorization = Literal["auto", "simplicial", "supernodal"]
_CHOLMOD_ORDERINGS = frozenset(("auto", "amd", "metis", "nesdis"))
_CHOLMOD_FACTORIZATIONS = frozenset(("auto", "simplicial", "supernodal"))

try:
    from . import _xolky_cuda
except ImportError as error:
    _xolky_cuda = None
    _cuda_import_error = error
else:
    _cuda_import_error = None

try:
    from . import _xolky_cholmod
except ImportError as error:
    _xolky_cholmod = None
    _cholmod_import_error = error
else:
    _cholmod_import_error = None

if _xolky_cuda is not None:
    jffi.register_ffi_target("xolky_setup", _xolky_cuda.setup(), platform="CUDA")
    jffi.register_ffi_target(
        "xolky_refactor", _xolky_cuda.refactor(), platform="CUDA"
    )
    jffi.register_ffi_target("xolky_solve", _xolky_cuda.solve(), platform="CUDA")
    atexit.register(_xolky_cuda.shutdown)

if _xolky_cholmod is not None:
    jffi.register_ffi_target(
        "xolky_setup", _xolky_cholmod.setup(), platform="cpu"
    )
    jffi.register_ffi_target(
        "xolky_refactor", _xolky_cholmod.refactor(), platform="cpu"
    )
    jffi.register_ffi_target(
        "xolky_solve", _xolky_cholmod.solve(), platform="cpu"
    )
    atexit.register(_xolky_cholmod.shutdown)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SparseCholesky:
    """A linear handle to a native sparse Cholesky factorization.

    ``solver_id`` is a dynamic native-resource identifier and ``sequence``
    injects dependencies between stateful FFI calls. The remaining fields are
    immutable backend/shape metadata and participate in JIT caching. Previous
    values of a solver must not be reused after calling refactor or solve.
    """

    solver_id: jax.Array
    sequence: jax.Array
    n: int = field(metadata={"static": True})
    nnz: int = field(metadata={"static": True})
    device_ordinal: int = field(metadata={"static": True})
    backend: Literal["cuda", "cholmod"] = field(metadata={"static": True})
    ordering: CholmodOrdering | None = field(metadata={"static": True})
    factorization: CholmodFactorization | None = field(metadata={"static": True})

    def close(self) -> None:
        """Wait for pending operations and release native resources."""
        jax.block_until_ready(self.sequence)
        _backend_module(self.backend).destroy_solver(_concrete_solver_id(self))


def _backend_module(backend: Literal["cuda", "cholmod"]):
    if backend == "cuda":
        module = _xolky_cuda
        error = _cuda_import_error
    else:
        module = _xolky_cholmod
        error = _cholmod_import_error
    if module is None:
        detail = f": {error}" if error is not None else ""
        raise RuntimeError(f"xolky {backend} backend is unavailable{detail}")
    return module


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
        "constructed native batched solver instead"
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
        raise ValueError("xolky arrays must be placed on exactly one device")
    device = next(iter(devices))
    _device_backend(device)
    return device


def _device_backend(device: jax.Device) -> Literal["cuda", "cholmod"]:
    if device.platform == "cpu":
        return "cholmod"
    platform_version = getattr(device.client, "platform_version", "").lower()
    if device.platform == "gpu" and "cuda" in platform_version:
        return "cuda"
    raise ValueError(
        f"xolky supports JAX CPU and NVIDIA CUDA devices, not {device}"
    )


def _validate_solver(solver: SparseCholesky) -> None:
    if not isinstance(solver, SparseCholesky):
        raise TypeError("solver must be a SparseCholesky")
    solver_id_type = jax.typeof(solver.solver_id)
    if solver_id_type.shape != ():
        raise ValueError("solver_id must be scalar")
    if solver_id_type.dtype != np.dtype(np.uint64):
        raise TypeError("solver_id must have dtype uint64")
    if solver.backend not in ("cuda", "cholmod"):
        raise ValueError(f"unknown xolky backend {solver.backend!r}")
    if solver.backend == "cholmod":
        if solver.ordering not in _CHOLMOD_ORDERINGS:
            raise ValueError("a CHOLMOD solver must have an explicit ordering")
        if solver.factorization not in _CHOLMOD_FACTORIZATIONS:
            raise ValueError("a CHOLMOD solver must have a factorization policy")
    elif solver.ordering is not None or solver.factorization is not None:
        raise ValueError("a CUDA solver cannot have CHOLMOD policies")
    if (
        solver.backend == "cuda"
        and getattr(solver_id_type.memory_space, "name", None) != "Host"
    ):
        raise ValueError("solver_id must be stored in pinned host memory")


def setup(
    csr_indices: Any,
    csr_indptr: Any,
    *,
    ordering: CholmodOrdering | None = None,
    factorization: CholmodFactorization | None = None,
) -> SparseCholesky:
    """Create, reorder, and symbolically analyze an SPD CSR matrix structure.

    Setup is a host-side resource operation and must be called outside jit.
    The CSR structure is copied into backend-owned storage. CPU arrays require
    explicit CHOLMOD ordering and factorization policies. Each policy accepts
    ``auto`` or an explicit algorithm: ``amd``/``metis``/``nesdis`` and
    ``simplicial``/``supernodal``, respectively.
    """

    indices = _as_structure_array(csr_indices, "csr_indices")
    indptr = _as_structure_array(csr_indptr, "csr_indptr")
    if indptr.shape[0] < 2:
        raise ValueError("csr_indptr must describe a non-empty square matrix")

    indices_device = _array_device(indices)
    indptr_device = _array_device(indptr)
    if indices_device != indptr_device:
        raise ValueError("csr_indices and csr_indptr must be on the same device")

    backend = _device_backend(indices_device)
    if backend == "cholmod":
        if ordering is None:
            raise TypeError(
                "ordering is required for CHOLMOD setup; choose auto, amd, "
                "metis, or nesdis"
            )
        if factorization is None:
            raise TypeError(
                "factorization is required for CHOLMOD setup; choose auto, "
                "simplicial, or supernodal"
            )
        if ordering not in _CHOLMOD_ORDERINGS:
            raise ValueError("ordering must be auto, amd, metis, or nesdis")
        if factorization not in _CHOLMOD_FACTORIZATIONS:
            raise ValueError(
                "factorization must be auto, simplicial, or supernodal"
            )
    elif ordering is not None or factorization is not None:
        raise ValueError(
            "ordering and factorization are only valid for the CHOLMOD CPU "
            "backend"
        )

    native_module = _backend_module(backend)
    n = indptr.shape[0] - 1
    nnz = indices.shape[0]
    native_id = (
        native_module.create_solver(
            n, nnz, indices_device.id, ordering, factorization
        )
        if backend == "cholmod"
        else native_module.create_solver(n, nnz, indices_device.id)
    )

    try:
        with jax.enable_x64():
            solver_id_target = (
                _host_sharding(indices_device.id)
                if backend == "cuda"
                else indices_device
            )
            solver_id = jax.device_put(np.uint64(native_id), solver_id_target)
            sequence_seed = jax.device_put(np.uint8(0), indices_device)
            sequence = _setup_call()(
                solver_id,
                sequence_seed,
                indices,
                indptr,
            )
        jax.block_until_ready(sequence)
    except Exception:
        native_module.destroy_solver(native_id)
        raise

    return SparseCholesky(
        solver_id=solver_id,
        sequence=sequence,
        n=n,
        nnz=nnz,
        device_ordinal=indices_device.id,
        backend=backend,
        ordering=ordering,
        factorization=factorization,
    )


def refactor(solver: SparseCholesky, csr_values: Any) -> SparseCholesky:
    """Factorize new values using the solver's fixed CSR structure."""

    _validate_solver(solver)
    values = jnp.asarray(csr_values)
    if values.ndim != 1 or values.shape[0] != solver.nnz:
        raise ValueError(f"csr_values must have shape ({solver.nnz},)")
    if values.dtype != jnp.float64:
        raise TypeError("csr_values must have dtype float64")

    with jax.enable_x64():
        sequence = _refactor_ffi(
            solver.solver_id,
            solver.sequence,
            values,
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
    if rhs.dtype != jnp.float64:
        raise TypeError("right_hand_side must have dtype float64")

    with jax.enable_x64():
        sequence, result = _solve_ffi(solver.n)(
            solver.solver_id,
            solver.sequence,
            rhs,
        )
    return dataclasses.replace(solver, sequence=sequence), result
