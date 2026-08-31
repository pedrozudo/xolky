import jax
import jax.numpy as jnp
import numpy as np
import pytest

import xolky
from xolky.wrapper import _xolky_cuda as _xolky

from ._problems import device_problem


pytestmark = pytest.mark.cuda


def test_public_api_is_functional():
    assert xolky.__all__ == ["SparseCholesky", "setup", "refactor", "solve"]
    assert not hasattr(_xolky, "CuDssSparseCholesky")
    assert not hasattr(xolky.SparseCholesky, "__enter__")
    assert not hasattr(xolky.SparseCholesky, "__exit__")


def test_solver_is_a_pytree_with_dynamic_runtime_state():
    indices, indptr, _, _ = device_problem()
    solver = xolky.setup(indices, indptr)
    try:
        leaves, tree = jax.tree.flatten(solver)

        assert len(leaves) == 2
        assert leaves[0].dtype == jnp.uint64
        assert leaves[0].shape == ()
        assert leaves[1].dtype == jnp.uint8
        assert leaves[1].shape == ()
        assert "SparseCholesky" in str(tree)
        assert solver.n == 4
        assert solver.nnz == 8
        assert solver.backend == "cuda"
        assert solver.ordering is None
        assert solver.factorization is None
        assert jax.typeof(solver.solver_id).memory_space.name == "Host"
        assert jax.typeof(solver.sequence).memory_space.name == "Device"
        assert solver.sequence.devices() == indices.devices()
    finally:
        solver.close()


def test_setup_allocates_unique_ids_and_close_releases_resources():
    indices, indptr, _, _ = device_problem()
    first = xolky.setup(indices, indptr)
    second = xolky.setup(indices, indptr)
    try:
        assert int(np.asarray(first.solver_id)) != int(np.asarray(second.solver_id))
        assert _xolky.active_solver_count() == 2
    finally:
        first.close()
        second.close()

    assert _xolky.active_solver_count() == 0
    first.close()


@pytest.mark.parametrize(
    ("indices", "indptr", "error", "message"),
    [
        (
            jnp.array([[0]], dtype=jnp.int32),
            jnp.array([0, 1], dtype=jnp.int32),
            ValueError,
            "one-dimensional",
        ),
        (
            jnp.array([0], dtype=jnp.int64),
            jnp.array([0, 1], dtype=jnp.int32),
            TypeError,
            "dtype int32",
        ),
        (
            jnp.array([0], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            ValueError,
            "non-empty",
        ),
    ],
)
def test_setup_validates_structure(indices, indptr, error, message):
    with pytest.raises(error, match=message):
        xolky.setup(indices, indptr)


def test_cuda_setup_rejects_cholmod_policies():
    indices, indptr, _, _ = device_problem()
    with pytest.raises(ValueError, match="only valid for the CHOLMOD"):
        xolky.setup(
            indices, indptr, ordering="auto", factorization="auto"
        )


def test_setup_cannot_run_under_jax_transformations():
    indices, indptr, _, _ = device_problem()

    with pytest.raises(TypeError, match="JAX transformations"):
        jax.jit(xolky.setup)(indices, indptr)

    batched_indices = jnp.stack((indices, indices))
    with pytest.raises(TypeError, match="JAX transformations"):
        jax.vmap(lambda row: xolky.setup(row, indptr))(batched_indices)


def test_runtime_argument_validation():
    indices, indptr, values, _ = device_problem()
    solver = xolky.setup(indices, indptr)
    try:
        with pytest.raises(ValueError, match="csr_values"):
            xolky.refactor(solver, values[:-1])
        with pytest.raises(ValueError, match="right_hand_side"):
            xolky.solve(solver, jnp.ones(solver.n + 1))
    finally:
        solver.close()


@pytest.mark.parametrize(
    "dtype",
    [
        jnp.int32,
        jnp.float16,
        jnp.bfloat16,
        jnp.float32,
        jnp.complex64,
        jnp.complex128,
    ],
)
def test_runtime_numeric_arguments_require_float64(dtype):
    indices, indptr, _, _ = device_problem()
    solver = xolky.setup(indices, indptr)
    try:
        with pytest.raises(TypeError, match="csr_values must have dtype float64"):
            xolky.refactor(solver, jnp.ones(solver.nnz, dtype=dtype))
        with pytest.raises(
            TypeError,
            match="right_hand_side must have dtype float64",
        ):
            xolky.solve(solver, jnp.ones(solver.n, dtype=dtype))
    finally:
        solver.close()
