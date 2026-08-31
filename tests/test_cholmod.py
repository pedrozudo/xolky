import jax
import jax.numpy as jnp
import numpy as np
import pytest

import xolky
from xolky.wrapper import _xolky_cholmod

from ._problems import INDICES, INDPTR, VALUES_1, VALUES_2, dense


pytestmark = pytest.mark.skipif(
    _xolky_cholmod is None,
    reason="the optional system CHOLMOD backend is not installed",
)


def cpu_array(value):
    return jax.device_put(value, jax.devices("cpu")[0])


def cpu_problem():
    return (
        cpu_array(INDICES),
        cpu_array(INDPTR),
        cpu_array(VALUES_1),
        cpu_array(VALUES_2),
    )


def test_cpu_setup_requires_an_explicit_ordering():
    indices, indptr, _, _ = cpu_problem()
    with pytest.raises(TypeError, match="ordering is required"):
        xolky.setup(indices, indptr)


def test_cpu_setup_requires_an_explicit_factorization():
    indices, indptr, _, _ = cpu_problem()
    with pytest.raises(TypeError, match="factorization is required"):
        xolky.setup(indices, indptr, ordering="auto")


@pytest.mark.parametrize("ordering", ["colamd", "natural", 1])
def test_cpu_setup_rejects_unknown_ordering(ordering):
    indices, indptr, _, _ = cpu_problem()
    with pytest.raises(ValueError, match="auto, amd, metis, or nesdis"):
        xolky.setup(
            indices, indptr, ordering=ordering, factorization="auto"
        )


@pytest.mark.parametrize("factorization", ["ldl", "multifrontal", 1])
def test_cpu_setup_rejects_unknown_factorization(factorization):
    indices, indptr, _, _ = cpu_problem()
    with pytest.raises(
        ValueError, match="auto, simplicial, or supernodal"
    ):
        xolky.setup(
            indices, indptr, ordering="auto", factorization=factorization
        )


def test_cpu_backend_factorizes_refactorizes_and_solves():
    indices, indptr, values_1, values_2 = cpu_problem()
    rhs = cpu_array(np.array([1.0, -2.0, 3.0, 0.5]))
    solver = xolky.setup(indices, indptr, ordering="auto", factorization="auto")
    try:
        assert solver.backend == "cholmod"
        assert solver.ordering == "auto"
        assert solver.factorization == "auto"
        assert solver.solver_id.devices() == {jax.devices("cpu")[0]}

        solver = xolky.refactor(solver, values_1)
        solver, first = xolky.solve(solver, rhs)
        solver = xolky.refactor(solver, values_2)
        solver, second = xolky.solve(solver, rhs)

        np.testing.assert_allclose(
            first, np.linalg.solve(dense(VALUES_1), np.asarray(rhs))
        )
        np.testing.assert_allclose(
            second, np.linalg.solve(dense(VALUES_2), np.asarray(rhs))
        )
    finally:
        solver.close()


def test_cpu_backend_threads_state_through_jit_and_control_flow():
    indices, indptr, values, _ = cpu_problem()
    right_hand_sides = cpu_array(np.eye(4, dtype=np.float64))

    @jax.jit
    def solve_many(solver, matrix_values, rhs_values):
        solver = xolky.refactor(solver, matrix_values)

        def body(index, state):
            current_solver, solutions = state
            current_solver, solution = xolky.solve(
                current_solver, rhs_values[index]
            )
            return current_solver, solutions.at[index].set(solution)

        return jax.lax.fori_loop(
            0,
            rhs_values.shape[0],
            body,
            (solver, jnp.zeros_like(rhs_values)),
        )

    solver = xolky.setup(indices, indptr, ordering="auto", factorization="auto")
    try:
        solver, actual = solve_many(solver, values, right_hand_sides)
        expected = np.stack(
            [
                np.linalg.solve(dense(VALUES_1), rhs)
                for rhs in np.asarray(right_hand_sides)
            ]
        )
        np.testing.assert_allclose(actual, expected)
    finally:
        solver.close()


def test_cholmod_solve2_workspace_is_reused():
    indices, indptr, values, _ = cpu_problem()
    rhs = cpu_array(np.ones(4, dtype=np.float64))
    solver = xolky.setup(indices, indptr, ordering="auto", factorization="auto")
    solver_id = int(np.asarray(solver.solver_id))
    try:
        solver = xolky.refactor(solver, values)
        solver, first = xolky.solve(solver, rhs)
        first.block_until_ready()
        assert _xolky_cholmod._workspace_allocated_for_testing(solver_id)

        solver, second = xolky.solve(solver, 2 * rhs)
        np.testing.assert_allclose(second, 2 * first)
        assert _xolky_cholmod._workspace_allocated_for_testing(solver_id)
    finally:
        solver.close()


@pytest.mark.parametrize(
    ("indices", "indptr", "message"),
    [
        ([1, 0, 1], [0, 1, 3], "lower triangle"),
        ([0, 0], [0, 1, 2], "diagonal entry"),
        ([0, 0, 0, 1], [0, 1, 4], "strictly increasing"),
        ([0, 0, 1], [0, 1, 4], "end with nnz"),
    ],
)
def test_cpu_setup_validates_csr_structure(indices, indptr, message):
    indices = cpu_array(np.asarray(indices, dtype=np.int32))
    indptr = cpu_array(np.asarray(indptr, dtype=np.int32))
    with pytest.raises(jax.errors.JaxRuntimeError, match=message):
        xolky.setup(indices, indptr, ordering="auto", factorization="auto")


def test_non_positive_definite_factorization_poison_solver():
    indices = cpu_array(np.array([0, 0, 1], dtype=np.int32))
    indptr = cpu_array(np.array([0, 1, 3], dtype=np.int32))
    values = cpu_array(np.array([-1.0, 0.0, 1.0]))
    solver = xolky.setup(indices, indptr, ordering="auto", factorization="auto")
    try:
        with pytest.raises(jax.errors.JaxRuntimeError, match="positive definite"):
            failed = xolky.refactor(solver, values)
            failed.sequence.block_until_ready()

        with pytest.raises(jax.errors.JaxRuntimeError, match="failed state"):
            failed_again = xolky.refactor(solver, values)
            failed_again.sequence.block_until_ready()
    finally:
        solver.close()


def test_cholmod_capabilities_report_runtime_version():
    capabilities = _xolky_cholmod.capabilities()
    assert len(capabilities["version"]) == 3
    assert capabilities["version"][0] >= 5
    assert capabilities["int32"] is True
