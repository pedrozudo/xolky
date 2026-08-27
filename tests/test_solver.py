from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import xolky
from xolky import _xolky

from ._problems import VALUES_1, VALUES_2, dense, device_problem


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_factorize_and_solve_matches_dense(dtype):
    indices, indptr, values, _ = device_problem(dtype)
    rhs = jnp.asarray([1.0, -2.0, 3.0, 0.5], dtype=dtype)
    solver = xolky.setup(indices, indptr)
    try:
        solver = xolky.refactor(solver, values)
        solver, actual = xolky.solve(solver, rhs)

        expected = np.linalg.solve(dense(VALUES_1), np.asarray(rhs))
        assert actual.dtype == dtype
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    finally:
        solver.close()


def test_refactor_reuses_structure_for_new_values():
    indices, indptr, values_1, values_2 = device_problem()
    rhs_1 = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    rhs_2 = jnp.asarray([-1.0, 0.5, 2.0, 1.5])
    solver = xolky.setup(indices, indptr)
    try:
        solver = xolky.refactor(solver, values_1)
        solver, actual_1 = xolky.solve(solver, rhs_1)

        solver = xolky.refactor(solver, values_2)
        solver, actual_2 = xolky.solve(solver, rhs_2)

        np.testing.assert_allclose(
            actual_1, np.linalg.solve(dense(VALUES_1), np.asarray(rhs_1))
        )
        np.testing.assert_allclose(
            actual_2, np.linalg.solve(dense(VALUES_2), np.asarray(rhs_2))
        )
    finally:
        solver.close()


def test_state_threads_through_jit_and_reuses_compilation_for_different_ids():
    indices, indptr, values_1, values_2 = device_problem()
    rhs = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    traces = []

    def implementation(solver, values, right_hand_side):
        traces.append(None)
        solver = xolky.refactor(solver, values)
        return xolky.solve(solver, right_hand_side)

    compiled = jax.jit(implementation)
    first = xolky.setup(indices, indptr)
    second = xolky.setup(indices, indptr)
    try:
        first, actual_1 = compiled(first, values_1, rhs)
        second, actual_2 = compiled(second, values_2, rhs)
        jax.block_until_ready((actual_1, actual_2))

        assert len(traces) == 1
        np.testing.assert_allclose(
            actual_1, np.linalg.solve(dense(VALUES_1), np.asarray(rhs))
        )
        np.testing.assert_allclose(
            actual_2, np.linalg.solve(dense(VALUES_2), np.asarray(rhs))
        )
    finally:
        first.close()
        second.close()


def test_refactor_once_then_solve_repeatedly_in_compiled_loop():
    indices, indptr, values, _ = device_problem()
    right_hand_sides = jnp.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ]
    )

    @jax.jit
    def design_iteration(solver, matrix_values, rhs_values):
        solver = xolky.refactor(solver, matrix_values)

        def body(index, state):
            current_solver, solutions = state
            current_solver, solution = xolky.solve(
                current_solver, rhs_values[index]
            )
            return current_solver, solutions.at[index].set(solution)

        initial = jnp.zeros_like(rhs_values)
        return jax.lax.fori_loop(
            0, rhs_values.shape[0], body, (solver, initial)
        )

    solver = xolky.setup(indices, indptr)
    try:
        solver, actual = design_iteration(solver, values, right_hand_sides)
        expected = np.stack(
            [
                np.linalg.solve(dense(VALUES_1), rhs)
                for rhs in np.asarray(right_hand_sides)
            ]
        )
        np.testing.assert_allclose(actual, expected)
    finally:
        solver.close()

def test_refactor_explicitly_rejects_vmap():
    indices, indptr, values, _ = device_problem()
    solver = xolky.setup(indices, indptr)
    batched_values = jnp.stack((values, values))
    try:
        with pytest.raises(
            NotImplementedError,
            match=r"xolky\.refactor does not support jax\.vmap",
        ):
            jax.vmap(lambda item: xolky.refactor(solver, item))(
                batched_values
            )
    finally:
        solver.close()


def test_solve_explicitly_rejects_vmap():
    indices, indptr, values, _ = device_problem()
    solver = xolky.setup(indices, indptr)
    right_hand_sides = jnp.ones((2, solver.n))
    try:
        solver = xolky.refactor(solver, values)
        with pytest.raises(
            NotImplementedError,
            match=r"xolky\.solve does not support jax\.vmap",
        ):
            jax.vmap(lambda rhs: xolky.solve(solver, rhs)[1])(
                right_hand_sides
            )
    finally:
        solver.close()



def test_solve_before_factorization_reports_an_error():
    indices, indptr, values, _ = device_problem()
    solver = xolky.setup(indices, indptr)
    try:
        with pytest.raises(jax.errors.JaxRuntimeError, match="not been factorized"):
            _, result = xolky.solve(solver, jnp.ones(solver.n))
            result.block_until_ready()

        solver = xolky.refactor(solver, values)
        solver, result = xolky.solve(solver, jnp.ones(solver.n))
        np.testing.assert_allclose(
            result,
            np.linalg.solve(dense(VALUES_1), np.ones(solver.n)),
        )
    finally:
        solver.close()


def test_failed_solver_rejects_future_native_operations():
    indices, indptr, values, _ = device_problem()
    solver = xolky.setup(indices, indptr)
    solver_id = int(np.asarray(solver.solver_id))
    try:
        _xolky._poison_solver_for_testing(solver_id)

        with pytest.raises(
            jax.errors.JaxRuntimeError,
            match="solver is in a failed state; destroy and recreate it",
        ):
            failed = xolky.refactor(solver, values)
            failed.sequence.block_until_ready()
    finally:
        solver.close()

    replacement = xolky.setup(indices, indptr)
    try:
        replacement = xolky.refactor(replacement, values)
        replacement.sequence.block_until_ready()
    finally:
        replacement.close()


def test_same_solver_concurrent_solves_are_serialized_safely():
    indices, indptr, values, _ = device_problem()
    rhs_1 = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    rhs_2 = jnp.asarray([-2.0, 1.0, 0.5, 3.0])
    solver = xolky.setup(indices, indptr)
    solver = xolky.refactor(solver, values)
    solver.sequence.block_until_ready()

    @jax.jit
    def solve_only(current_solver, rhs):
        return xolky.solve(current_solver, rhs)[1]

    solve_only(solver, rhs_1).block_until_ready()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(lambda: np.asarray(solve_only(solver, rhs_1)))
            second = executor.submit(lambda: np.asarray(solve_only(solver, rhs_2)))
            actual_1 = first.result(timeout=20)
            actual_2 = second.result(timeout=20)

        np.testing.assert_allclose(
            actual_1, np.linalg.solve(dense(VALUES_1), np.asarray(rhs_1))
        )
        np.testing.assert_allclose(
            actual_2, np.linalg.solve(dense(VALUES_1), np.asarray(rhs_2))
        )
    finally:
        solver.close()

def test_closed_solver_identifier_is_rejected_by_native_registry():
    indices, indptr, values, _ = device_problem()
    solver = xolky.setup(indices, indptr)
    solver.close()

    try:
        with pytest.raises(jax.errors.JaxRuntimeError, match="unknown or closed"):
            stale = xolky.refactor(solver, values)
            stale.sequence.block_until_ready()
    finally:
        replacement = xolky.setup(indices, indptr)
        try:
            replacement = xolky.refactor(replacement, values)
            replacement.sequence.block_until_ready()
        finally:
            replacement.close()


def test_independent_solvers_can_execute_concurrently():
    indices, indptr, values_1, values_2 = device_problem()
    rhs_1 = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    rhs_2 = jnp.asarray([-2.0, 1.0, 0.5, 3.0])
    first = xolky.refactor(xolky.setup(indices, indptr), values_1)
    second = xolky.refactor(xolky.setup(indices, indptr), values_2)
    jax.block_until_ready((first.sequence, second.sequence))

    @jax.jit
    def solve_only(current_solver, rhs):
        return xolky.solve(current_solver, rhs)[1]

    solve_only(first, rhs_1).block_until_ready()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_1 = executor.submit(
                lambda: np.asarray(solve_only(first, rhs_1))
            )
            future_2 = executor.submit(
                lambda: np.asarray(solve_only(second, rhs_2))
            )
            actual_1 = future_1.result(timeout=20)
            actual_2 = future_2.result(timeout=20)

        np.testing.assert_allclose(
            actual_1, np.linalg.solve(dense(VALUES_1), np.asarray(rhs_1))
        )
        np.testing.assert_allclose(
            actual_2, np.linalg.solve(dense(VALUES_2), np.asarray(rhs_2))
        )
    finally:
        first.close()
        second.close()

