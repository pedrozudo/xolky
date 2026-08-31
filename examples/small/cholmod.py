import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np

import xolky


jax.config.update("jax_enable_x64", True)


def device_array(device, values, dtype):
    return jax.device_put(np.asarray(values, dtype=dtype), device)


@jax.jit
def dense_solve(matrix, rhs):
    return jnp.linalg.solve(matrix, rhs)


@jax.jit
def refactor_and_solve(solver, values, rhs):
    solver = xolky.refactor(solver, values)
    return xolky.solve(solver, rhs)


def collect_problems(device):
    specifications = [
        ([4.0, 2.0, 3.0, 1.0, 1.0], [0, 1, 2, 0, 1],
         [0, 1, 2, 1, 0], [1.0, 2.0, 3.0], 3),
        ([0.5, 0.5, 1.0, 1.0], [0, 1, 2, 3],
         [0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0], 4),
        ([1.0, 1.0, 2.0, 2.0, 10.0], [0, 1, 2, 3, 4],
         [0, 1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0, 1.0], 5),
        ([4.0, 2.0, 2.0, 2.0, 10.0, 1.0, 1.0],
         [0, 1, 2, 3, 4, 0, 1], [0, 1, 2, 3, 4, 1, 0],
         [1.0, 2.0, 3.0, 4.0, 1.0], 5),
    ]
    return [
        (
            device_array(device, data, np.float64),
            device_array(device, row, np.int32),
            device_array(device, col, np.int32),
            device_array(device, rhs, np.float64),
            size,
        )
        for data, row, col, rhs, size in specifications
    ]


def solve_problem(problem):
    data, row, col, rhs, size = problem
    matrix = (
        jsparse.COO((data, row, col), shape=(size, size))
        ._sort_indices()
        .todense()
    )
    expected = dense_solve(matrix, rhs)
    lower = jsparse.csr_fromdense(jnp.tril(matrix))

    solver = xolky.setup(lower.indices, lower.indptr, ordering="auto", factorization="auto")
    try:
        solver, actual = refactor_and_solve(solver, lower.data, rhs)
        actual.block_until_ready()
        return solver.backend, expected, actual
    finally:
        solver.close()


def main() -> None:
    device = jax.devices("cpu")[0]
    for index, problem in enumerate(collect_problems(device)):
        backend, expected, actual = solve_problem(problem)
        matches = bool(jnp.allclose(expected, actual))
        print(
            f"problem {index}: backend={backend}, device={actual.device}, "
            f"dense and sparse match={matches}"
        )


if __name__ == "__main__":
    main()
