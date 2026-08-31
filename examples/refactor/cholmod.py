import time
from contextlib import contextmanager

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
def refactor(solver, values):
    return xolky.refactor(solver, values)


@jax.jit
def solve(solver, rhs):
    return xolky.solve(solver, rhs)


@contextmanager
def time_block(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label} took {time.perf_counter() - start:.4f} seconds")


def main() -> None:
    device = jax.devices("cpu")[0]
    row = device_array(device, [0, 1, 2, 0, 1], np.int32)
    col = device_array(device, [0, 1, 2, 1, 0], np.int32)

    data1 = device_array(device, [2.0, 3.0, 3.0, 1.0, 1.0], np.float64)
    rhs1 = device_array(device, [1.0, 2.0, 3.0], np.float64)
    data2 = device_array(device, [1.0, 4.0, 3.0, 0.5, 0.5], np.float64)
    rhs2 = device_array(device, [3.0, 1.0, 2.0], np.float64)

    shape = (rhs1.shape[0], rhs1.shape[0])
    matrix1 = jsparse.COO((data1, row, col), shape=shape)._sort_indices().todense()
    matrix2 = jsparse.COO((data2, row, col), shape=shape)._sort_indices().todense()
    dense_solution1 = dense_solve(matrix1, rhs1)
    dense_solution2 = dense_solve(matrix2, rhs2)

    # Xolky consumes one triangle. Lower CSR maps directly to both backends.
    lower1 = jsparse.csr_fromdense(jnp.tril(matrix1))
    lower2 = jsparse.csr_fromdense(jnp.tril(matrix2))
    solver = xolky.setup(lower1.indices, lower1.indptr, ordering="auto", factorization="auto")

    try:
        with time_block("Refactor 1"):
            solver = refactor(solver, lower1.data)
            solver.sequence.block_until_ready()
        with time_block("Solve 1"):
            solver, sparse_solution1 = solve(solver, rhs1)
            sparse_solution1.block_until_ready()

        with time_block("Refactor 2"):
            solver = refactor(solver, lower2.data)
            solver.sequence.block_until_ready()
        with time_block("Solve 2"):
            solver, sparse_solution2 = solve(solver, rhs2)
            sparse_solution2.block_until_ready()

        print(f"backend: {solver.backend}")
        print(f"device: {sparse_solution1.device}")
        print(f"problem 1 dense:  {dense_solution1}")
        print(f"problem 1 sparse: {sparse_solution1}")
        print(f"problem 2 dense:  {dense_solution2}")
        print(f"problem 2 sparse: {sparse_solution2}")
    finally:
        solver.close()


if __name__ == "__main__":
    main()
