import time
from contextlib import contextmanager
from pathlib import Path

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
import numpy as np

import xolky


jax.config.update("jax_enable_x64", True)


def device_array(device, values, dtype):
    return jax.device_put(np.asarray(values, dtype=dtype), device)


@jax.jit
def cg_solve(matrix, rhs):
    return cg(matrix, rhs, tol=1e-6, maxiter=None)


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


def load_problem(device):
    with np.load(Path(__file__).with_name("nd6k.npz")) as archive:
        shape = tuple(int(value) for value in archive["shape"])
        full_indices_host = np.asarray(archive["indices"], dtype=np.int32)
        full_indptr_host = np.asarray(archive["indptr"], dtype=np.int32)
        full_values_host = np.asarray(archive["data"], dtype=np.float64)

    # The archive stores both triangles. Xolky consumes lower-triangular CSR.
    rows = np.repeat(
        np.arange(shape[0], dtype=np.int32),
        np.diff(full_indptr_host),
    )
    lower_mask = full_indices_host <= rows
    lower_indices_host = full_indices_host[lower_mask]
    lower_values_host = full_values_host[lower_mask]
    lower_counts = np.bincount(rows[lower_mask], minlength=shape[0])
    lower_indptr_host = np.empty(shape[0] + 1, dtype=np.int32)
    lower_indptr_host[0] = 0
    lower_indptr_host[1:] = np.cumsum(lower_counts, dtype=np.int64)

    full_indices = device_array(device, full_indices_host, np.int32)
    full_indptr = device_array(device, full_indptr_host, np.int32)
    full_values = device_array(device, full_values_host, np.float64)
    lower_indices = device_array(device, lower_indices_host, np.int32)
    lower_indptr = device_array(device, lower_indptr_host, np.int32)
    lower_values = device_array(device, lower_values_host, np.float64)
    rhs = device_array(device, np.ones(shape[0]), np.float64)

    full_matrix = jsparse.CSR(
        (full_values, full_indices, full_indptr),
        shape=shape,
    )
    return full_matrix, lower_indices, lower_indptr, lower_values, rhs


def main() -> None:
    device = jax.devices("gpu")[0]
    matrix, indices, indptr, values, rhs = load_problem(device)

    print(f"full non-zeros: {matrix.nse}")
    print(f"lower-triangle non-zeros: {values.shape[0]}")
    print(f"matrix shape: {matrix.shape}")

    with time_block("JIT CG solve"):
        cg_solution, _ = cg_solve(matrix, rhs)
        cg_solution.block_until_ready()

    solver = xolky.setup(indices, indptr)
    try:
        with time_block("JIT xolky refactor"):
            solver = refactor(solver, values)
            solver.sequence.block_until_ready()

        with time_block("JIT xolky solve"):
            solver, solution = solve(solver, rhs)
            solution.block_until_ready()

        matches = bool(jnp.allclose(solution, cg_solution, rtol=1e-5, atol=1e-7))
        print(f"backend: {solver.backend}")
        print(f"device: {solution.device}")
        print(f"xolky and CG solutions match: {matches}")
    finally:
        solver.close()


if __name__ == "__main__":
    main()
