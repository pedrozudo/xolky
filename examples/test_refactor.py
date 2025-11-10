import time
from contextlib import contextmanager


import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import xolky


@contextmanager
def time_block(label="Block"):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{label} took {end - start:.4f} seconds")


def main():
    row = jnp.array([0, 1, 2, 0, 1])
    col = jnp.array([0, 1, 2, 1, 0])

    data1 = jnp.array([2.0, 3.0, 3.0, 1.0, 1.0])
    b1 = jnp.array([1.0, 2.0, 3.0])

    data2 = jnp.array([1.0, 4.0, 3.0, 0.5, 0.5])
    b2 = jnp.array([3.0, 1.0, 2.0])

    n_rows = b1.shape[0]
    nzz = data1.shape[0]

    ################################################################

    A1 = (
        jsparse.COO((data1, row, col), shape=(n_rows, n_rows))._sort_indices().todense()
    )
    A2 = (
        jsparse.COO((data2, row, col), shape=(n_rows, n_rows))._sort_indices().todense()
    )

    x1_dense = jnp.linalg.solve(A1, b1)
    x2_dense = jnp.linalg.solve(A2, b2)

    ################################################################

    A1 = jsparse.csr_fromdense(A1)
    A2 = jsparse.csr_fromdense(A2)

    csr_inds = A1.indices
    csr_ptrs = A1.indptr

    solver = xolky.SparseCholesky(n_rows, nzz, csr_inds, csr_ptrs)

    solver.reorder()
    solver.analyze()

    ################################################################

    csr_data = A1.data
    solver.factorize(csr_data)

    solver.solve = jax.jit(solver.solve)

    with time_block("Solve 1"):
        x1_sparse = solver.solve(b1)

    ################################################################

    csr_data = A2.data
    solver.refactorize(csr_data)
    with time_block("Solve 2"):
        x2_sparse = solver.solve(b2)

    ################################################################

    print("Problem 1")
    print(f"Dense solution: {x1_dense}")
    print(f"Sparse solution: {x1_sparse}")
    print()

    print("Problem 2")
    print(f"Dense solution: {x2_dense}")
    print(f"Sparse solution: {x2_sparse}")


if __name__ == "__main__":
    main()
