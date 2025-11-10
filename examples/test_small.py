import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import xolky


def solve(p):
    A = jsparse.COO((p[0], p[1], p[2]), shape=(p[4], p[4]))._sort_indices().todense()

    x_dense = jnp.linalg.solve(A, p[3])

    A = jsparse.csr_fromdense(A)

    size = A.shape[0]
    nse = A.nse

    csr_inds = A.indices
    csr_ptrs = A.indptr
    csr_data = A.data

    solver = xolky.SparseCholesky(size, nse, csr_inds, csr_ptrs)
    print()

    solver.reorder()
    solver.analyze()
    solver.factorize(csr_data)
    x_sparse = solver.solve(p[3])
    return x_dense, x_sparse


def collect_problems():
    problems = []

    data = jnp.array([4.0, 2.0, 3.0, 1.0, 1.0])
    row = jnp.array([0, 1, 2, 0, 1])
    col = jnp.array([0, 1, 2, 1, 0])
    b = jnp.array([1.0, 2.0, 3.0])
    n_rows = 3
    problems.append((data, row, col, b, n_rows))

    data = 0.5 * jnp.array([1.0, 1.0, 2.0, 2.0])
    row = jnp.array([0, 1, 2, 3])
    col = jnp.array([0, 1, 2, 3])
    b = jnp.array([1.0, 2.0, 3.0, 4.0])
    n_rows = 4
    problems.append((data, row, col, b, n_rows))

    data = jnp.array([1.0, 1.0, 2.0, 2.0, 10])
    row = jnp.array([0, 1, 2, 3, 4])
    col = jnp.array([0, 1, 2, 3, 4])
    b = jnp.array([1.0, 2.0, 3.0, 4.0, 1.0])
    n_rows = 5
    problems.append((data, row, col, b, n_rows))

    data = jnp.array([4.0, 2.0, 2.0, 2.0, 10, 1.0, 1.0])
    row = jnp.array([0, 1, 2, 3, 4, 0, 1])
    col = jnp.array([0, 1, 2, 3, 4, 1, 0])
    b = jnp.array([1.0, 2.0, 3.0, 4.0, 1.0])
    n_rows = 5
    problems.append((data, row, col, b, n_rows))

    return problems


if __name__ == "__main__":
    problems = collect_problems()
    for i, p in enumerate(problems):
        x_dense, x_sparse = solve(p)

        print(
            f"Sparse and dense solution match for problem {i}: {jnp.all(jnp.isclose(x_dense, x_sparse))}"
        )
