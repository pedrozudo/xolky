import os
import time
from contextlib import contextmanager

import numpy as np
import jax
import jax.experimental.sparse as jsparse
from jax.scipy.sparse.linalg import cg as jcg
import jax.numpy as jnp


import xolky

# cg need f64 to run correctly
jax.config.update("jax_enable_x64", True)


FILE_PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)


@contextmanager
def time_block(label="Block"):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{label} took {end - start:.4f} seconds")


if __name__ == "__main__":
    data = np.load(DIR_PATH + "/nd6k.npz")

    csr_inds = jnp.array(data["indices"], dtype=jnp.int32)
    csr_ptrs = jnp.array(data["indptr"], dtype=jnp.int32)
    csr_data = jnp.array(data["data"], dtype=jnp.float64)
    csr_shape = jnp.array(data["shape"])

    nse = csr_data.shape[0]
    size = np.int64(csr_shape[0])

    print(f"number of non-zeros: {nse}")
    print(f"number of columns: {size}")

    A = jsparse.CSR((csr_data, csr_inds, csr_ptrs), shape=csr_shape)
    b = jnp.ones(csr_shape[0], jnp.float64)

    ########################################
    ########################################
    ########################################

    x_cg = jnp.zeros_like(b)
    jax.block_until_ready(x_cg)
    with time_block("Solve CG"):
        x_cg, info = jcg(A, b, x_cg, tol=1e-6, maxiter=None)

    ########################################
    ########################################
    ########################################

    jax.block_until_ready(b)
    jax.block_until_ready(csr_data)

    print(jnp.max(csr_data))
    print(jnp.min(csr_data))

    solver = xolky.SparseCholesky(size, nse, csr_inds, csr_ptrs)

    with time_block("Reorder"):
        solver.reorder()

    with time_block("Analyze"):
        solver.analyze()

    with time_block("Factorize"):
        solver.factorize(csr_data)

    with time_block("Solve"):
        x = solver.solve(b)

    is_close = jnp.isclose(x, x_cg).all()
    print(f"CuDSS and CG solutions match: {is_close}")
