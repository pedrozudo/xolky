import jax
import jax.numpy as jnp

import xolky


def main() -> None:
    indices = jnp.array([0, 0, 1, 2], dtype=jnp.int32)
    indptr = jnp.array([0, 1, 3, 4], dtype=jnp.int32)

    values_fp32 = jnp.array([4.0, 1.0, 3.0, 2.0], dtype=jnp.float32)
    rhs_fp32 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

    solver = xolky.setup(indices, indptr)

    # enable_x64 permits explicit FP64 array creation in this block. It does
    # not implicitly promote the existing FP32 source arrays.
    with jax.enable_x64():
        values = values_fp32.astype(jnp.float64)
        rhs = rhs_fp32.astype(jnp.float64)

        solver = xolky.refactor(solver, values)
        solver, solution = xolky.solve(solver, rhs)
        solution = solution.astype(jnp.float32)

    solver.close()
    print(solution)
    print(solution.dtype)


if __name__ == "__main__":
    main()
