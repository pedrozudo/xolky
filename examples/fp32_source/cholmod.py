import jax
import jax.numpy as jnp
import numpy as np

import xolky


def device_array(device, values, dtype):
    return jax.device_put(np.asarray(values, dtype=dtype), device)


@jax.jit
def refactor_and_solve(solver, values, rhs):
    solver = xolky.refactor(solver, values)
    return xolky.solve(solver, rhs)


def main() -> None:
    device = jax.devices("cpu")[0]
    indices = device_array(device, [0, 0, 1, 2], np.int32)
    indptr = device_array(device, [0, 1, 3, 4], np.int32)
    values_fp32 = device_array(device, [4.0, 1.0, 3.0, 2.0], np.float32)
    rhs_fp32 = device_array(device, [1.0, 2.0, 3.0], np.float32)

    solver = xolky.setup(indices, indptr, ordering="auto", factorization="auto")
    try:
        # Conversion stays on the CPU device.
        with jax.enable_x64():
            values = values_fp32.astype(jnp.float64)
            rhs = rhs_fp32.astype(jnp.float64)
            solver, solution = refactor_and_solve(solver, values, rhs)
            solution_fp32 = solution.astype(jnp.float32)

        solution_fp32.block_until_ready()
        print(f"backend: {solver.backend}")
        print(f"device: {solution_fp32.device}")
        print(solution_fp32)
        print(solution_fp32.dtype)
    finally:
        solver.close()


if __name__ == "__main__":
    main()
