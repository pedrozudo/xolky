from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

import xolky
from xolky import _xolky


def tridiagonal_problem(
    size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if size < 2:
        raise ValueError("size must be at least 2")

    nnz = 2 * size - 1
    indptr = np.empty(size + 1, dtype=np.int32)
    indptr[0] = 0
    indptr[1:] = 2 * np.arange(1, size + 1, dtype=np.int32) - 1

    indices = np.empty(nnz, dtype=np.int32)
    indices[0] = 0
    indices[1::2] = np.arange(size - 1, dtype=np.int32)
    indices[2::2] = np.arange(1, size, dtype=np.int32)

    values = np.empty(nnz, dtype=np.float64)
    values[0] = 4.0
    values[1::2] = -1.0
    values[2::2] = 4.0

    rhs = np.ones(size, dtype=np.float64)
    return tuple(jnp.asarray(array) for array in (indices, indptr, values, rhs))


def benchmark(
    size: int,
    iterations: int,
    warmups: int,
    repetitions: int,
) -> tuple[list[float], float, int]:
    indices, indptr, values, rhs = tridiagonal_problem(size)

    @jax.jit
    def solve_loop(solver, right_hand_side):
        def body(_, state):
            current_solver, checksum = state
            current_solver, solution = xolky.solve(
                current_solver, right_hand_side
            )
            return current_solver, checksum + jnp.sum(solution)

        return jax.lax.fori_loop(
            0,
            iterations,
            body,
            (solver, jnp.asarray(0.0, dtype=right_hand_side.dtype)),
        )

    timings = []
    checksum_value = 0.0
    slot_count = 0
    with xolky.setup(indices, indptr) as solver:
        solver_id = int(np.asarray(solver.solver_id))
        solver = xolky.refactor(solver, values)
        solver.sequence.block_until_ready()

        for _ in range(warmups):
            solver, checksum = solve_loop(solver, rhs)
            checksum.block_until_ready()

        for _ in range(repetitions):
            start = time.perf_counter()
            solver, checksum = solve_loop(solver, rhs)
            checksum.block_until_ready()
            timings.append(time.perf_counter() - start)
            checksum_value = float(checksum)
        slot_count = _xolky._solve_slot_count_for_testing(solver_id)

    return timings, checksum_value, slot_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark repeated solves with one Xolky factorization."
    )
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    arguments = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    timings, checksum, slot_count = benchmark(
        arguments.size,
        arguments.iterations,
        arguments.warmups,
        arguments.repetitions,
    )

    median = statistics.median(timings)
    print(f"size: {arguments.size}")
    print(f"iterations per repetition: {arguments.iterations}")
    print(f"median repetition: {median * 1e3:.3f} ms")
    print(f"median solve: {median / arguments.iterations * 1e6:.3f} us")
    print(f"range: {min(timings) * 1e3:.3f}-{max(timings) * 1e3:.3f} ms")
    print(f"solve-slot high-water mark: {slot_count}")
    print(f"checksum: {checksum:.12g}")


if __name__ == "__main__":
    main()
