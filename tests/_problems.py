import jax.numpy as jnp
import numpy as np


INDICES = np.array([0, 0, 1, 0, 2, 1, 2, 3], dtype=np.int32)
INDPTR = np.array([0, 1, 3, 5, 8], dtype=np.int32)

VALUES_1 = np.array([6.0, 1.0, 5.0, 0.5, 4.0, 0.25, 1.0, 3.0])
VALUES_2 = np.array([4.0, 0.25, 3.0, 0.0, 5.0, 0.5, 0.75, 6.0])


def dense(values):
    matrix = np.zeros((4, 4), dtype=np.asarray(values).dtype)
    for row in range(4):
        for offset in range(INDPTR[row], INDPTR[row + 1]):
            column = INDICES[offset]
            matrix[row, column] = values[offset]
            matrix[column, row] = values[offset]
    return matrix


def device_problem(dtype=jnp.float64):
    return (
        jnp.asarray(INDICES),
        jnp.asarray(INDPTR),
        jnp.asarray(VALUES_1, dtype=dtype),
        jnp.asarray(VALUES_2, dtype=dtype),
    )
