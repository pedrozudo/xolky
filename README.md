# xolky (/ˈʃɔl.ki/)

Xolky solves sparse symmetric positive-definite linear systems with NVIDIA
cuDSS from JAX-compiled CUDA functions. It separates the solver lifecycle into
three operations:

1. setup the fixed CSR sparsity structure once;
2. refactor the numeric values whenever the matrix changes;
3. solve repeatedly with the current factorization.

## Installation

Install CUDA 13 and cuDSS first, then install Xolky:

~~~bash
pip install git+ssh://git@github.com/pedrozudo/xolky.git
~~~

For development and regression tests:

~~~bash
pip install -e ".[test]"
pytest
~~~

## Functional API

The CSR input represents the lower triangle of a square SPD matrix. Indices and
indptr must use int32.

~~~python
import jax
import jax.numpy as jnp
import xolky

jax.config.update("jax_enable_x64", True)

indices = jnp.array([0, 0, 1, 2], dtype=jnp.int32)
indptr = jnp.array([0, 1, 3, 4], dtype=jnp.int32)
values = jnp.array([4.0, 1.0, 3.0, 2.0])
rhs = jnp.array([1.0, 2.0, 3.0])

solver = xolky.setup(indices, indptr)
solver = xolky.refactor(solver, values)
solver, solution = xolky.solve(solver, rhs)
solver.close()
~~~

Setup is a host-side resource operation and must run outside jax.jit.
Refactor and solve accept and return the solver PyTree:

~~~python
@jax.jit
def refactor_and_solve(solver, values, rhs):
    solver = xolky.refactor(solver, values)
    return xolky.solve(solver, rhs)
~~~

The solver identifier is a dynamic pinned-host uint64, so different solver
instances with the same static dimensions reuse the same JIT compilation. A
small internal sequence scalar establishes ordering between stateful FFI calls.

Solver states have linear semantics: after passing a state to refactor or solve,
do not reuse the previous state. Calls using one native solver are serialized;
different solver instances own independent CUDA streams and may run
concurrently.

`jax.vmap` is explicitly rejected. Mapping the current single-system FFI call
would only repeat operations against one mutable native solver; it would not
construct a cuDSS batched solver. Native batching requires its own solver
resource and buffer layout and will be implemented separately.

Use close, or the solver as a context manager, to release its CUDA and cuDSS
resources. Remaining resources are released at interpreter shutdown.

## Native resource model

Each solver owns:

- a cuDSS handle, configuration, data object, and matrix descriptors;
- a dedicated non-blocking CUDA stream and synchronization events;
- fixed device buffers for CSR structure, values, right-hand side, and result.

Persistent cuDSS descriptors only reference solver-owned buffers. Runtime calls
copy values or vectors into those buffers, execute on the dedicated stream, and
copy solutions back to JAX outputs. The descriptors and handle stream are never
retargeted after setup.

## Supported

- Linux and NVIDIA CUDA GPUs
- JAX 0.11.1 or newer
- CUDA 13 and cuDSS
- int32 CSR structure
- float32 and float64 inputs; cuDSS computation uses float64 internally
- one right-hand side
- jax.jit and JAX control-flow loops

Automatic differentiation, vmap, pmap, non-SPD matrices, multiple right-hand
sides, Windows, and macOS are not currently supported.
