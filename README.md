# xolky (/ˈʃɔl.ki/)

Xolky solves sparse symmetric positive-definite linear systems with NVIDIA
cuDSS from JAX-compiled CUDA functions. It separates the solver lifecycle into
three operations:

1. setup the fixed CSR sparsity structure once;
2. refactor the numeric values whenever the matrix changes;
3. solve repeatedly with the current factorization.

## Installation

Install CUDA 13 and cuDSS first.

By default, the build searches `CUDA_HOME`, `/usr/local/cuda`, and standard
system development directories. A common cuDSS installation layout is:

~~~text
/usr/local/cuda/include/cudss.h
/usr/local/cuda/lib64/libcudss.so
~~~

At runtime, `libcudss.so` and its dependencies must also be visible to the
dynamic loader. If necessary, register its library directory with the system
linker configuration, or set, for example:

~~~bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
~~~

Verify that the runtime loader can find cuDSS before installing:

~~~bash
python -c 'import ctypes; ctypes.CDLL("libcudss.so"); print("cuDSS is visible")'
~~~

If cuDSS is installed elsewhere, set `CUDSS_ROOT` to its installation prefix.
Set `CUDA_HOME` similarly when CUDA itself is not under `/usr/local/cuda`:

~~~bash
export CUDA_HOME=/path/to/cuda
export CUDSS_ROOT=/path/to/cudss
export LD_LIBRARY_PATH=$CUDSS_ROOT/lib64:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
~~~

The build checks for `cudss.h` and the linkable `libcudss.so` and reports their
searched paths if either is missing.

Once those build-time and runtime paths are configured, install Xolky:

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
resources. `close()` is idempotent. Remaining resources are released at
interpreter shutdown.

## Native resource model

Each solver owns:

- a cuDSS handle, configuration, data object, and CSR matrix descriptor;
- a dedicated non-blocking CUDA stream and synchronization events;
- fixed device buffers for the CSR structure and numeric values;
- a dynamically grown pool of dense descriptor pairs and per-solve events.

The persistent CSR descriptor references only solver-owned buffers. Refactor
copies numeric CSR values once per matrix update. Solve slots instead wrap the
FFI's JAX-owned FP64 right-hand-side and output buffers directly, eliminating
both device-to-device copies from the repeated-solve path. A slot is rebound
only after its CUDA completion event has fired, so asynchronous solves never
retarget a descriptor that is still in flight.

The pool grows to the maximum number of solve submissions observed in flight
and reuses completed slots without host synchronization. Solver teardown first
synchronizes the private stream, then destroys every pooled descriptor and
event. Float32 inputs still require the documented JAX float32-to-float64 and
float64-to-float32 conversions around the native solve.

## Benchmarking

Use the repeated-solve benchmark to measure the tight coarse-solver path:

~~~bash
python benchmarks/benchmark_repeated_solve.py --size 4096 --iterations 1000
~~~

The FFI handlers intentionally do not advertise XLA command-buffer
compatibility. Their fixed solver-owned device buffers and private CUDA stream
do not satisfy that trait's execution contract. Reconsider the trait only if
the native execution model changes and CUDA graph capture is validated
independently.

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
