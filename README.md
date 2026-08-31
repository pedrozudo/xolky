# xolky (/ˈʃɔl.ki/)

Xolky solves sparse symmetric positive-definite linear systems from
JAX-compiled functions. CPU arrays use a system-installed SuiteSparse CHOLMOD;
NVIDIA CUDA arrays use cuDSS. Xolky separates the solver lifecycle into three
operations:

1. setup the fixed CSR sparsity structure once;
2. refactor the numeric values whenever the matrix changes;
3. solve repeatedly with the current factorization.

## Installation

Xolky builds each backend only when its development files are available. The
CHOLMOD library is never bundled: the resulting adapter dynamically links to
the user's system installation.

For the CPU backend, install SuiteSparse/CHOLMOD development files first. A
typical system layout is:

~~~text
/usr/include/suitesparse/cholmod.h
/usr/lib/x86_64-linux-gnu/libcholmod.so
~~~

The build uses `pkg-config cholmod` when available and also searches standard
system locations. Set `CHOLMOD_ROOT` for a non-standard prefix:

~~~bash
export CHOLMOD_ROOT=/path/to/suitesparse
export LD_LIBRARY_PATH=$CHOLMOD_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
~~~

The installed CHOLMOD build determines which licensed modules and algorithms
are available. Xolky only ships its Apache-licensed adapter.

For the CUDA backend, install CUDA 13 and cuDSS first.

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

Backend builds can be controlled explicitly with `auto` (the default), `on`,
or `off`:

~~~bash
XOLKY_BUILD_CHOLMOD=on XOLKY_BUILD_CUDA=off pip install .
XOLKY_BUILD_CHOLMOD=off XOLKY_BUILD_CUDA=on pip install ".[cuda13]"
~~~

An explicitly enabled backend fails the build when its development files are
missing. Automatic mode simply omits unavailable backends.

Once those build-time and runtime paths are configured, install Xolky:

~~~bash
pip install git+ssh://git@github.com/pedrozudo/xolky.git
~~~

For development and regression tests:

~~~bash
pip install -e ".[test]"
pytest
~~~

## Linear functional API

The CSR input represents the lower triangle of a square SPD matrix. Indices and
indptr must use int32. Numeric matrix values and right-hand sides must use
float64; Xolky performs no implicit dtype conversions.

~~~python
import jax
import jax.numpy as jnp
import xolky

jax.config.update("jax_enable_x64", True)

cpu = jax.devices("cpu")[0]
indices = jax.device_put(jnp.array([0, 0, 1, 2], dtype=jnp.int32), cpu)
indptr = jax.device_put(jnp.array([0, 1, 3, 4], dtype=jnp.int32), cpu)
values = jax.device_put(
    jnp.array([4.0, 1.0, 3.0, 2.0], dtype=jnp.float64), cpu
)
rhs = jax.device_put(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64), cpu)

solver = xolky.setup(
    indices, indptr, ordering="auto", factorization="auto"
)
solver = xolky.refactor(solver, values)
solver, solution = xolky.solve(solver, rhs)
solver.close()
~~~

Backend selection follows the concrete placement of `indices` and `indptr`;
values and right-hand sides must use the same backend. CHOLMOD setup requires
both policies to be stated explicitly:

- `ordering="auto"` lets CHOLMOD choose between its ordering methods; explicit
  choices are `"amd"`, `"metis"`, and `"nesdis"`.
- `factorization="auto"` lets CHOLMOD choose its factor form; explicit choices
  are `"simplicial"` and `"supernodal"`.

Using `"auto"` records that the choice is intentional while allowing CHOLMOD
to adapt it to the matrix. Available explicit methods depend on the
user-installed CHOLMOD build. These options apply only to CPU arrays. For CUDA arrays, omit
them and Xolky selects cuDSS.

FP32 source data must be converted explicitly. The local x64 context permits
the conversion without changing JAX's process-wide configuration:

~~~python
values_fp32 = jnp.array([4.0, 1.0, 3.0, 2.0], dtype=jnp.float32)
rhs_fp32 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

solver = xolky.setup(indices, indptr, ordering="auto", factorization="auto")

with jax.enable_x64():
    values = values_fp32.astype(jnp.float64)
    rhs = rhs_fp32.astype(jnp.float64)

    solver = xolky.refactor(solver, values)
    solver, solution = xolky.solve(solver, rhs)

solver.close()
~~~

Complete runnable versions are available for
[cuDSS](examples/fp32_source/cudss.py) and
[CHOLMOD](examples/fp32_source/cholmod.py).

Setup is a host-side resource operation and must run outside jax.jit.
Refactor and solve accept and return the solver PyTree:

~~~python
@jax.jit
def refactor_and_solve(solver, values, rhs):
    solver = xolky.refactor(solver, values)
    return xolky.solve(solver, rhs)
~~~

The solver identifier is a dynamic uint64 (pinned host memory for CUDA and CPU
memory for CHOLMOD), so different solver instances with the same static
metadata reuse one JIT compilation. A small internal sequence scalar
establishes ordering between stateful FFI calls.

Solver states have linear semantics: after passing a state to refactor or solve,
do not reuse the previous state. Calls using one native solver are serialized;
different solver instances may run concurrently. CUDA solvers additionally own
independent streams.

`jax.vmap` is explicitly rejected. Mapping the current single-system FFI call
would only repeat operations against one mutable native solver; it would not
construct a native batched solver. Native batching requires its own solver
resource and buffer layout.

Call `close()` on the latest solver state to release its native resources. It
waits for operations ordered before that state to finish and is idempotent.
Remaining resources are released at interpreter shutdown.

## Native resource model

Each CHOLMOD solver owns a `cholmod_common`, the symbolic/numeric factor, owned
int32 structure arrays, and reusable `cholmod_solve2` solution/workspace
objects. Lower CSR is interpreted directly as upper CSC with `stype=1`, so no
structural transpose is performed. Refactor temporarily wraps the JAX-owned
numeric values. Solve wraps the right-hand side directly, reuses CHOLMOD-owned
workspace, and copies the final `n` float64 values into the JAX output.

CHOLMOD ordering and factorization are selected by the policies passed to
`setup`. The `"auto"` policies delegate those decisions to CHOLMOD; explicit
ordering choices are AMD, METIS, and nested dissection, while explicit factor
forms are simplicial and supernodal. Xolky always requires an LL' result so
non-positive-definite matrices violate its SPD contract instead of being
accepted as indefinite LDL' factorizations.

Each CUDA solver owns:

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
event. Because the public API is float64-only, the repeated-solve path performs
no hidden dtype conversions.

## Examples

Examples are grouped by scenario. Each directory contains a `cudss.py` version
that explicitly places arrays on an NVIDIA GPU and a `cholmod.py` version that
explicitly places arrays on a CPU. Native setup runs outside JIT because it
creates the solver resource; refactor and solve calls are JIT-compiled.

- [`fp32_source`](examples/fp32_source)
- [`small`](examples/small)
- [`refactor`](examples/refactor)
- [`nd6k`](examples/nd6k)

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

- Linux CPU and NVIDIA CUDA devices
- JAX 0.11.1 or newer
- a system SuiteSparse CHOLMOD installation for CPU
- CUDA 13 and cuDSS for NVIDIA GPUs
- int32 CSR structure
- float64 numeric values and right-hand sides
- one right-hand side
- jax.jit and JAX control-flow loops

Automatic differentiation, vmap, pmap, non-SPD matrices, multiple right-hand
sides, Windows, and macOS are not currently supported.
