# xolky (/ˈʃɔl.ki/)

Xolky solves sparse symmetric positive-definite linear systems from JAX-compiled functions.
CPU arrays use a system-installed SuiteSparse CHOLMOD; NVIDIA CUDA arrays use cuDSS.
Xolky separates the solver lifecycle into three operations:

1. setup the fixed CSR sparsity structure once;
2. refactor the numeric values whenever the matrix changes;
3. solve repeatedly with the current factorization.

## Installation

Xolky supports CPU solves through SuiteSparse CHOLMOD and GPU solves through NVIDIA cuDSS. Each backend is built only when its development files are available. Both libraries are dynamically linked and are not bundled with Xolky.

### CPU (CHOLMOD)

Install SuiteSparse/CHOLMOD development files:

```bash
sudo apt update
sudo apt install build-essential libsuitesparse-dev
```

Install Xolky with the CPU backend:

```bash
XOLKY_BUILD_CHOLMOD=on XOLKY_BUILD_CUDA=off \
pip install git+ssh://git@github.com/pedrozudo/xolky.git
```

For non-standard installations, specify the SuiteSparse installation prefix:

```bash
export CHOLMOD_ROOT=/path/to/suitesparse
```

Xolky uses `pkg-config cholmod` when available and also searches standard system locations.

The installed CHOLMOD build determines which licensed modules and algorithms are available. Xolky only ships its Apache-licensed adapter.

### GPU (cuDSS)

Install CUDA 12 or 13 using the [NVIDIA CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

Configure the [NVIDIA CUDA APT repository](https://developer.nvidia.com/cuda-downloads) for your Ubuntu version, then install cuDSS.

For CUDA 13:

```bash
sudo apt update
sudo apt install libcudss1-cuda-13 libcudss1-dev-cuda-13
```

For CUDA 12:

```bash
sudo apt update
sudo apt install libcudss1-cuda-12 libcudss1-dev-cuda-12
```

Install Xolky with the GPU backend:

```bash
XOLKY_BUILD_CUDA=on XOLKY_BUILD_CHOLMOD=off \
pip install git+ssh://git@github.com/pedrozudo/xolky.git
```

Xolky automatically detects cuDSS installed in NVIDIA's standard APT directories.

For non-standard installations, specify the CUDA and cuDSS installation prefixes:

```bash
export CUDA_HOME=/path/to/cuda
export CUDSS_ROOT=/path/to/cudss
```

Then install Xolky using the same command above.

At runtime, the CUDA and cuDSS shared libraries must be accessible to the dynamic linker. If necessary, configure `LD_LIBRARY_PATH` accordingly.

### Development

Clone the repository and install Xolky in editable mode:

```bash
git clone git@github.com:pedrozudo/xolky.git
cd xolky
pip install -e ".[test]"
```

Run the tests:

```bash
pytest
```

Backend builds can be controlled through `XOLKY_BUILD_CHOLMOD` and `XOLKY_BUILD_CUDA`, each accepting `auto` (default), `on`, or `off`. Explicitly enabled backends fail the build if their development files are missing.

## Linear functional API

The CSR input represents the lower triangle of a square SPD matrix. Indices and indptr must use int32. Numeric matrix values and right-hand sides must use float64; Xolky performs no implicit dtype conversions.

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

Backend selection follows the concrete placement of `indices` and `indptr`; values and right-hand sides must use the same backend. CHOLMOD setup requires both policies to be stated explicitly:

- `ordering="auto"` lets CHOLMOD choose between its ordering methods; explicit
  choices are `"amd"`, `"metis"`, and `"nesdis"`.
- `factorization="auto"` lets CHOLMOD choose its factor form; explicit choices are `"simplicial"` and `"supernodal"`.

Using `"auto"` records that the choice is intentional while allowing CHOLMOD to adapt it to the matrix. Available explicit methods depend on the user-installed CHOLMOD build. These options apply only to CPU arrays. For CUDA arrays, omit them and Xolky selects cuDSS.

FP32 source data must be converted explicitly. The local x64 context permits the conversion without changing JAX's process-wide configuration:

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

Setup is a host-side resource operation and must run outside jax.jit. Refactor and solve accept and return the solver PyTree:

~~~python
@jax.jit
def refactor_and_solve(solver, values, rhs):
    solver = xolky.refactor(solver, values)
    return xolky.solve(solver, rhs)
~~~

The solver identifier is a dynamic uint64 (pinned host memory for CUDA and CPU memory for CHOLMOD), so different solver instances with the same static metadata reuse one JIT compilation. A small internal sequence scalar establishes ordering between stateful FFI calls.

Solver states have linear semantics: after passing a state to refactor or solve, do not reuse the previous state. Calls using one native solver are serialized; different solver instances may run concurrently. CUDA solvers additionally own independent streams.

`jax.vmap` is explicitly rejected. Mapping the current single-system FFI call would only repeat operations against one mutable native solver; it would not construct a native batched solver. Native batching requires its own solver resource and buffer layout.

Call `close()` on the latest solver state to release its native resources. It waits for operations ordered before that state to finish and is idempotent. Remaining resources are released at interpreter shutdown.

## Native resource model

Each CHOLMOD solver owns a `cholmod_common`, the symbolic/numeric factor, owned int32 structure arrays, and reusable `cholmod_solve2` solution/workspace objects. Lower CSR is interpreted directly as upper CSC with `stype=1`, so no structural transpose is performed. Refactor temporarily wraps the JAX-owned numeric values. Solve wraps the right-hand side directly, reuses CHOLMOD-owned workspace, and copies the final `n` float64 values into the JAX output.

CHOLMOD ordering and factorization are selected by the policies passed to `setup`. The `"auto"` policies delegate those decisions to CHOLMOD; explicit ordering choices are AMD, METIS, and nested dissection, while explicit factor forms are simplicial and supernodal. Xolky always requires an LL' result so non-positive-definite matrices violate its SPD contract instead of being accepted as indefinite LDL' factorizations.

Each CUDA solver owns:

- a cuDSS handle, configuration, data object, and CSR matrix descriptor;
- a dedicated non-blocking CUDA stream and synchronization events;
- fixed device buffers for the CSR structure and numeric values;
- a dynamically grown pool of dense descriptor pairs and per-solve events.

The persistent CSR descriptor references only solver-owned buffers. Refactor copies numeric CSR values once per matrix update. Solve slots instead wrap the FFI's JAX-owned FP64 right-hand-side and output buffers directly, eliminating both device-to-device copies from the repeated-solve path. A slot is rebound only after its CUDA completion event has fired, so asynchronous solves never retarget a descriptor that is still in flight.

The pool grows to the maximum number of solve submissions observed in flight and reuses completed slots without host synchronization. Solver teardown first synchronizes the private stream, then destroys every pooled descriptor and event. Because the public API is float64-only, the repeated-solve path performs no hidden dtype conversions.

## Examples

Examples are grouped by scenario. Each directory contains a `cudss.py` version that explicitly places arrays on an NVIDIA GPU and a `cholmod.py` version that explicitly places arrays on a CPU. Native setup runs outside JIT because it creates the solver resource; refactor and solve calls are JIT-compiled.

- [`fp32_source`](examples/fp32_source)
- [`small`](examples/small)
- [`refactor`](examples/refactor)
- [`nd6k`](examples/nd6k)

## Benchmarking

Use the repeated-solve benchmark to measure the tight coarse-solver path:

~~~bash
python benchmarks/benchmark_repeated_solve.py --size 4096 --iterations 1000
~~~

The FFI handlers intentionally do not advertise XLA command-buffer compatibility. Their fixed solver-owned device buffers and private CUDA stream do not satisfy that trait's execution contract. Reconsider the trait only if the native execution model changes and CUDA graph capture is validated independently.

## Supported

- Linux CPU and NVIDIA CUDA devices
- JAX 0.11.1 or newer
- a system SuiteSparse CHOLMOD installation for CPU
- CUDA 13 and cuDSS for NVIDIA GPUs
- int32 CSR structure
- float64 numeric values and right-hand sides
- one right-hand side
- jax.jit and JAX control-flow loops

Automatic differentiation, vmap, pmap, non-SPD matrices, multiple right-hand sides, Windows, and macOS are not currently supported.
