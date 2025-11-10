# Xolky
Solve linear systems using sparse Cholesky decomposition with cuDSS in Jax within jitted functions.


## Installation

1. Install the CUDA 12 Toolkit ([https://developer.nvidia.com/cuda-12-0-0-download-archive](link))
2. Install CuDSS ([https://developer.nvidia.com/cudss-downloads](link)).

```bash
pip install git+ssh://git@github.com/pedrozudo/xolky.git
```

# Examples

Go to the examples directory 🙂

# What is Supported?

### Operating System
- ✅ Linux
- ❌ Windows
- ❌ macOS

### Which Higher Order Functions?
- ✅ jit
- ❌ grad
- ❌ vmap
- ❌ pmap

### What Part of the cuDSS API?
- ✅ solving sparse positive definite linear systems
- ❌ all the rest

### Precision?
- ✅ fp32 (on the Jax side uses fp64 on the cuDSS side)
- ❌ fp64



