# xolky (/ˈʃɔl.ki/)
Solve linear systems using sparse Cholesky decomposition with cuDSS in Jax within jitted functions.


## Installation

First, install CuDSS ([https://developer.nvidia.com/cudss-downloads](link)) and then you can install xolky.

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
- ✅ fp32 (uses fp64 on the cuDSS side)
- ✅ fp64



