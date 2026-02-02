# Dendrite

GPU-accelerated CUDA kernels for lithium-ion battery simulation.

**133x faster than NumPy. 197x faster than naive GPU code.**

## Why

The battery simulation community is stuck on CPU. Existing tools:

| Tool | Backend | Notes |
|------|---------|-------|
| [PyBaMM](https://github.com/pybamm-team/PyBaMM) | CPU (CasADi/IDAKLU) | Excellent physics, no GPU |
| [BattMo](https://github.com/BattMoTeam/BattMo) | CPU (MATLAB) | No GPU |
| [COMSOL](https://www.comsol.com) | CPU + cuDSS | 2-5x GPU speedup (generic sparse solver) |

We wrote hand-tuned CUDA kernels. Result:

| Method | Time (10K particles, 100s sim) | Speedup |
|--------|-------------------------------|---------|
| NumPy vectorized | 351 ms | 1x |
| CuPy naive | 518 ms | 0.7x (slower!) |
| **Dendrite** | **2.6 ms** | **133x** |

Naive GPU ports don't help. Hand-tuned kernels do.

## Build

```bash
make            # builds lib/libdendrite.{a,so}
make examples   # builds bin/simple_diffusion, bin/battery_particle
make benchmarks # builds bin/benchmark
```

Requires CUDA Toolkit 11.0+ and an NVIDIA GPU.

## Usage

```c
#include "dendrite.h"

// 2D diffusion (89% of peak bandwidth on RTX 3090)
dendrite_diffusion_2d(c_in, c_out, D, dx, dy, dt, nx, ny, stream);

// Butler-Volmer kinetics
dendrite_butler_volmer(eta, i0, j, alpha, T, n, stream);

// Spherical particle diffusion (for SPM-style models)
dendrite_spherical_diffusion(c, j_surf, D_s, R_p, dt, nr, n_particles, stream);
```

## Performance

RTX 3090 (936 GB/s theoretical peak):

| Kernel | Bandwidth | Peak % |
|--------|-----------|--------|
| 2D Diffusion (4K x 4K) | 832 GB/s | 89% |
| Butler-Volmer (4M pts) | 794 GB/s | 85% |
| Spherical Diffusion | 712 GB/s | 76% |

## Docs

See `docs/` for:
- `DEVELOPMENT.md` - Build setup, benchmarking methodology, LLM instructions
- `BENCHMARKS.md` - Full benchmark results
- `OPTIMIZATION.md` - Performance tuning findings
- `PHYSICS.md` - Battery physics background

## License

MIT
