# Development Guide

Verbose documentation for contributors and LLMs.

## Project Structure

```
dendrite/
  include/
    dendrite.h              # Public C API - all function declarations
  csrc/
    diffusion.cu            # 2D Cartesian + spherical diffusion kernels
    electrochemistry.cu     # Butler-Volmer kinetics
    thermal.cu              # Heat equation with source term
  examples/
    simple_diffusion.cu     # Gaussian pulse diffusion demo
    battery_particle.cu     # SPM-style particle simulation
  benchmarks/
    benchmark.cu            # C++ kernel timing (proper per-kernel sync)
    benchmark_comparison.py # Python: NumPy vs CuPy vs Dendrite
  docs/
    DEVELOPMENT.md          # This file
    BENCHMARKS.md           # Results and methodology
    OPTIMIZATION.md         # Performance findings
    PHYSICS.md              # Battery physics background
    ROOFLINE.md             # Memory vs compute bound analysis
  Makefile
  README.md
```

## Build System

This is a pure C/CUDA project. No Python, no uv, no pip.

### Prerequisites

- CUDA Toolkit 11.0 or later
- GCC or compatible C compiler
- GNU Make
- NVIDIA GPU (tested on RTX 3090, SM 8.6)

### Build Commands

```bash
# Build static and shared libraries
make

# Output:
#   lib/libdendrite.a   - static library
#   lib/libdendrite.so  - shared library

# Build examples
make examples

# Output:
#   bin/simple_diffusion
#   bin/battery_particle

# Build benchmarks
make benchmarks

# Output:
#   bin/benchmark

# Run benchmarks
./bin/benchmark

# Clean build artifacts
make clean

# Generate PTX (for optimization analysis)
make ptx

# Generate SASS (GPU assembly)
make sass
```

### Makefile Targets

| Target | Description |
|--------|-------------|
| `all` | Build static and shared libraries |
| `examples` | Build example programs |
| `benchmarks` | Build benchmark suite |
| `bench` | Build and run benchmarks |
| `install` | Install to /usr/local (or PREFIX) |
| `clean` | Remove build artifacts |
| `ptx` | Generate PTX intermediate representation |
| `sass` | Generate GPU assembly |

### Compiler Flags

```makefile
NVCC_FLAGS := -arch=sm_86 -O3 -Xcompiler -fPIC -Iinclude
NVCC_FLAGS += --use_fast_math
NVCC_FLAGS += -Xptxas -v  # Show register usage
```

Change `-arch=sm_86` for different GPU architectures:
- RTX 3090/3080: sm_86
- RTX 4090/4080: sm_89
- A100: sm_80
- V100: sm_70

## Benchmarking Methodology

### Correct CUDA Timing

**Wrong** (measures queue time, not execution):
```c
cudaEventRecord(start);
for (int i = 0; i < iterations; i++) {
    kernel<<<grid, block>>>();  // Queued asynchronously!
}
cudaEventRecord(stop);
cudaStreamSynchronize(stream);
// BUG: Measures time to queue, not execute
```

**Correct** (per-kernel sync):
```c
for (int i = 0; i < iterations; i++) {
    cudaEventRecord(start, stream);
    kernel<<<grid, block>>>();
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);  // Wait for THIS kernel
    cudaEventElapsedTime(&ms, start, stop);
    total_ms += ms;
}
```

### Warmup

Always run 10+ warmup iterations before timing:
```c
// Warmup (JIT compilation, cache warming)
for (int i = 0; i < 10; i++) {
    kernel<<<grid, block>>>();
    cudaStreamSynchronize(stream);
}

// Now benchmark...
```

### Bandwidth Calculation

For stencil codes, use minimum memory traffic (cache serves neighbors):
```c
// 2D stencil: 1 read + 1 write per point (neighbors from L2 cache)
float bytes = nx * ny * sizeof(float) * 2.0f * iterations;
float bandwidth_gbs = bytes / (elapsed_ms * 1e-3f) / 1e9f;
float peak_percent = bandwidth_gbs / 936.0f * 100.0f;  // RTX 3090
```

## Key Optimizations

### 1. Low Occupancy Wins for Memory-Bound

Counter-intuitive: 8% occupancy beats 33% for stencil codes.

| Block Size | Occupancy | Bandwidth |
|------------|-----------|-----------|
| 32x16 | 33% | 787 GB/s |
| 32x8 | 17% | 815 GB/s |
| **32x4** | **8%** | **836 GB/s** |

Why: Fewer warps = more L2 cache per warp = less DRAM traffic.

### 2. Warp Shuffles Beat Shared Memory

For data that fits in a warp (32 elements), use `__shfl_sync`:

```cuda
// Shared memory version (slow)
__shared__ float s_data[32];
s_data[threadIdx.x] = my_value;
__syncthreads();
float neighbor = s_data[threadIdx.x + 1];

// Warp shuffle version (fast, no sync)
float neighbor = __shfl_down_sync(0xFFFFFFFF, my_value, 1);
```

Spherical diffusion: 36% -> 85% of peak after this change.

### 3. Naive GPU is Often Slower Than CPU

CuPy/PyTorch vectorized operations have massive kernel launch overhead:

```python
# This is SLOWER than NumPy for small/medium arrays!
c_new[:, 1:-1] = c[:, 1:-1] + alpha * (c[:, 2:] - 2*c[:, 1:-1] + c[:, :-2])
```

Each slice and operation is a separate kernel launch (~3us each).
Hand-tuned kernels fuse everything into one launch.

## API Reference

### Error Handling

```c
typedef enum {
    DENDRITE_SUCCESS = 0,
    DENDRITE_ERROR_CUDA = 1,
    DENDRITE_ERROR_INVALID_ARGUMENT = 2,
    DENDRITE_ERROR_CFL_VIOLATION = 3,
    DENDRITE_ERROR_OUT_OF_MEMORY = 4
} dendrite_error_t;

const char* dendrite_get_error_string(dendrite_error_t error);
```

### 2D Diffusion

```c
dendrite_error_t dendrite_diffusion_2d(
    const float* c_in,   // Input concentration [ny x nx], device ptr
    float* c_out,        // Output concentration [ny x nx], device ptr
    float D,             // Diffusion coefficient [m^2/s]
    float dx, float dy,  // Grid spacing [m]
    float dt,            // Time step [s]
    int nx, int ny,      // Grid dimensions
    cudaStream_t stream  // CUDA stream (0 for default)
);

// Get maximum stable time step (CFL condition)
float dendrite_get_max_dt_diffusion_2d(float D, float dx, float dy);
```

### Spherical Diffusion

```c
dendrite_error_t dendrite_spherical_diffusion(
    float* c,              // Concentration [n_particles x nr], in-place
    const float* j_surf,   // Surface flux [n_particles]
    float D_s,             // Solid diffusivity [m^2/s]
    float R_p,             // Particle radius [m]
    float dt,              // Time step [s]
    int nr,                // Radial points (MUST be 32)
    int n_particles,       // Number of particles
    cudaStream_t stream
);

// nr must be 32 (one warp) for warp shuffle optimization
// j_surf > 0 = discharge (Li leaving), j_surf < 0 = charge
```

### Butler-Volmer Kinetics

```c
dendrite_error_t dendrite_butler_volmer(
    const float* eta,    // Overpotential [V]
    const float* i0,     // Exchange current density [A/m^2]
    float* j,            // Output current density [A/m^2]
    float alpha,         // Transfer coefficient (0.5 for symmetric)
    float T,             // Temperature [K]
    int n,               // Number of points
    cudaStream_t stream
);

// Uses sinh optimization for alpha = 0.5
```

### Thermal Solver

```c
dendrite_error_t dendrite_thermal_2d(
    const float* T_in,   // Input temperature [K]
    float* T_out,        // Output temperature [K]
    const float* Q,      // Heat generation [W/m^3], can be NULL
    float k,             // Thermal conductivity [W/(m*K)]
    float rho,           // Density [kg/m^3]
    float Cp,            // Heat capacity [J/(kg*K)]
    float dx, float dy,  // Grid spacing [m]
    float dt,            // Time step [s]
    int nx, int ny,      // Grid dimensions
    cudaStream_t stream
);
```

## Comparison with Existing Tools

### PyBaMM

PyBaMM is the gold standard for battery physics modeling:
- Full DFN (P2D) model with electrolyte dynamics
- Thermal coupling, SEI growth, degradation
- Excellent documentation and validation

But it's CPU-only. Their IDAKLU solver is fast for single cells, but doesn't scale to batch particle simulation.

**PyBaMM is not wrong to be CPU-based** - implicit DAE solvers don't parallelize easily. Dendrite targets a different use case: explicit batch simulation of many particles.

### Why Naive GPU Fails

Tools that "add GPU support" via CuPy/PyTorch/JAX often see no speedup because:

1. **Kernel launch overhead** (~3us) dominates small operations
2. **Memory allocation** for temporaries on every timestep
3. **Python object overhead** between operations
4. **No kernel fusion** - each array operation is separate

Dendrite fuses entire stencil updates into single kernels with:
- Zero temporary allocations
- Zero Python overhead
- Zero kernel launch between operations

## Running the Comparison Benchmarks

```bash
# Install dependencies (only for comparison, not for Dendrite itself)
pip install numpy cupy-cuda12x pybamm

# Run comparison
cd benchmarks
python benchmark_comparison.py
```

Expected output:
```
NumPy vectorized (CPU):       351.06 ms
CuPy vectorized (naive GPU):  517.61 ms
Dendrite CUDA (per-step sync):  2.63 ms
Dendrite CUDA (throughput):     1.42 ms

SPEEDUPS vs NumPy CPU:
  CuPy naive:              0.7x
  Dendrite (sync):       133.4x
  Dendrite (throughput): 247.7x
```

## Contributing

1. Fork the repo
2. Create a feature branch
3. Run benchmarks before and after changes
4. Ensure no regression in bandwidth utilization
5. Submit PR with benchmark results

## License

MIT
