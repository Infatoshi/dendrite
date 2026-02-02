# Optimization Findings

Key discoveries from optimizing battery simulation kernels on RTX 3090.

## Summary

| Kernel | Final Bandwidth | Peak % |
|--------|-----------------|--------|
| 2D Diffusion | 832 GB/s | 89% |
| Butler-Volmer | 794 GB/s | 85% |
| Spherical Diffusion | 712 GB/s | 76% |

## Key Insight: Lower Occupancy Wins

For memory-bound stencil codes, 8% occupancy outperforms 33%.

| Block Size | Threads | Occupancy | Bandwidth |
|------------|---------|-----------|-----------|
| 32x16 | 512 | 33% | 787 GB/s |
| 32x8 | 256 | 17% | 815 GB/s |
| **32x4** | **128** | **8%** | **836 GB/s** |

**Why:** Fewer active warps = more L2 cache per warp = less DRAM traffic for stencil neighbors.

## Warp Shuffles vs Shared Memory

For data fitting in one warp (32 elements), `__shfl_sync` beats shared memory.

Before (shared memory + 2x `__syncthreads`):
```cuda
__shared__ float s_c[32];
s_c[lane] = c_val;
__syncthreads();
float c_right = s_c[lane + 1];
```

After (warp shuffle, no sync):
```cuda
float c_right = __shfl_down_sync(0xFFFFFFFF, c_val, 1);
```

Spherical diffusion: **36% -> 85% of peak** (2.4x speedup).

## What Works

1. **`__restrict__`** - Enables compiler optimization (~5% gain)
2. **`__launch_bounds__`** - Controls register allocation
3. **Coalesced access** - Natural for stencils
4. **32x4 blocks** - Best for 2D stencils on Ampere

## What Doesn't Help (at 85%+ peak)

1. **Shared memory tiling** - Adds overhead without reducing DRAM traffic
2. **Register tiling** - Only helps compute-bound kernels
3. **FP16** - Half2 breaks at stencil boundaries
4. **Async copy** - Minimal compute to overlap with

## Roofline Position

All kernels are firmly memory-bound:

| Kernel | Arithmetic Intensity | Ridge Point |
|--------|---------------------|-------------|
| 2D Diffusion | 0.42 FLOPS/byte | 38 FLOPS/byte |
| Butler-Volmer | 1.67 FLOPS/byte | 38 FLOPS/byte |

At <5 FLOPS/byte, optimize for bandwidth, not compute.

## Profiling Commands

```bash
# Register usage
nvcc -Xptxas -v kernel.cu

# Full analysis
ncu --set full ./benchmark

# Roofline
ncu --set roofline ./benchmark
```
