# Benchmark Results

All measurements on RTX 3090 (936 GB/s theoretical peak).
See `DEVELOPMENT.md` for methodology.

## Kernel Performance

### 2D Diffusion

| Grid Size | Bandwidth | Peak % | Time/step |
|-----------|-----------|--------|-----------|
| 256x256 | 239 GB/s | 25.5% | 2.19 us |
| 512x512 | 449 GB/s | 48.0% | 4.67 us |
| 1024x1024 | 617 GB/s | 66.0% | 13.59 us |
| 2048x2048 | 775 GB/s | 82.8% | 43.28 us |
| 4096x4096 | 832 GB/s | 88.9% | 161.25 us |

### Butler-Volmer Kinetics

| Points | Bandwidth | Peak % | Time/step |
|--------|-----------|--------|-----------|
| 64K | 252 GB/s | 26.9% | 3.13 us |
| 256K | 807 GB/s | 86.2% | 3.90 us |
| 1M | 729 GB/s | 77.8% | 17.27 us |
| 4M | 794 GB/s | 84.8% | 63.42 us |

### Spherical Diffusion

| Particles | Bandwidth | Peak % | Time/step |
|-----------|-----------|--------|-----------|
| 1K | 82 GB/s | 8.8% | 3.16 us |
| 10K | 570 GB/s | 60.9% | 4.56 us |
| 100K | 712 GB/s | 76.1% | 36.50 us |

Small sizes dominated by ~3us kernel launch overhead.

## vs Existing Tools

**Task:** 10K particles, 32 radial points, 288 timesteps (100s simulation)

| Method | Time | vs Dendrite |
|--------|------|-------------|
| PyBaMM IDAKLU (projected) | 35,744 ms | 13,490x slower |
| NumPy vectorized | 351 ms | 133x slower |
| CuPy vectorized | 518 ms | 197x slower |
| **Dendrite (sync)** | **2.63 ms** | baseline |
| **Dendrite (throughput)** | **1.42 ms** | 1.9x faster |

Key insight: Naive GPU (CuPy) is **slower than CPU** due to kernel launch overhead.

## Reproduce

```bash
# Kernel benchmarks
make benchmarks
./bin/benchmark

# Comparison benchmarks (requires Python + NumPy + CuPy)
cd benchmarks
python benchmark_comparison.py
```
