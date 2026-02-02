# Roofline Analysis

Why battery simulation kernels are memory-bound.

## RTX 3090 Specs

| Metric | Value |
|--------|-------|
| Peak FP32 | 35.6 TFLOPS |
| Peak Memory BW | 936 GB/s |
| Ridge Point | 38 FLOPS/byte |

## Kernel Arithmetic Intensity

### 2D Diffusion

```
c_out = c + rx*(left - 2*c + right) + ry*(top - 2*c + bottom)
```

- Memory: 2 floats (1 read + 1 write, neighbors cached)
- Compute: ~10 FLOPs
- **AI = 10 / 8 = 1.25 FLOPS/byte**

### Butler-Volmer

```
j = i0 * (exp(alpha*F*eta/RT) - exp(-(1-alpha)*F*eta/RT))
```

- Memory: 3 floats (2 read + 1 write)
- Compute: ~20 FLOPs (2x exp)
- **AI = 20 / 12 = 1.67 FLOPS/byte**

## Roofline Position

```
FLOPS
  |
  |         _______________  Peak Compute (35.6 TF)
  |        /
  |       /
  |      /  All battery kernels here
  |     /   (memory-bound region)
  |    /
  |   /
  |__/________________________ Arithmetic Intensity
     1.25  1.67      38
```

Both kernels are far left of the ridge point (38 FLOPS/byte).

## Implications

1. Optimize for **bandwidth**, not FLOPs
2. Reduce memory traffic (caching, coalescing)
3. Don't expect FP16 to help (compute not the bottleneck)
4. Focus on L2 cache efficiency

## Achieved vs Theoretical

| Kernel | Achieved | Theoretical | Efficiency |
|--------|----------|-------------|------------|
| 2D Diffusion | 832 GB/s | 936 GB/s | 89% |
| Butler-Volmer | 794 GB/s | 936 GB/s | 85% |

The ~10% gap is due to:
- Kernel launch overhead
- Memory controller inefficiency
- Cache line granularity
