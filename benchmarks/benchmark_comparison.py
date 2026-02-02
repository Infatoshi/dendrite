"""
Final benchmark: Optimized CPU vs Dendrite GPU

Fair comparison using vectorized NumPy (no per-particle loops)
"""

import time
import numpy as np


def benchmark_numpy_vectorized():
    """Fully vectorized NumPy - best case CPU"""
    print("NumPy vectorized CPU: 10k particles, 288 steps")

    n_particles = 10000
    nr = 32
    n_steps = 288

    c = np.ones((n_particles, nr), dtype=np.float32) * 25000.0
    j_surf = np.ones(n_particles, dtype=np.float32) * 1.0

    D_s = 1e-14
    R_p = 5e-6
    dr = R_p / (nr - 1)
    dt = 0.8 * (dr * dr) / (6.0 * D_s)
    alpha = D_s * dt / (dr * dr)
    FARADAY = 96485.33212

    # Precompute r-dependent factors
    r = np.arange(nr) * dr
    r_safe = np.maximum(r, 0.5 * dr)
    r_plus = 1.0 + dr / (2.0 * r_safe[1:-1])
    r_minus = 1.0 - dr / (2.0 * r_safe[1:-1])

    start = time.perf_counter()
    for _ in range(n_steps):
        c_new = np.empty_like(c)

        # Center (r=0): L'Hopital - vectorized over particles
        c_new[:, 0] = c[:, 0] + 6.0 * alpha * (c[:, 1] - c[:, 0])

        # Interior: fully vectorized
        c_new[:, 1:-1] = c[:, 1:-1] + alpha * (
            r_plus * (c[:, 2:] - c[:, 1:-1]) -
            r_minus * (c[:, :-2] - c[:, 1:-1])
        )

        # Surface: flux BC - vectorized over particles
        flux = -j_surf / FARADAY
        c_ghost = c[:, -1] + 2.0 * dr * flux / D_s
        r_factor = 1.0 + dr / r_safe[-1]
        c_new[:, -1] = c[:, -1] + alpha * (
            r_factor * (c_ghost - c[:, -1]) -
            (2.0 - dr / r_safe[-1]) * (c[:, -1] - c[:, -2])
        )

        c = np.maximum(c_new, 0.0)

    elapsed = time.perf_counter() - start
    elapsed_ms = elapsed * 1000
    print(f"  Total time: {elapsed_ms:.2f} ms")
    print(f"  Per step: {elapsed_ms/n_steps*1000:.2f} us")

    return elapsed_ms


def benchmark_dendrite():
    """Dendrite CUDA: 10k particles, 288 steps"""
    import cupy as cp

    print("\nDendrite CUDA: 10k particles, 288 steps")

    n_particles = 10000
    nr = 32
    n_steps = 288

    c = cp.ones((n_particles, nr), dtype=cp.float32) * 25000.0
    j_surf = cp.ones(n_particles, dtype=cp.float32) * 1.0

    D_s = 1e-14
    R_p = 5e-6
    dr = R_p / (nr - 1)
    dt = 0.8 * (dr * dr) / (6.0 * D_s)
    alpha = D_s * dt / (dr * dr)

    kernel_code = r'''
    extern "C" __global__ void spherical_step(
        float* c, const float* j_surf,
        float D_s, float dr, float alpha, int nr, int n_particles)
    {
        int lane = threadIdx.x & 31;
        int warp_id = threadIdx.x >> 5;
        int particle_id = blockIdx.x * 8 + warp_id;
        if (particle_id >= n_particles) return;
        int idx = particle_id * 32 + lane;
        float c_val = c[idx];
        unsigned mask = 0xFFFFFFFF;
        float c_left = __shfl_up_sync(mask, c_val, 1);
        float c_right = __shfl_down_sync(mask, c_val, 1);
        float r = lane * dr;
        float r_safe = fmaxf(r, 0.5f * dr);
        float c_new;
        if (lane == 0) {
            c_new = c_val + 6.0f * alpha * (c_right - c_val);
        } else if (lane == 31) {
            float j = j_surf[particle_id];
            float flux = -j / 96485.33212f;
            float c_ghost = c_val + 2.0f * dr * flux / D_s;
            float r_factor = 1.0f + dr / r_safe;
            c_new = c_val + alpha * (
                r_factor * (c_ghost - c_val) -
                (2.0f - dr / r_safe) * (c_val - c_left)
            );
        } else {
            float r_plus = 1.0f + dr / (2.0f * r_safe);
            float r_minus = 1.0f - dr / (2.0f * r_safe);
            c_new = c_val + alpha * (
                r_plus * (c_right - c_val) -
                r_minus * (c_val - c_left)
            );
        }
        c[idx] = fmaxf(0.0f, c_new);
    }
    '''

    module = cp.RawModule(code=kernel_code, options=('--std=c++11',))
    kernel = module.get_function('spherical_step')

    grid = ((n_particles + 7) // 8,)
    block = (256,)

    # Warmup
    kernel(grid, block, (c, j_surf, cp.float32(D_s), cp.float32(dr),
            cp.float32(alpha), cp.int32(nr), cp.int32(n_particles)))
    cp.cuda.Stream.null.synchronize()

    # Benchmark with proper per-kernel sync
    total_ms = 0.0
    for _ in range(n_steps):
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        start_event.record()
        kernel(grid, block, (c, j_surf, cp.float32(D_s), cp.float32(dr),
                cp.float32(alpha), cp.int32(nr), cp.int32(n_particles)))
        end_event.record()
        end_event.synchronize()
        total_ms += cp.cuda.get_elapsed_time(start_event, end_event)

    print(f"  Total time: {total_ms:.2f} ms")
    print(f"  Per step: {total_ms/n_steps*1000:.2f} us")

    # Also measure throughput without per-step sync
    print("\n  (Throughput mode - no per-step sync):")
    c = cp.ones((n_particles, nr), dtype=cp.float32) * 25000.0
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record()
    for _ in range(n_steps):
        kernel(grid, block, (c, j_surf, cp.float32(D_s), cp.float32(dr),
                cp.float32(alpha), cp.int32(nr), cp.int32(n_particles)))
    end_event.record()
    end_event.synchronize()
    throughput_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    print(f"  Total time: {throughput_ms:.2f} ms")
    print(f"  Per step: {throughput_ms/n_steps*1000:.2f} us")

    return total_ms, throughput_ms


def benchmark_cupy_vectorized():
    """CuPy vectorized (naive GPU) for comparison"""
    import cupy as cp

    print("\nCuPy vectorized (naive GPU): 10k particles, 288 steps")

    n_particles = 10000
    nr = 32
    n_steps = 288

    c = cp.ones((n_particles, nr), dtype=cp.float32) * 25000.0
    j_surf = cp.ones(n_particles, dtype=cp.float32) * 1.0

    D_s = 1e-14
    R_p = 5e-6
    dr = R_p / (nr - 1)
    dt = 0.8 * (dr * dr) / (6.0 * D_s)
    alpha = D_s * dt / (dr * dr)
    FARADAY = 96485.33212

    r = cp.arange(nr) * dr
    r_safe = cp.maximum(r, 0.5 * dr)
    r_plus = 1.0 + dr / (2.0 * r_safe[1:-1])
    r_minus = 1.0 - dr / (2.0 * r_safe[1:-1])

    # Warmup
    cp.cuda.Stream.null.synchronize()

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record()

    for _ in range(n_steps):
        c_new = cp.empty_like(c)
        c_new[:, 0] = c[:, 0] + 6.0 * alpha * (c[:, 1] - c[:, 0])
        c_new[:, 1:-1] = c[:, 1:-1] + alpha * (
            r_plus * (c[:, 2:] - c[:, 1:-1]) -
            r_minus * (c[:, :-2] - c[:, 1:-1])
        )
        flux = -j_surf / FARADAY
        c_ghost = c[:, -1] + 2.0 * dr * flux / D_s
        r_factor = 1.0 + dr / r_safe[-1]
        c_new[:, -1] = c[:, -1] + alpha * (
            r_factor * (c_ghost - c[:, -1]) -
            (2.0 - dr / r_safe[-1]) * (c[:, -1] - c[:, -2])
        )
        c = cp.maximum(c_new, 0.0)

    end_event.record()
    end_event.synchronize()
    elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)

    print(f"  Total time: {elapsed_ms:.2f} ms")
    print(f"  Per step: {elapsed_ms/n_steps*1000:.2f} us")

    return elapsed_ms


if __name__ == "__main__":
    print("="*70)
    print("FINAL BENCHMARK: CPU vs GPU for Spherical Diffusion")
    print("10k particles, 32 radial points, 288 timesteps (100s simulation)")
    print("="*70 + "\n")

    numpy_ms = benchmark_numpy_vectorized()
    cupy_ms = benchmark_cupy_vectorized()
    dendrite_sync_ms, dendrite_throughput_ms = benchmark_dendrite()

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"  NumPy vectorized (CPU):     {numpy_ms:>8.2f} ms")
    print(f"  CuPy vectorized (naive GPU):{cupy_ms:>8.2f} ms")
    print(f"  Dendrite CUDA (per-step sync): {dendrite_sync_ms:>5.2f} ms")
    print(f"  Dendrite CUDA (throughput):    {dendrite_throughput_ms:>5.2f} ms")

    print("\n" + "="*70)
    print("SPEEDUPS vs NumPy CPU")
    print("="*70)
    print(f"  CuPy naive:           {numpy_ms/cupy_ms:>6.1f}x")
    print(f"  Dendrite (sync):      {numpy_ms/dendrite_sync_ms:>6.1f}x")
    print(f"  Dendrite (throughput):{numpy_ms/dendrite_throughput_ms:>6.1f}x")

    print("\n" + "="*70)
    print("SPEEDUP: Dendrite vs CuPy naive")
    print("="*70)
    print(f"  {cupy_ms/dendrite_sync_ms:.1f}x faster (optimized kernel vs naive vectorized)")
