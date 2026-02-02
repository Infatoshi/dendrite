/*
 * Dendrite Benchmark Suite
 *
 * Measures bandwidth and compares against theoretical peak.
 * RTX 3090: 936 GB/s theoretical peak memory bandwidth.
 *
 * Build: make benchmarks
 * Run:   ./bin/benchmark
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "dendrite.h"

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_DENDRITE(call) do { \
    dendrite_error_t err = call; \
    if (err != DENDRITE_SUCCESS) { \
        fprintf(stderr, "Dendrite error: %s\n", dendrite_get_error_string(err)); \
        exit(1); \
    } \
} while(0)

/* GPU specs (RTX 3090) */
#define PEAK_BANDWIDTH_GBS 936.0f

typedef struct {
    const char* name;
    float bandwidth_gbs;
    float peak_percent;
    float time_us;
} BenchmarkResult;

/*
 * Benchmark 2D Diffusion
 */
BenchmarkResult benchmark_diffusion_2d(int nx, int ny, int iterations, cudaStream_t stream)
{
    size_t size = nx * ny * sizeof(float);
    float *d_c0, *d_c1;

    CHECK_CUDA(cudaMalloc(&d_c0, size));
    CHECK_CUDA(cudaMalloc(&d_c1, size));
    CHECK_CUDA(cudaMemset(d_c0, 0, size));

    /* Parameters that satisfy CFL */
    float D = 1e-5f;
    float dx = 1.0f / nx;
    float dy = 1.0f / ny;
    float dt = 0.4f * dendrite_get_max_dt_diffusion_2d(D, dx, dy);

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        dendrite_diffusion_2d(d_c0, d_c1, D, dx, dy, dt, nx, ny, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    /* Benchmark - sync after each kernel for accurate timing */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float total_ms = 0.0f;
    float* c_in = d_c0;
    float* c_out = d_c1;

    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        dendrite_diffusion_2d(c_in, c_out, D, dx, dy, dt, nx, ny, stream);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float iter_ms;
        CHECK_CUDA(cudaEventElapsedTime(&iter_ms, start, stop));
        total_ms += iter_ms;

        float* tmp = c_in; c_in = c_out; c_out = tmp;
    }

    float elapsed_ms = total_ms;

    /* Minimum memory traffic: 1 read + 1 write per point (neighbors served from cache) */
    float bytes = (float)nx * ny * sizeof(float) * 2.0f * iterations;
    float bandwidth = bytes / (elapsed_ms * 1e-3f) / 1e9f;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_c0));
    CHECK_CUDA(cudaFree(d_c1));

    BenchmarkResult result;
    result.name = "Diffusion 2D";
    result.bandwidth_gbs = bandwidth;
    result.peak_percent = bandwidth / PEAK_BANDWIDTH_GBS * 100.0f;
    result.time_us = elapsed_ms * 1000.0f / iterations;
    return result;
}

/*
 * Benchmark Butler-Volmer
 */
BenchmarkResult benchmark_butler_volmer(int n, int iterations, cudaStream_t stream)
{
    size_t size = n * sizeof(float);
    float *d_eta, *d_i0, *d_j;

    CHECK_CUDA(cudaMalloc(&d_eta, size));
    CHECK_CUDA(cudaMalloc(&d_i0, size));
    CHECK_CUDA(cudaMalloc(&d_j, size));

    /* Initialize with small values to avoid overflow */
    float* h_data = (float*)malloc(size);
    for (int i = 0; i < n; i++) {
        h_data[i] = 0.01f * ((float)i / n - 0.5f);  /* -0.005 to 0.005 V */
    }
    CHECK_CUDA(cudaMemcpy(d_eta, h_data, size, cudaMemcpyHostToDevice));

    for (int i = 0; i < n; i++) {
        h_data[i] = 10.0f;  /* 10 A/m^2 */
    }
    CHECK_CUDA(cudaMemcpy(d_i0, h_data, size, cudaMemcpyHostToDevice));
    free(h_data);

    float alpha = 0.5f;
    float T = 298.15f;

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        dendrite_butler_volmer(d_eta, d_i0, d_j, alpha, T, n, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    /* Benchmark - sync after each kernel */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float total_ms = 0.0f;
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        dendrite_butler_volmer(d_eta, d_i0, d_j, alpha, T, n, stream);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float iter_ms;
        CHECK_CUDA(cudaEventElapsedTime(&iter_ms, start, stop));
        total_ms += iter_ms;
    }

    float elapsed_ms = total_ms;

    /* 2 reads (eta, i0) + 1 write (j) per point */
    float bytes = (float)n * sizeof(float) * 3.0f * iterations;
    float bandwidth = bytes / (elapsed_ms * 1e-3f) / 1e9f;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_eta));
    CHECK_CUDA(cudaFree(d_i0));
    CHECK_CUDA(cudaFree(d_j));

    BenchmarkResult result;
    result.name = "Butler-Volmer";
    result.bandwidth_gbs = bandwidth;
    result.peak_percent = bandwidth / PEAK_BANDWIDTH_GBS * 100.0f;
    result.time_us = elapsed_ms * 1000.0f / iterations;
    return result;
}

/*
 * Benchmark Spherical Diffusion
 */
BenchmarkResult benchmark_spherical_diffusion(int nr, int n_particles, int iterations, cudaStream_t stream)
{
    size_t c_size = n_particles * nr * sizeof(float);
    size_t j_size = n_particles * sizeof(float);
    float *d_c, *d_j;

    CHECK_CUDA(cudaMalloc(&d_c, c_size));
    CHECK_CUDA(cudaMalloc(&d_j, j_size));

    /* Initialize */
    float* h_c = (float*)malloc(c_size);
    float* h_j = (float*)malloc(j_size);
    for (int p = 0; p < n_particles; p++) {
        for (int r = 0; r < nr; r++) {
            h_c[p * nr + r] = 25000.0f;  /* 50% of c_max */
        }
        h_j[p] = 1.0f;  /* Small current */
    }
    CHECK_CUDA(cudaMemcpy(d_c, h_c, c_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_j, h_j, j_size, cudaMemcpyHostToDevice));
    free(h_c);
    free(h_j);

    float D_s = 1e-14f;
    float R_p = 5e-6f;
    float dt = 0.8f * dendrite_get_max_dt_spherical(D_s, R_p, nr);

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        dendrite_spherical_diffusion(d_c, d_j, D_s, R_p, dt, nr, n_particles, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    /* Benchmark - sync after each kernel */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float total_ms = 0.0f;
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        dendrite_spherical_diffusion(d_c, d_j, D_s, R_p, dt, nr, n_particles, stream);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float iter_ms;
        CHECK_CUDA(cudaEventElapsedTime(&iter_ms, start, stop));
        total_ms += iter_ms;
    }

    float elapsed_ms = total_ms;

    /* Each particle: nr reads + 1 j_surf read + nr writes */
    float bytes = (float)(n_particles * (nr * 2 + 1)) * sizeof(float) * iterations;
    float bandwidth = bytes / (elapsed_ms * 1e-3f) / 1e9f;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_j));

    BenchmarkResult result;
    result.name = "Spherical Diff";
    result.bandwidth_gbs = bandwidth;
    result.peak_percent = bandwidth / PEAK_BANDWIDTH_GBS * 100.0f;
    result.time_us = elapsed_ms * 1000.0f / iterations;
    return result;
}

void print_result(BenchmarkResult r)
{
    printf("%-16s  %8.1f GB/s  %6.1f%%   %8.2f us/iter\n",
           r.name, r.bandwidth_gbs, r.peak_percent, r.time_us);
}

int main(int argc, char** argv)
{
    /* Print GPU info */
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    printf("Dendrite Benchmark Suite\n");
    printf("========================\n\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Memory: %.1f GB\n", prop.totalGlobalMem / 1e9f);
    printf("Memory BW (theoretical): %.0f GB/s\n", PEAK_BANDWIDTH_GBS);
    printf("\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    const int iterations = 1000;

    printf("Kernel             Bandwidth    Peak%%   Time\n");
    printf("----------------------------------------------\n");

    /* 2D Diffusion at various sizes */
    print_result(benchmark_diffusion_2d(256, 256, iterations, stream));
    print_result(benchmark_diffusion_2d(512, 512, iterations, stream));
    print_result(benchmark_diffusion_2d(1024, 1024, iterations, stream));
    print_result(benchmark_diffusion_2d(2048, 2048, iterations, stream));
    print_result(benchmark_diffusion_2d(4096, 4096, iterations, stream));

    printf("\n");

    /* Butler-Volmer at various sizes */
    print_result(benchmark_butler_volmer(1 << 16, iterations, stream));
    print_result(benchmark_butler_volmer(1 << 18, iterations, stream));
    print_result(benchmark_butler_volmer(1 << 20, iterations, stream));
    print_result(benchmark_butler_volmer(1 << 22, iterations, stream));

    printf("\n");

    /* Spherical diffusion */
    print_result(benchmark_spherical_diffusion(32, 1000, iterations, stream));
    print_result(benchmark_spherical_diffusion(32, 10000, iterations, stream));
    print_result(benchmark_spherical_diffusion(32, 100000, iterations, stream));

    CHECK_CUDA(cudaStreamDestroy(stream));

    printf("\n");
    printf("Notes:\n");
    printf("- Peak%% = measured / theoretical (936 GB/s for RTX 3090)\n");
    printf("- 89%% of peak is near-optimal for memory-bound kernels\n");
    printf("- Lower efficiency at small sizes due to kernel launch overhead\n");

    return 0;
}
