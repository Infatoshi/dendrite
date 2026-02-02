/*
 * Dendrite Example: 2D Diffusion
 *
 * Demonstrates solving a simple 2D diffusion problem with
 * an initial Gaussian concentration profile.
 *
 * Build: make examples
 * Run:   ./bin/simple_diffusion
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "dendrite.h"

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_DENDRITE(call) do { \
    dendrite_error_t err = call; \
    if (err != DENDRITE_SUCCESS) { \
        fprintf(stderr, "Dendrite error at %s:%d: %s\n", \
                __FILE__, __LINE__, dendrite_get_error_string(err)); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv)
{
    /* Grid parameters */
    const int nx = 2048;
    const int ny = 2048;
    const float Lx = 1.0f;  /* Domain size [m] */
    const float Ly = 1.0f;
    const float dx = Lx / (nx - 1);
    const float dy = Ly / (ny - 1);

    /* Physical parameters */
    const float D = 1e-5f;  /* Diffusion coefficient [m^2/s] */

    /* Time stepping */
    const float dt_max = dendrite_get_max_dt_diffusion_2d(D, dx, dy);
    const float dt = 0.9f * dt_max;  /* 90% of max stable dt */
    const float t_end = 10.0f;       /* 10 seconds of simulation */
    const int n_steps = (int)(t_end / dt);

    printf("Dendrite 2D Diffusion Example\n");
    printf("==============================\n");
    printf("Grid:       %d x %d\n", nx, ny);
    printf("Domain:     %.2f x %.2f m\n", Lx, Ly);
    printf("D:          %.2e m^2/s\n", D);
    printf("dt:         %.2e s (max stable: %.2e s)\n", dt, dt_max);
    printf("Steps:      %d (t_end = %.3f s)\n", n_steps, t_end);
    printf("\n");

    /* Allocate host memory */
    size_t size = nx * ny * sizeof(float);
    float* h_c = (float*)malloc(size);

    /* Initialize with Gaussian pulse at center */
    const float cx = Lx / 2.0f;
    const float cy = Ly / 2.0f;
    const float sigma = 0.05f;

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            float x = i * dx;
            float y = j * dy;
            float r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            h_c[j * nx + i] = expf(-r2 / (2.0f * sigma * sigma));
        }
    }

    /* Compute initial statistics */
    float sum_init = 0.0f;
    for (int i = 0; i < nx * ny; i++) {
        sum_init += h_c[i];
    }
    printf("Initial mass: %.6f\n", sum_init * dx * dy);

    /* Allocate device memory (double buffer) */
    float *d_c0, *d_c1;
    CHECK_CUDA(cudaMalloc(&d_c0, size));
    CHECK_CUDA(cudaMalloc(&d_c1, size));

    /* Copy initial condition to device */
    CHECK_CUDA(cudaMemcpy(d_c0, h_c, size, cudaMemcpyHostToDevice));

    /* Create CUDA stream */
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    /* Time the simulation */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, stream));

    /* Main time-stepping loop */
    float* c_in = d_c0;
    float* c_out = d_c1;

    for (int step = 0; step < n_steps; step++) {
        CHECK_DENDRITE(dendrite_diffusion_2d(
            c_in, c_out, D, dx, dy, dt, nx, ny, stream
        ));

        /* Swap buffers */
        float* tmp = c_in;
        c_in = c_out;
        c_out = tmp;
    }

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float elapsed_ms;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    /* Copy result back */
    CHECK_CUDA(cudaMemcpy(h_c, c_in, size, cudaMemcpyDeviceToHost));

    /* Compute final statistics */
    float sum_final = 0.0f;
    float c_max = 0.0f;
    for (int i = 0; i < nx * ny; i++) {
        sum_final += h_c[i];
        if (h_c[i] > c_max) c_max = h_c[i];
    }
    printf("Final mass:   %.6f (conservation error: %.2e)\n",
           sum_final * dx * dy, fabsf(sum_final - sum_init) / sum_init);
    printf("Peak conc:    %.6f (initial: 1.0)\n", c_max);
    printf("\n");

    /* Performance metrics */
    printf("Performance\n");
    printf("-----------\n");
    printf("Total time:   %.2f ms\n", elapsed_ms);
    printf("Time/step:    %.3f us\n", elapsed_ms * 1000.0f / n_steps);

    /* Bandwidth: minimum 1 read + 1 write per point (neighbors from cache) */
    float bytes_per_step = (float)nx * ny * sizeof(float) * 2.0f;
    float bandwidth = (bytes_per_step * n_steps) / (elapsed_ms * 1e-3f) / 1e9f;
    printf("Bandwidth:    %.1f GB/s (%.1f%% of 936 GB/s peak)\n",
           bandwidth, bandwidth / 936.0f * 100.0f);

    /* Cleanup */
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_c0));
    CHECK_CUDA(cudaFree(d_c1));
    free(h_c);

    printf("\nDone.\n");
    return 0;
}
