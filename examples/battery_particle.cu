/*
 * Dendrite Example: Battery Particle Simulation
 *
 * Simulates lithium diffusion in spherical electrode particles
 * coupled with Butler-Volmer electrochemical kinetics.
 *
 * This is a simplified version of the Single Particle Model (SPM).
 *
 * Build: make examples
 * Run:   ./bin/battery_particle
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

/* Physical constants */
#define FARADAY 96485.33212f
#define GAS_CONST 8.314462618f

int main(int argc, char** argv)
{
    /* Particle parameters (typical NMC cathode) */
    const int n_particles = 10000;  /* Number of particles to simulate */
    const int nr = 32;              /* Radial grid points per particle */
    const float R_p = 5e-6f;        /* Particle radius [m] */
    const float D_s = 1e-14f;       /* Solid diffusivity [m^2/s] */
    const float c_max = 51000.0f;   /* Max Li concentration [mol/m^3] */

    /* Electrochemical parameters */
    const float i0_ref = 10.0f;     /* Exchange current density [A/m^2] */
    const float alpha = 0.5f;       /* Transfer coefficient */
    const float T = 298.15f;        /* Temperature [K] */

    /* Simulation parameters */
    const float dt_max = dendrite_get_max_dt_spherical(D_s, R_p, nr);
    const float dt = 0.8f * dt_max;
    const float t_end = 100.0f;     /* Simulate 100 seconds */
    const int n_steps = (int)(t_end / dt);
    const int print_every = n_steps / 10;

    /* Applied current (1C discharge) */
    const float capacity = 4.0f * 3.14159f / 3.0f * R_p * R_p * R_p * c_max * FARADAY;
    const float I_1C = capacity / 3600.0f;  /* 1C rate */
    const float j_applied = I_1C / (4.0f * 3.14159f * R_p * R_p);

    printf("Dendrite Battery Particle Example\n");
    printf("==================================\n");
    printf("Particles:    %d\n", n_particles);
    printf("Radial pts:   %d\n", nr);
    printf("R_p:          %.1f um\n", R_p * 1e6f);
    printf("D_s:          %.2e m^2/s\n", D_s);
    printf("dt:           %.2e s (max: %.2e s)\n", dt, dt_max);
    printf("Steps:        %d (t_end = %.1f s)\n", n_steps, t_end);
    printf("j_applied:    %.2f A/m^2 (1C rate)\n", j_applied);
    printf("\n");

    /* Allocate host memory */
    size_t c_size = n_particles * nr * sizeof(float);
    size_t j_size = n_particles * sizeof(float);

    float* h_c = (float*)malloc(c_size);
    float* h_j = (float*)malloc(j_size);
    float* h_eta = (float*)malloc(j_size);
    float* h_i0 = (float*)malloc(j_size);

    /* Initialize: 50% SOC (uniform concentration) */
    const float c_init = 0.5f * c_max;
    for (int p = 0; p < n_particles; p++) {
        for (int r = 0; r < nr; r++) {
            h_c[p * nr + r] = c_init;
        }
        h_i0[p] = i0_ref;
        h_eta[p] = 0.0f;  /* Will compute from OCV later */
    }

    /* Allocate device memory */
    float *d_c, *d_j, *d_eta, *d_i0;
    CHECK_CUDA(cudaMalloc(&d_c, c_size));
    CHECK_CUDA(cudaMalloc(&d_j, j_size));
    CHECK_CUDA(cudaMalloc(&d_eta, j_size));
    CHECK_CUDA(cudaMalloc(&d_i0, j_size));

    /* Copy to device */
    CHECK_CUDA(cudaMemcpy(d_c, h_c, c_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_i0, h_i0, j_size, cudaMemcpyHostToDevice));

    /* Create stream */
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    /* Timing */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, stream));

    printf("Time [s]    SOC [%%]    c_surf [mol/m^3]    eta [mV]\n");
    printf("----------------------------------------------------\n");

    /* Main simulation loop */
    for (int step = 0; step < n_steps; step++) {
        /* For simplicity, use constant overpotential that gives j_applied */
        /* In a full model, this would come from voltage-SOC relationship */
        const float F_RT = FARADAY / (GAS_CONST * T);

        /* Set overpotential to achieve desired current (inverse BV) */
        /* j = i0 * 2 * sinh(0.5 * F/RT * eta) */
        /* eta = (RT/F) * 2 * asinh(j / (2*i0)) */
        float eta_val = (2.0f / F_RT) * asinhf(j_applied / (2.0f * i0_ref));

        for (int p = 0; p < n_particles; p++) {
            h_eta[p] = eta_val;
        }
        CHECK_CUDA(cudaMemcpyAsync(d_eta, h_eta, j_size, cudaMemcpyHostToDevice, stream));

        /* Compute current from Butler-Volmer */
        CHECK_DENDRITE(dendrite_butler_volmer(
            d_eta, d_i0, d_j, alpha, T, n_particles, stream
        ));

        /* Update concentration via spherical diffusion */
        CHECK_DENDRITE(dendrite_spherical_diffusion(
            d_c, d_j, D_s, R_p, dt, nr, n_particles, stream
        ));

        /* Print progress */
        if (step % print_every == 0 || step == n_steps - 1) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaMemcpy(h_c, d_c, c_size, cudaMemcpyDeviceToHost));

            /* Average surface concentration and SOC */
            float c_surf_avg = 0.0f;
            float soc_avg = 0.0f;
            for (int p = 0; p < n_particles; p++) {
                c_surf_avg += h_c[p * nr + (nr - 1)];
                /* Average concentration in particle (volume weighted) */
                float c_avg = 0.0f;
                float vol = 0.0f;
                for (int r = 0; r < nr; r++) {
                    float r_val = (float)r / (nr - 1) * R_p;
                    float dV = r_val * r_val;
                    c_avg += h_c[p * nr + r] * dV;
                    vol += dV;
                }
                soc_avg += (c_avg / vol) / c_max;
            }
            c_surf_avg /= n_particles;
            soc_avg /= n_particles;

            printf("%8.2f    %6.2f    %14.1f    %8.2f\n",
                   step * dt, soc_avg * 100.0f, c_surf_avg, eta_val * 1000.0f);
        }
    }

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float elapsed_ms;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    printf("\nPerformance\n");
    printf("-----------\n");
    printf("Total time:     %.2f ms\n", elapsed_ms);
    printf("Time/step:      %.3f us\n", elapsed_ms * 1000.0f / n_steps);
    printf("Particles/sec:  %.2e\n", (float)n_particles * n_steps / (elapsed_ms * 1e-3f));

    /* Cleanup */
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_j));
    CHECK_CUDA(cudaFree(d_eta));
    CHECK_CUDA(cudaFree(d_i0));
    free(h_c);
    free(h_j);
    free(h_eta);
    free(h_i0);

    printf("\nDone.\n");
    return 0;
}
