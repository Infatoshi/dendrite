/*
 * Dendrite - GPU-Accelerated Battery Simulation Kernels
 *
 * High-performance CUDA kernels for lithium-ion battery physics:
 * - Diffusion (1D, 2D, spherical)
 * - Butler-Volmer electrochemical kinetics
 * - Thermal transport
 *
 * Achieves 89% of theoretical peak memory bandwidth on RTX 3090.
 *
 * https://github.com/infatoshi/dendrite
 */

#ifndef DENDRITE_H
#define DENDRITE_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version */
#define DENDRITE_VERSION_MAJOR 0
#define DENDRITE_VERSION_MINOR 1
#define DENDRITE_VERSION_PATCH 0

/* Error codes */
typedef enum {
    DENDRITE_SUCCESS = 0,
    DENDRITE_ERROR_CUDA = 1,
    DENDRITE_ERROR_INVALID_ARGUMENT = 2,
    DENDRITE_ERROR_CFL_VIOLATION = 3,
    DENDRITE_ERROR_OUT_OF_MEMORY = 4
} dendrite_error_t;

/* Get error string */
const char* dendrite_get_error_string(dendrite_error_t error);

/* ==========================================================================
 * Diffusion Solvers
 * ========================================================================== */

/*
 * 2D Diffusion Solver
 *
 * Solves: dc/dt = D * (d2c/dx2 + d2c/dy2)
 *
 * Parameters:
 *   c_in    - Input concentration field [ny x nx], device pointer
 *   c_out   - Output concentration field [ny x nx], device pointer
 *   D       - Diffusion coefficient [m^2/s]
 *   dx, dy  - Grid spacing [m]
 *   dt      - Time step [s]
 *   nx, ny  - Grid dimensions
 *   stream  - CUDA stream (0 for default)
 *
 * Performance: 836 GB/s (89% of peak) on RTX 3090
 */
dendrite_error_t dendrite_diffusion_2d(
    const float* c_in,
    float* c_out,
    float D,
    float dx,
    float dy,
    float dt,
    int nx,
    int ny,
    cudaStream_t stream
);

/*
 * Spherical Diffusion Solver (for particle-level modeling)
 *
 * Solves: dc/dt = (1/r^2) * d/dr(D * r^2 * dc/dr)
 *
 * Parameters:
 *   c         - Concentration [n_particles x nr], device pointer (in-place)
 *   j_surf    - Surface flux [n_particles], device pointer
 *   D_s       - Solid diffusivity [m^2/s]
 *   R_p       - Particle radius [m]
 *   dt        - Time step [s]
 *   nr        - Radial grid points
 *   n_particles - Number of particles
 *   stream    - CUDA stream
 *
 * Note: j_surf > 0 means discharge (Li leaving), j_surf < 0 means charge
 */
dendrite_error_t dendrite_spherical_diffusion(
    float* c,
    const float* j_surf,
    float D_s,
    float R_p,
    float dt,
    int nr,
    int n_particles,
    cudaStream_t stream
);

/* ==========================================================================
 * Electrochemistry
 * ========================================================================== */

/*
 * Butler-Volmer Kinetics
 *
 * Computes: j = i0 * (exp(alpha*F*eta/RT) - exp(-(1-alpha)*F*eta/RT))
 *
 * Parameters:
 *   eta     - Overpotential [V], device pointer
 *   i0      - Exchange current density [A/m^2], device pointer
 *   j       - Output current density [A/m^2], device pointer
 *   alpha   - Transfer coefficient (typically 0.5)
 *   T       - Temperature [K]
 *   n       - Number of points
 *   stream  - CUDA stream
 *
 * Performance: 832 GB/s (89% of peak) on RTX 3090
 */
dendrite_error_t dendrite_butler_volmer(
    const float* eta,
    const float* i0,
    float* j,
    float alpha,
    float T,
    int n,
    cudaStream_t stream
);

/* ==========================================================================
 * Thermal Solver
 * ========================================================================== */

/*
 * 2D Thermal Diffusion
 *
 * Solves: rho*Cp*dT/dt = k*(d2T/dx2 + d2T/dy2) + Q
 *
 * Parameters:
 *   T_in    - Input temperature [K], device pointer
 *   T_out   - Output temperature [K], device pointer
 *   Q       - Heat generation [W/m^3], device pointer
 *   k       - Thermal conductivity [W/(m*K)]
 *   rho     - Density [kg/m^3]
 *   Cp      - Heat capacity [J/(kg*K)]
 *   dx, dy  - Grid spacing [m]
 *   dt      - Time step [s]
 *   nx, ny  - Grid dimensions
 *   stream  - CUDA stream
 */
dendrite_error_t dendrite_thermal_2d(
    const float* T_in,
    float* T_out,
    const float* Q,
    float k,
    float rho,
    float Cp,
    float dx,
    float dy,
    float dt,
    int nx,
    int ny,
    cudaStream_t stream
);

/* ==========================================================================
 * Utilities
 * ========================================================================== */

/* Get optimal block size for current GPU */
void dendrite_get_optimal_block_size(int* block_x, int* block_y);

/* Check CFL stability condition, returns max stable dt */
float dendrite_get_max_dt_diffusion_2d(float D, float dx, float dy);
float dendrite_get_max_dt_spherical(float D_s, float R_p, int nr);

#ifdef __cplusplus
}
#endif

#endif /* DENDRITE_H */
