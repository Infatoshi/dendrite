/*
 * Dendrite - Thermal Kernels
 *
 * 2D thermal diffusion with heat generation term.
 * Optimized for RTX 3090 (Ampere) with 32x4 block size.
 */

#include "dendrite.h"

/*
 * 2D Thermal Diffusion Kernel
 *
 * Solves: rho*Cp*dT/dt = k*(d2T/dx2 + d2T/dy2) + Q
 *
 * Same optimization strategy as diffusion_2d_kernel:
 * - 32x4 blocks (8% occupancy)
 * - __restrict__ for compiler optimization
 */
__global__ __launch_bounds__(128, 16)
void thermal_2d_kernel(
    const float* __restrict__ T_in,
    float* __restrict__ T_out,
    const float* __restrict__ Q,
    const float rx,
    const float ry,
    const float q_factor,
    const int nx,
    const int ny)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int gid = gy * nx + gx;

    if (gx > 0 && gx < nx - 1 && gy > 0 && gy < ny - 1) {
        const float T = T_in[gid];
        const float left = T_in[gid - 1];
        const float right = T_in[gid + 1];
        const float top = T_in[gid - nx];
        const float bottom = T_in[gid + nx];

        /* Heat diffusion + source term */
        T_out[gid] = T + rx * (left - 2.0f * T + right)
                       + ry * (top - 2.0f * T + bottom)
                       + q_factor * Q[gid];
    } else if (gx < nx && gy < ny) {
        T_out[gid] = T_in[gid];  /* Dirichlet BC */
    }
}

/*
 * 2D Thermal Diffusion (no heat source)
 *
 * Slightly faster when Q = 0 everywhere.
 */
__global__ __launch_bounds__(128, 16)
void thermal_2d_no_source_kernel(
    const float* __restrict__ T_in,
    float* __restrict__ T_out,
    const float rx,
    const float ry,
    const int nx,
    const int ny)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int gid = gy * nx + gx;

    if (gx > 0 && gx < nx - 1 && gy > 0 && gy < ny - 1) {
        const float T = T_in[gid];
        const float left = T_in[gid - 1];
        const float right = T_in[gid + 1];
        const float top = T_in[gid - nx];
        const float bottom = T_in[gid + nx];

        T_out[gid] = T + rx * (left - 2.0f * T + right)
                       + ry * (top - 2.0f * T + bottom);
    } else if (gx < nx && gy < ny) {
        T_out[gid] = T_in[gid];
    }
}

extern "C"
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
    cudaStream_t stream)
{
    if (nx <= 0 || ny <= 0) {
        return DENDRITE_ERROR_INVALID_ARGUMENT;
    }

    if (k <= 0.0f || rho <= 0.0f || Cp <= 0.0f) {
        return DENDRITE_ERROR_INVALID_ARGUMENT;
    }

    /* Thermal diffusivity */
    const float alpha = k / (rho * Cp);

    /* CFL coefficients */
    const float rx = alpha * dt / (dx * dx);
    const float ry = alpha * dt / (dy * dy);

    /* Check CFL condition */
    if (rx + ry > 0.5f) {
        return DENDRITE_ERROR_CFL_VIOLATION;
    }

    /* Heat source factor: Q * dt / (rho * Cp) */
    const float q_factor = dt / (rho * Cp);

    dim3 block(32, 4);
    dim3 grid((nx + 31) / 32, (ny + 3) / 4);

    if (Q != nullptr) {
        thermal_2d_kernel<<<grid, block, 0, stream>>>(
            T_in, T_out, Q, rx, ry, q_factor, nx, ny
        );
    } else {
        thermal_2d_no_source_kernel<<<grid, block, 0, stream>>>(
            T_in, T_out, rx, ry, nx, ny
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return DENDRITE_ERROR_CUDA;
    }

    return DENDRITE_SUCCESS;
}

/* Get max stable dt for thermal diffusion */
extern "C"
float dendrite_get_max_dt_thermal_2d(float k, float rho, float Cp, float dx, float dy)
{
    const float alpha = k / (rho * Cp);
    /* CFL: alpha*dt*(1/dx^2 + 1/dy^2) <= 0.5 */
    return 0.5f / (alpha * (1.0f / (dx * dx) + 1.0f / (dy * dy)));
}
