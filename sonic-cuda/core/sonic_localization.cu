#include <sonic-cuda/core/sonic_localization.h>
#include <sonic-cuda/core/sonic_roi.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <iostream>
#include <cstdio>

constexpr float PI = 3.14159265358979323846;

__device__ inline float fft_value_to_angle(cufftComplex fft_value) {
    return atan2(fft_value.y, fft_value.x);
}

__global__ void fft_value_to_locations(const cufftComplex* d_fft_values,
        const int* d_peak_x, const int* d_peak_y, int n_locs,
        float* d_loc_x, float* d_loc_y) {
    
    #define get_fft_value(n, x, y) (d_fft_values[(n) * ROI_WIDTH * (ROI_WIDTH / 2 + 1) + (x) * (ROI_WIDTH / 2 + 1) + (y)])
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_locs) {

        cufftComplex fft_value_x = get_fft_value(idx, 1, 0);
        float angle_x = fft_value_to_angle(fft_value_x);
        angle_x = angle_x - 2 * PI * (int) (angle_x > 0);

        // This kernel will produce the same result as the MATLAB Sonic code
        // Matlab peak uses 1-based indexing, we use 0-based indexing, so 1.0 is added to the offset
        // The Sonic localization considers the center of a 7x7 ROI as (4.0, 4.0), so additional 0.5 is added to the offset
        float offset_x = (abs(angle_x) / (2 * PI) * ROI_WIDTH) - ROI_WIDTH / 2.0 + 1.5;

        cufftComplex fft_value_y = get_fft_value(idx, 0, 1);
        float angle_y = fft_value_to_angle(fft_value_y);
        angle_y = angle_y - 2 * PI * (int) (angle_y > 0);
        float offset_y = (abs(angle_y) / (2 * PI) * ROI_WIDTH) - ROI_WIDTH / 2.0 + 1.5;

        d_loc_x[idx] = d_peak_x[idx] + offset_x;
        d_loc_y[idx] = d_peak_y[idx] + offset_y;
    }

    #undef get_fft_value
}

void sonic_localization(const float* d_roi, const int* d_peak_x, const int* d_peak_y, int n_locs,
        float* d_loc_x, float* d_loc_y) {
    
    cufftHandle plan;
    int dims[2] = {ROI_WIDTH, ROI_WIDTH};
    cufftComplex* fft_values;
    cudaMalloc((void**)&fft_values, n_locs * ROI_WIDTH * (ROI_WIDTH / 2 + 1) * sizeof(cufftComplex));

    cufftResult error_code = cufftPlanMany(
            &plan, 2, dims, NULL, 0, 0, NULL, 0, 0, CUFFT_R2C, n_locs);
    

    if (error_code != CUFFT_SUCCESS) {
        std::cerr << "Failed to create plan for FFT" << std::endl;
        exit(1);
    }
    
    error_code = cufftExecR2C(plan, (cufftReal*)d_roi, fft_values);
    if (error_code != CUFFT_SUCCESS) {
        std::cerr << "Failed to execute FFT" << std::endl;
        exit(1);
    }

    int block_size = 256;
    int grid_size = (n_locs + block_size - 1) / block_size;
    fft_value_to_locations<<<grid_size, block_size>>>(fft_values, d_peak_x, d_peak_y, n_locs, d_loc_x, d_loc_y);

    cudaFree(fft_values);
    cufftDestroy(plan);
}