#include <sonic-cuda/core/sonic_localization.h>
#include <sonic-cuda/core/sonic_roi.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <iostream>
#include <cstdio>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "sonic_localization:CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            goto cleanup; \
        } \
    } while(0)

// cuFFT error checking macro
#define CUFFT_CHECK(call) \
    do { \
        cufftResult error = call; \
        if (error != CUFFT_SUCCESS) { \
            fprintf(stderr, "sonic_localization:cuFFT error at %s:%d - %s\n", __FILE__, __LINE__, cufftGetErrorString(error)); \
            goto cleanup; \
        } \
    } while(0)

constexpr float PI = 3.14159265358979323846;

const char* cufftGetErrorString(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
        case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
        case CUFFT_PARSE_ERROR: return "CUFFT_PARSE_ERROR";
        case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
        case CUFFT_LICENSE_ERROR: return "CUFFT_LICENSE_ERROR";
        case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
        default: return "Unknown CUFFT error";
    }
}

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

int sonic_localization(const float* d_roi, const int* d_peak_x, const int* d_peak_y, int n_locs,
        float* d_loc_x, float* d_loc_y) {
    
    // Early return if no locations to process
    if (n_locs == 0) {
        return 0;
    }
    
    // Input parameter validation
    if (d_roi == nullptr) {
        fprintf(stderr, "sonic_localization:Error: d_roi is NULL\n");
        return -1;
    }
    if (d_peak_x == nullptr || d_peak_y == nullptr) {
        fprintf(stderr, "sonic_localization:Error: peak arrays are NULL\n");
        return -1;
    }
    if (d_loc_x == nullptr || d_loc_y == nullptr) {
        fprintf(stderr, "sonic_localization:Error: location arrays are NULL\n");
        return -1;
    }
    if (n_locs < 0) {
        fprintf(stderr, "sonic_localization:Error: invalid n_locs (%d)\n", n_locs);
        return -1;
    }

    // Initialize all variables at the beginning to avoid goto bypass issues
    cufftHandle plan = 0;
    cufftComplex* fft_values = nullptr;
    int dims[2] = {ROI_WIDTH, ROI_WIDTH};
    int block_size = 256;
    int grid_size;
    bool plan_created = false;
    
    CUDA_CHECK(cudaMalloc((void**)&fft_values, n_locs * ROI_WIDTH * (ROI_WIDTH / 2 + 1) * sizeof(cufftComplex)));

    CUFFT_CHECK(cufftPlanMany(&plan, 2, dims, NULL, 0, 0, NULL, 0, 0, CUFFT_R2C, n_locs));
    plan_created = true;
    
    CUFFT_CHECK(cufftExecR2C(plan, (cufftReal*)d_roi, fft_values));

    grid_size = (n_locs + block_size - 1) / block_size;
    fft_value_to_locations<<<grid_size, block_size>>>(fft_values, d_peak_x, d_peak_y, n_locs, d_loc_x, d_loc_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Success - clean up and return
    if (fft_values) cudaFree(fft_values);
    if (plan_created) cufftDestroy(plan);
    
    return 0;

cleanup:
    // Error occurred - free all allocated resources
    if (fft_values) cudaFree(fft_values);
    if (plan_created) cufftDestroy(plan);
    
    return -1;
}