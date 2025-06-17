#include <sonic-cuda/core/sonic_roi.h>
#include <cuda_runtime.h>
#include <cstdio>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "sonic_roi:CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            goto cleanup; \
        } \
    } while(0)

// Kernel function to extract ROI
__global__ void extract_roi(const float* d_data, int height, int width, int frames,
    const int *d_peak_x, const int *d_peak_y, const int *d_peak_f, int n_locs,
    float* d_roi) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_locs) {
        int peak_y = d_peak_x[idx]; // d_peak_x is vertical
        int peak_x = d_peak_y[idx]; // d_peak_y is horizontal
        int peak_f = d_peak_f[idx];
        int half_width = ROI_WIDTH / 2;

        for (int i = -half_width; i <= half_width; ++i) {
            for (int j = -half_width; j <= half_width; ++j) {
                int y = peak_y + i;
                int x = peak_x + j;
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    d_roi[idx * ROI_WIDTH * ROI_WIDTH + (i + half_width) * ROI_WIDTH + (j + half_width)] =
                        d_data[peak_f * height * width + y * width + x];
                } else {
                    d_roi[idx * ROI_WIDTH * ROI_WIDTH + (i + half_width) * ROI_WIDTH + (j + half_width)] = 0.0f;
                }
            }
        }
    }
}

int sonic_roi(const float* d_data, int height, int width, int frames,
    const int *d_peak_x, const int *d_peak_y, const int *d_peak_f, int n_locs,
    float* d_roi) {
    
    // Early return if no locations to process
    if (n_locs == 0) {
        return 0;
    }
    
    // Input parameter validation
    if (d_data == nullptr) {
        fprintf(stderr, "sonic_roi:Error: d_data is NULL\n");
        return -1;
    }
    if (d_peak_x == nullptr || d_peak_y == nullptr || d_peak_f == nullptr) {
        fprintf(stderr, "sonic_roi:Error: peak arrays are NULL\n");
        return -1;
    }
    if (d_roi == nullptr) {
        fprintf(stderr, "sonic_roi:Error: d_roi is NULL\n");
        return -1;
    }
    if (height <= 0 || width <= 0 || frames <= 0) {
        fprintf(stderr, "sonic_roi:Error: invalid dimensions (height=%d, width=%d, frames=%d)\n", 
                height, width, frames);
        return -1;
    }
    if (n_locs < 0) {
        fprintf(stderr, "sonic_roi:Error: invalid n_locs (%d)\n", n_locs);
        return -1;
    }
    
    int block_size = 256;
    int grid_size = (n_locs + block_size - 1) / block_size;
    
    extract_roi<<<grid_size, block_size>>>(d_data, height, width, frames, d_peak_x, d_peak_y, d_peak_f, n_locs, d_roi);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;

cleanup:
    // Error occurred during kernel execution
    return -1;
}