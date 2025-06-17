#include <sonic-cuda/core/sonic.h>
#include <sonic-cuda/core/sonic_clean_v2.h>
#include <sonic-cuda/core/sonic_clean.h>
#include <sonic-cuda/core/sonic_roi.h>
#include <sonic-cuda/core/sonic_localization.h>
#include <cuda_runtime.h>
#include <cstdio>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "sonic:CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            goto cleanup; \
        } \
    } while(0)

int sonic(const float* data, int height, int width, int frames,
          const float* background, float threshold, int ignore_border_px,
          float* loc_x, float* loc_y, int* loc_f, int* n_locs) {
    
    if (background == nullptr) {
        fprintf(stderr, "sonic:Error: background is NULL and compute_background is false\n");
        return -1;
    }

    // Initialize pointers to NULL for safe cleanup
    float *d_data = nullptr, *d_background = nullptr;
    int *d_peak_x = nullptr, *d_peak_y = nullptr, *d_peak_f = nullptr;
    float *d_roi = nullptr, *d_loc_x = nullptr, *d_loc_y = nullptr;
    int error_code = 0;

    // The peak filter guarantees that there is at most one local maximum in each 7x7 region.
    int max_loc_count = (height * width + 48) / 49 * frames;
    
    CUDA_CHECK(cudaMalloc(&d_data, height * width * frames * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_background, height * width * frames * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data, data, height * width * frames * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_background, background, height * width * frames * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_peak_x, max_loc_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_peak_y, max_loc_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_peak_f, max_loc_count * sizeof(int)));

    error_code = sonic_clean_v2(d_data, height, width, frames, d_background, threshold, ignore_border_px,
              d_peak_x, d_peak_y, d_peak_f, n_locs);
    if (error_code != 0) {
        fprintf(stderr, "sonic:Error in sonic_clean_v2: %d\n", error_code);
        goto cleanup;
    }
    
    CUDA_CHECK(cudaFree(d_background));
    d_background = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_roi, *n_locs * ROI_WIDTH * ROI_WIDTH * sizeof(float)));

    error_code = sonic_roi(d_data, height, width, frames, d_peak_x, d_peak_y, d_peak_f, *n_locs, d_roi);
    if (error_code != 0) {
        fprintf(stderr, "sonic:Error in sonic_roi: %d\n", error_code);
        goto cleanup;
    }

    CUDA_CHECK(cudaFree(d_data));
    d_data = nullptr;

    CUDA_CHECK(cudaMalloc(&d_loc_x, *n_locs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loc_y, *n_locs * sizeof(float)));
    
    error_code = sonic_localization(d_roi, d_peak_x, d_peak_y, *n_locs, d_loc_x, d_loc_y);
    if (error_code != 0) {
        fprintf(stderr, "sonic:Error in sonic_localization: %d\n", error_code);
        goto cleanup;
    }

    CUDA_CHECK(cudaMemcpy(loc_x, d_loc_x, *n_locs * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(loc_y, d_loc_y, *n_locs * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(loc_f, d_peak_f, *n_locs * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Success - clean up and return
    if (d_peak_x) cudaFree(d_peak_x);
    if (d_peak_y) cudaFree(d_peak_y);
    if (d_peak_f) cudaFree(d_peak_f);
    if (d_roi) cudaFree(d_roi);
    if (d_loc_x) cudaFree(d_loc_x);
    if (d_loc_y) cudaFree(d_loc_y);

    return 0;

cleanup:
    // Error occurred - free all allocated resources
    if (d_data) cudaFree(d_data);
    if (d_background) cudaFree(d_background);
    if (d_peak_x) cudaFree(d_peak_x);
    if (d_peak_y) cudaFree(d_peak_y);
    if (d_peak_f) cudaFree(d_peak_f);
    if (d_roi) cudaFree(d_roi);
    if (d_loc_x) cudaFree(d_loc_x);
    if (d_loc_y) cudaFree(d_loc_y);
    
    return -1;
}