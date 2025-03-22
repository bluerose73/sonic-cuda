#include <sonic-cuda/core/sonic.h>
#include <sonic-cuda/core/sonic_clean_v2.h>
#include <sonic-cuda/core/sonic_clean.h>
#include <sonic-cuda/core/sonic_roi.h>
#include <sonic-cuda/core/sonic_localization.h>
#include <cuda_runtime.h>
#include <cstdio>

void sonic(const float* data, int height, int width, int frames,
           const float* background, float threshold, int ignore_border_px,
           float* loc_x, float* loc_y, int* loc_f, int* n_locs) {
    

    // The peak filter guarantees that there is at most one local maximum in each 7x7 region.
    int max_loc_count = (height * width + 48) / 49 * frames;
    float *d_data, *d_background;
    cudaMalloc(&d_data, height * width * frames * sizeof(float));
    cudaMalloc(&d_background, height * width * frames * sizeof(float));

    cudaMemcpy(d_data, data, height * width * frames * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_background, background, height * width * frames * sizeof(float), cudaMemcpyHostToDevice);

    int *d_peak_x, *d_peak_y, *d_peak_f;
    cudaMalloc(&d_peak_x, max_loc_count * sizeof(int));
    cudaMalloc(&d_peak_y, max_loc_count * sizeof(int));
    cudaMalloc(&d_peak_f, max_loc_count * sizeof(int));

    sonic_clean_v2(d_data, height, width, frames, d_background, threshold, ignore_border_px,
              d_peak_x, d_peak_y, d_peak_f, n_locs);
    
    cudaFree(d_background);
    
    float *d_roi;
    cudaMalloc(&d_roi, *n_locs * ROI_WIDTH * ROI_WIDTH * sizeof(float));

    sonic_roi(d_data, height, width, frames, d_peak_x, d_peak_y, d_peak_f, *n_locs, d_roi);
    
    cudaFree(d_data);

    float *d_loc_x, *d_loc_y;
    cudaMalloc(&d_loc_x, *n_locs * sizeof(float));
    cudaMalloc(&d_loc_y, *n_locs * sizeof(float));
    
    sonic_localization(d_roi, d_peak_x, d_peak_y, *n_locs, d_loc_x, d_loc_y);

    cudaMemcpy(loc_x, d_loc_x, *n_locs * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(loc_y, d_loc_y, *n_locs * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(loc_f, d_peak_f, *n_locs * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_peak_x);
    cudaFree(d_peak_y);
    cudaFree(d_peak_f);
    cudaFree(d_roi);
    cudaFree(d_loc_x);
    cudaFree(d_loc_y);

}