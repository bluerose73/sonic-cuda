#include <sonic-cuda/core/sonic_roi.h>
#include <cuda_runtime.h>

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

void sonic_roi(const float* d_data, int height, int width, int frames,
    const int *d_peak_x, const int *d_peak_y, const int *d_peak_f, int n_locs,
    float* d_roi) {
    
    int block_size = 256;
    int grid_size = (n_locs + block_size - 1) / block_size;
    extract_roi<<<grid_size, block_size>>>(d_data, height, width, frames, d_peak_x, d_peak_y, d_peak_f, n_locs, d_roi);
}