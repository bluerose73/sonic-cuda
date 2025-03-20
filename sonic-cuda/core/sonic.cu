#include <sonic-cuda/core/sonic.h>
#include <sonic-cuda/core/sonic_clean.h>


void sonic(const float* data, int height, int width, int frames,
           const float* background, float threshold, int ignore_border_px,
           float* loc_x, float* loc_y, float* loc_f, int* n_loc) {
    

    // The peak filter guarantees that there is at most one local maximum in each 7x7 region.
    int max_loc_count = (height * width + 48) / 49 * frames;
    float *d_data, *d_background;
    cudaMalloc(&d_data, height * width * frames * sizeof(float));
    cudaMalloc(&d_background, height * width * sizeof(float));

    cudaMemcpy(d_data, data, height * width * frames * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_background, background, height * width * sizeof(float), cudaMemcpyHostToDevice);

    int *d_peak_x, *d_peak_y, *d_peak_f;
    cudaMalloc(&d_peak_x, max_loc_count * sizeof(int));
    cudaMalloc(&d_peak_y, max_loc_count * sizeof(int));
    cudaMalloc(&d_peak_f, max_loc_count * sizeof(int));

    sonic_clean(d_data, height, width, frames, d_background, threshold, ignore_border_px,
                d_peak_x, d_peak_y, d_peak_f, n_loc);

    
}