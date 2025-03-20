#include <sonic-cuda/core/sonic_clean.h>
#include <cuda_runtime.h>

/*
32 * 32 input tile
22 * 22 blurred tile
20 * 20 regional max tile
14 * 14 output tile
*/

#define IN_TILE_WIDTH 32
#define MED_TILE_WIDTH 22  // 22 = 32 - 11 + 1
#define LOCAL_MAX_TILE_WIDTH 20  // 20 = 22 - 3 + 1
#define OUT_TILE_WIDTH 14  // 14 = 20 - 7 + 1

static float h_filter_11[11][11];
static float h_peak_filter[7][7];
__constant__ static float d_filter_11[11][11];
__constant__ static float d_peak_filter[7][7];

static void fill_blur_filter_11(float filter[11][11]) {
    for (int i = 0; i < 11; ++i) {
        for (int j = 0; j < 11; ++j) {
            filter[i][j] = -1.0 / 112;
        }
    }
    
    for (int i = 4; i <= 6; ++i) {
        for (int j = 4; j <= 6; ++j) {
            filter[i][j] = 1.0 / 10;
        }
    }
    
    filter[5][5] = 1.0 / 5;
}

static void fill_peak_filter(float filter[7][7]) {
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            filter[i][j] = -1.0;
        }
    }

    filter[3][3] = 1.0;
}

__global__ void sonic_clean_kernel(
        const float* data, int height, int width, int frames,
        const float* background, float threshold, int ignore_border_px,
        int* peak_x, int* peak_y, int* peak_f, int* n_locs) {
    
    __shared__ float tile_in[IN_TILE_WIDTH][IN_TILE_WIDTH];
    __shared__ float tile_med[MED_TILE_WIDTH][MED_TILE_WIDTH];

    int f = blockIdx.z;

    int out_x = blockIdx.x * OUT_TILE_WIDTH + threadIdx.x;
    int out_y = blockIdx.y * OUT_TILE_WIDTH + threadIdx.y;

    int local_max_x = out_x - 3;
    int local_max_y = out_y - 3;

    int med_x = out_x - 4;
    int med_y = out_y - 4;

    int in_x = out_x - 9;
    int in_y = out_y - 9;

    // Load data
    if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
        tile_in[threadIdx.y][threadIdx.x] = data[f * height * width + in_y * width + in_x];
    } else {
        tile_in[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    // Blur filter: 11 x 11
    if (threadIdx.x < MED_TILE_WIDTH && threadIdx.y < MED_TILE_WIDTH) {
        float sum = 0;
        for (int i = 0; i < 11; i++) {
            for (int j = 0; j < 11; j++) {
                sum += tile_in[threadIdx.y + i][threadIdx.x + j] * d_filter_11[i][j];
            }
        }

        // Apply threshold
        float adjusted_threshold = threshold + 4 * sqrt(
                background[f * height * width + med_y * width + med_x]);
        if (sum < adjusted_threshold) {
            sum = 0;
        }

        // Apply border cropping
        if (med_x < ignore_border_px || med_x >= width - ignore_border_px ||
            med_y < ignore_border_px || med_y >= height - ignore_border_px) {
            sum = 0;
        }

        tile_med[threadIdx.y][threadIdx.x] = sum;
    }
    __syncthreads();


    // Find 3x3 8-connected local maxima
    if (threadIdx.x < LOCAL_MAX_TILE_WIDTH && threadIdx.y < LOCAL_MAX_TILE_WIDTH) {
        float center = tile_med[threadIdx.y + 1][threadIdx.x + 1];
        bool is_maxima = true;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (tile_med[threadIdx.y + i + 1][threadIdx.x + j + 1] > center) {
                    is_maxima = false;
                }
            }
        }
        if (local_max_x <= 0 || local_max_x >= width - 1 ||
            local_max_y <= 0 || local_max_y >= height - 1) {
            is_maxima = false;
        }
        if (is_maxima) {
            tile_in[threadIdx.y][threadIdx.x] = center;
        } else {
            tile_in[threadIdx.y][threadIdx.x] = 0;
        }
    }
    __syncthreads();
            
    // Apply peak filter: 7 x 7 and store local maxima
    if (threadIdx.x < OUT_TILE_WIDTH && threadIdx.y < OUT_TILE_WIDTH
            && out_x < width && out_y < height) {
        float sum = 0;
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                sum += tile_in[threadIdx.y + i][threadIdx.x + j] * d_peak_filter[i][j];
            }
        }
        if (sum > 0) {
            int idx = atomicAdd(n_locs, 1);

            // In the kernel, the x dimension is horizontal,
            // but in the output, the x dimension is vertical.
            peak_x[idx] = out_y;
            peak_y[idx] = out_x;
            peak_f[idx] = f;
        }
    }
}

void sonic_clean(const float* d_data, int height, int width, int frames,
                  const float* d_background, float threshold, int ignore_border_px,
                  int* d_peak_x, int* d_peak_y, int* d_peak_f, int* n_locs) {
    fill_blur_filter_11(h_filter_11);
    fill_peak_filter(h_peak_filter);
    cudaMemcpyToSymbol(d_filter_11, h_filter_11, 11 * 11 * sizeof(float));
    cudaMemcpyToSymbol(d_peak_filter, h_peak_filter, 7 * 7 * sizeof(float));

    int *d_n_locs;
    cudaMalloc(&d_n_locs, sizeof(int));
    cudaMemset(d_n_locs, 0, sizeof(int));

    dim3 grid((width - 1) / OUT_TILE_WIDTH + 1, (height - 1) / OUT_TILE_WIDTH + 1, frames);
    dim3 block(IN_TILE_WIDTH, IN_TILE_WIDTH);
    sonic_clean_kernel<<<grid, block>>>(d_data, height, width, frames, d_background,
            threshold, ignore_border_px, d_peak_x, d_peak_y, d_peak_f, d_n_locs);
    
    cudaMemcpy(n_locs, d_n_locs, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_n_locs);
}