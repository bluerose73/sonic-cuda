#include <sonic-cuda/core/sonic_clean_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "sonic_clean_v2:CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            goto cleanup; \
        } \
    } while(0)

static float h_filter_11[11][11];
static float h_peak_filter[7][7];
__constant__ static float d_filter_11[11][11];
__constant__ static float d_peak_filter[7][7];

#define IN_TILE_WIDTH 32

#define BLUR_MASK_WIDTH 11
#define BLUR_MASK_RADIUS 5
#define BLUR_OUT_TILE_WIDTH 22

#define LOCAL_MAX_MASK_WIDTH 3
#define LOCAL_MAX_MASK_RADIUS 1
#define LOCAL_MAX_OUT_TILE_WIDTH 30

#define PEAK_MASK_WIDTH 7
#define PEAK_MASK_RADIUS 3
#define PEAK_OUT_TILE_WIDTH 26

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

__global__ void blur_kernel(const float* d_data, const float* d_background,
        int height, int width, int frames, float threshold, int ignore_border_px, float* d_result) {
    __shared__ float tile_in[IN_TILE_WIDTH][IN_TILE_WIDTH];

    int f = blockIdx.z;
    int out_x = blockIdx.x * BLUR_OUT_TILE_WIDTH + threadIdx.x;
    int out_y = blockIdx.y * BLUR_OUT_TILE_WIDTH + threadIdx.y;

    int in_x = out_x - BLUR_MASK_RADIUS;
    int in_y = out_y - BLUR_MASK_RADIUS;

    if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
        tile_in[threadIdx.y][threadIdx.x] = d_data[f * height * width + in_y * width + in_x];
    } else {
        tile_in[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    float sum = 0;

    if (threadIdx.x < BLUR_OUT_TILE_WIDTH && threadIdx.y < BLUR_OUT_TILE_WIDTH
            && out_x < width && out_y < height) {
        for (int i = 0; i < BLUR_MASK_WIDTH; ++i) {
            for (int j = 0; j < BLUR_MASK_WIDTH; ++j) {
                sum += tile_in[threadIdx.y + i][threadIdx.x + j] * d_filter_11[i][j];
            }
        }

        if (out_x < ignore_border_px || out_x >= width - ignore_border_px ||
            out_y < ignore_border_px || out_y >= height - ignore_border_px) {
            sum = 0;
        }

        float adjusted_threshold = threshold + 4 * sqrt(
                d_background[f * height * width + out_y * width + out_x]);
        if (sum < adjusted_threshold) {
            sum = 0;
        }

        d_result[f * height * width + out_y * width + out_x] = sum;
    }
}

__global__ void local_max_kernel(const float* d_data, int height, int width, int frames,
        float* d_result) {
    __shared__ float tile_in[IN_TILE_WIDTH][IN_TILE_WIDTH];

    int f = blockIdx.z;
    int out_x = blockIdx.x * LOCAL_MAX_OUT_TILE_WIDTH + threadIdx.x;
    int out_y = blockIdx.y * LOCAL_MAX_OUT_TILE_WIDTH + threadIdx.y;

    int in_x = out_x - LOCAL_MAX_MASK_RADIUS;
    int in_y = out_y - LOCAL_MAX_MASK_RADIUS;

    if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
        tile_in[threadIdx.y][threadIdx.x] = d_data[f * height * width + in_y * width + in_x];
    } else {
        tile_in[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    if (threadIdx.x < LOCAL_MAX_OUT_TILE_WIDTH && threadIdx.y < LOCAL_MAX_OUT_TILE_WIDTH
            && out_x < width && out_y < height) {
        float center_value = tile_in[threadIdx.y + LOCAL_MAX_MASK_RADIUS][threadIdx.x + LOCAL_MAX_MASK_RADIUS];
        bool is_local_max = true;

        for (int i = 0; i < LOCAL_MAX_MASK_WIDTH; ++i) {
            for (int j = 0; j < LOCAL_MAX_MASK_WIDTH; ++j) {
                if (tile_in[threadIdx.y + i][threadIdx.x + j] > center_value) {
                    is_local_max = false;
                }
            }
        }

        if (is_local_max && in_x > 0 && in_x < width - 1 && in_y > 0 && in_y < height - 1) {
            d_result[f * height * width + out_y * width + out_x] = center_value;
        } else {
            d_result[f * height * width + out_y * width + out_x] = 0;
        }
    }
}

__global__ void find_peak_kernel(const float* d_data, int height, int width, int frames,
        int* peak_x, int* peak_y, int* peak_f, int* n_locs) {
    __shared__ float tile_in[IN_TILE_WIDTH][IN_TILE_WIDTH];

    int f = blockIdx.z;
    int out_x = blockIdx.x * PEAK_OUT_TILE_WIDTH + threadIdx.x;
    int out_y = blockIdx.y * PEAK_OUT_TILE_WIDTH + threadIdx.y;
    int in_x = out_x - PEAK_MASK_RADIUS;
    int in_y = out_y - PEAK_MASK_RADIUS;
    
    if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
        tile_in[threadIdx.y][threadIdx.x] = d_data[f * height * width + in_y * width + in_x];
    } else {
        tile_in[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    if (threadIdx.x < PEAK_OUT_TILE_WIDTH && threadIdx.y < PEAK_OUT_TILE_WIDTH
            && out_x < width && out_y < height) {
        float sum = 0;
        for (int i = 0; i < PEAK_MASK_WIDTH; ++i) {
            for (int j = 0; j < PEAK_MASK_WIDTH; ++j) {
                sum += tile_in[threadIdx.y + i][threadIdx.x + j] * d_peak_filter[i][j];
            }
        }

        if (sum > 0) {
            int idx = atomicAdd(n_locs, 1);
            peak_x[idx] = out_y;
            peak_y[idx] = out_x;
            peak_f[idx] = f;
        }
    }
}

int sonic_clean_v2(const float* d_data, int height, int width, int frames,
    const float* d_background, float threshold, int ignore_border_px,
    int* d_peak_x, int* d_peak_y, int* d_peak_f, int* n_locs) {
    
    // Initialize all variables at the beginning to avoid goto bypass issues
    int *n_loc = nullptr;
    float *d_blurred_data = nullptr, *d_local_max_data = nullptr;
    dim3 block_size(IN_TILE_WIDTH, IN_TILE_WIDTH);
    dim3 blur_grid_size, local_max_grid_size, peak_grid_size;
    
    // Input parameter validation
    if (d_data == nullptr) {
        fprintf(stderr, "sonic_clean_v2:Error: d_data is NULL\n");
        return -1;
    }
    if (d_background == nullptr) {
        fprintf(stderr, "sonic_clean_v2:Error: d_background is NULL\n");
        return -1;
    }
    if (d_peak_x == nullptr || d_peak_y == nullptr || d_peak_f == nullptr) {
        fprintf(stderr, "sonic_clean_v2:Error: peak arrays are NULL\n");
        return -1;
    }
    if (n_locs == nullptr) {
        fprintf(stderr, "sonic_clean_v2:Error: n_locs is NULL\n");
        return -1;
    }
    if (height <= 0 || width <= 0 || frames <= 0) {
        fprintf(stderr, "sonic_clean_v2:Error: invalid dimensions (height=%d, width=%d, frames=%d)\n", 
                height, width, frames);
        return -1;
    }
    
    fill_blur_filter_11(h_filter_11);
    fill_peak_filter(h_peak_filter);
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_filter_11, h_filter_11, 11 * 11 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_peak_filter, h_peak_filter, 7 * 7 * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&n_loc, sizeof(int)));
    CUDA_CHECK(cudaMemset(n_loc, 0, sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_blurred_data, frames * height * width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_local_max_data, frames * height * width * sizeof(float)));

    blur_grid_size = dim3((width - 1) / BLUR_OUT_TILE_WIDTH + 1, (height - 1) / BLUR_OUT_TILE_WIDTH + 1, frames);
    blur_kernel<<<blur_grid_size, block_size>>>(d_data, d_background, height, width, frames, threshold, ignore_border_px, d_blurred_data);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    local_max_grid_size = dim3((width - 1) / LOCAL_MAX_OUT_TILE_WIDTH + 1, (height - 1) / LOCAL_MAX_OUT_TILE_WIDTH + 1, frames);
    local_max_kernel<<<local_max_grid_size, block_size>>>(d_blurred_data, height, width, frames, d_local_max_data);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    peak_grid_size = dim3((width - 1) / PEAK_OUT_TILE_WIDTH + 1, (height - 1) / PEAK_OUT_TILE_WIDTH + 1, frames);
    find_peak_kernel<<<peak_grid_size, block_size>>>(d_local_max_data, height, width, frames, d_peak_x, d_peak_y, d_peak_f, n_loc);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(n_locs, n_loc, sizeof(int), cudaMemcpyDeviceToHost));

    // Success - clean up and return
    if (n_loc) cudaFree(n_loc);
    if (d_blurred_data) cudaFree(d_blurred_data);
    if (d_local_max_data) cudaFree(d_local_max_data);

    return 0;

cleanup:
    // Error occurred - free all allocated resources
    if (n_loc) cudaFree(n_loc);
    if (d_blurred_data) cudaFree(d_blurred_data);
    if (d_local_max_data) cudaFree(d_local_max_data);
    
    return -1;
}