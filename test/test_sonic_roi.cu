#include <sonic-cuda/core/sonic_roi.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <filesystem>
#include <chrono>
#include <test/utils.h>

void convert_to_row_major(const std::vector<float>& col_major, std::vector<float>& row_major, int height, int width, int frames) {
    for (int f = 0; f < frames; ++f) {
        for (int w = 0; w < width; ++w) {
            for (int h = 0; h < height; ++h) {
                row_major[f * height * width + h * width + w] = col_major[f * height * width + w * height + h];
            }
        }
    }
}



int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <output-dir>" << std::endl;
        return 1;
    }

    std::string output_dir = argv[1];

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);

    const int height = 1024;
    const int width = 1024;
    const int frames = 100;
    const int roi_size = 7;

    std::vector<float> h_data_col_major;
    std::vector<int> h_peak_x;
    std::vector<int> h_peak_y;
    std::vector<int> h_peak_f;

    read_data("./test/data/imdata.bin", h_data_col_major, height * width * frames);

    std::vector<float> h_data(height * width * frames);
    convert_to_row_major(h_data_col_major, h_data, height, width, frames);

    // Print top-left 3x3 of the row-major imdata
    std::cout << "Top-left 3x3 of the row-major imdata:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << h_data[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    read_data("./test/data/peak_x.bin", h_peak_x);
    read_data("./test/data/peak_y.bin", h_peak_y);
    read_data("./test/data/F.bin", h_peak_f);

    convert_to_zero_indexed(h_peak_x);
    convert_to_zero_indexed(h_peak_y);
    convert_to_zero_indexed(h_peak_f);

    // Print first 3 elements of the corrected (x, y, f)
    std::cout << "First 3 elements of the corrected (x, y, f):" << std::endl;
    for (int i = 0; i < 3; ++i) {
        std::cout << "(" << h_peak_x[i] << ", " << h_peak_y[i] << ", " << h_peak_f[i] << ")" << std::endl;
    }

    int n_locs = h_peak_f.size();
    std::vector<float> h_roi(n_locs * roi_size * roi_size);

    float *d_data, *d_roi;
    int *d_peak_x, *d_peak_y, *d_peak_f;

    cudaMalloc(&d_data, h_data.size() * sizeof(float));
    cudaMalloc(&d_peak_x, h_peak_x.size() * sizeof(int));
    cudaMalloc(&d_peak_y, h_peak_y.size() * sizeof(int));
    cudaMalloc(&d_peak_f, h_peak_f.size() * sizeof(int));
    cudaMalloc(&d_roi, h_roi.size() * sizeof(float));

    cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_peak_x, h_peak_x.data(), h_peak_x.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_peak_y, h_peak_y.data(), h_peak_y.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_peak_f, h_peak_f.data(), h_peak_f.size() * sizeof(int), cudaMemcpyHostToDevice);

    // timer starts here
    auto start = std::chrono::high_resolution_clock::now();

    sonic_roi(d_data, height, width, frames, d_peak_x, d_peak_y, d_peak_f, n_locs, d_roi);

    cudaDeviceSynchronize();

    // timer ends here
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;

    cudaMemcpy(h_roi.data(), d_roi, h_roi.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> given_roi;
    read_data("./test/data/ROI.bin", given_roi, n_locs * roi_size * roi_size);


    bool match = (h_roi == given_roi);
    std::cout << "Detected ROI match given ROI: " << (match ? "Yes" : "No") << std::endl;

    std::ofstream detected_roi_csv(output_dir + "/detected_roi.csv");
    if (detected_roi_csv.is_open()) {
        for (int i = 0; i < n_locs; ++i) {
            for (int j = 0; j < roi_size; ++j) {
                for (int k = 0; k < roi_size; ++k) {
                    detected_roi_csv << h_roi[i * roi_size * roi_size + j * roi_size + k] << ",";
                }
                detected_roi_csv << "\n";
            }
            detected_roi_csv << "\n";
        }
        detected_roi_csv.close();
    } else {
        std::cerr << "Failed to open detected_roi.csv for writing" << std::endl;
    }

    std::ofstream given_roi_csv(output_dir + "/given_roi.csv");
    if (given_roi_csv.is_open()) {
        for (int i = 0; i < n_locs; ++i) {
            for (int j = 0; j < roi_size; ++j) {
                for (int k = 0; k < roi_size; ++k) {
                    given_roi_csv << given_roi[i * roi_size * roi_size + j * roi_size + k] << ",";
                }
                given_roi_csv << "\n";
            }
            given_roi_csv << "\n";
        }
        given_roi_csv.close();
    } else {
        std::cerr << "Failed to open given_roi.csv for writing" << std::endl;
    }

    cudaFree(d_data);
    cudaFree(d_peak_x);
    cudaFree(d_peak_y);
    cudaFree(d_peak_f);
    cudaFree(d_roi);

    return 0;
}


