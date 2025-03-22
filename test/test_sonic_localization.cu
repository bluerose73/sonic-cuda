#include <sonic-cuda/core/sonic_localization.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <filesystem>
#include <chrono>
#include <test/utils.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <output-dir>" << std::endl;
        return 1;
    }

    std::string output_dir = argv[1];

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);

    const int roi_size = 7;

    std::vector<int> h_peak_x;
    std::vector<int> h_peak_y;
    std::vector<int> h_peak_f;

    read_data("./test/data/peak_x.bin", h_peak_x);
    read_data("./test/data/peak_y.bin", h_peak_y);
    read_data("./test/data/F.bin", h_peak_f);

    convert_to_zero_indexed(h_peak_x);
    convert_to_zero_indexed(h_peak_y);
    convert_to_zero_indexed(h_peak_f);

    int n_locs = h_peak_f.size();
    std::vector<float> h_loc_x(n_locs);
    std::vector<float> h_loc_y(n_locs);

    std::vector<float> h_roi;
    read_data("./test/data/ROI.bin", h_roi, roi_size * roi_size * n_locs);

    float *d_roi, *d_loc_x, *d_loc_y;
    int *d_peak_x, *d_peak_y;

    cudaMalloc(&d_roi, h_roi.size() * sizeof(float));
    cudaMalloc(&d_peak_x, h_peak_x.size() * sizeof(int));
    cudaMalloc(&d_peak_y, h_peak_y.size() * sizeof(int));
    cudaMalloc(&d_loc_x, h_loc_x.size() * sizeof(float));
    cudaMalloc(&d_loc_y, h_loc_y.size() * sizeof(float));

    cudaMemcpy(d_roi, h_roi.data(), h_roi.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_peak_x, h_peak_x.data(), h_peak_x.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_peak_y, h_peak_y.data(), h_peak_y.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Print the first 7x7 ROI matrix for the first location
    std::cout << "First 7x7 ROI matrix for the first location:" << std::endl;
    for (int i = 0; i < roi_size; ++i) {
        for (int j = 0; j < roi_size; ++j) {
            std::cout << h_roi[i * roi_size + j] << " ";
        }
        std::cout << std::endl;
    }

    // timer starts here
    auto start = std::chrono::high_resolution_clock::now();

    sonic_localization(d_roi, d_peak_x, d_peak_y, n_locs, d_loc_x, d_loc_y);

    cudaDeviceSynchronize();

    // timer ends here
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;

    cudaMemcpy(h_loc_x.data(), d_loc_x, h_loc_x.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_loc_y.data(), d_loc_y, h_loc_y.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> given_loc_x;
    std::vector<float> given_loc_y;
    read_data("./test/data/X.bin", given_loc_x, n_locs);
    read_data("./test/data/Y.bin", given_loc_y, n_locs);

    bool match_x = (h_loc_x == given_loc_x);
    bool match_y = (h_loc_y == given_loc_y);
    std::cout << "Detected X locations match given X locations: " << (match_x ? "Yes" : "No") << std::endl;
    std::cout << "Detected Y locations match given Y locations: " << (match_y ? "Yes" : "No") << std::endl;

    // Comparison within 0.0001 tolerance
    auto within_tolerance = [](const std::vector<float>& a, const std::vector<float>& b, float tolerance) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(a[i] - b[i]) > tolerance) return false;
        }
        return true;
    };

    bool match_x_tolerance = within_tolerance(h_loc_x, given_loc_x, 0.0001f);
    bool match_y_tolerance = within_tolerance(h_loc_y, given_loc_y, 0.0001f);
    std::cout << "Detected X locations match given X locations within 0.0001 tolerance: " << (match_x_tolerance ? "Yes" : "No") << std::endl;
    std::cout << "Detected Y locations match given Y locations within 0.0001 tolerance: " << (match_y_tolerance ? "Yes" : "No") << std::endl;

    std::ofstream detected_loc_x_csv(output_dir + "/detected_loc_x.csv");
    if (detected_loc_x_csv.is_open()) {
        for (const auto& val : h_loc_x) {
            detected_loc_x_csv << val << "\n";
        }
        detected_loc_x_csv.close();
    } else {
        std::cerr << "Failed to open detected_loc_x.csv for writing" << std::endl;
    }

    std::ofstream detected_loc_y_csv(output_dir + "/detected_loc_y.csv");
    if (detected_loc_y_csv.is_open()) {
        for (const auto& val : h_loc_y) {
            detected_loc_y_csv << val << "\n";
        }
        detected_loc_y_csv.close();
    } else {
        std::cerr << "Failed to open detected_loc_y.csv for writing" << std::endl;
    }

    std::ofstream given_loc_x_csv(output_dir + "/given_loc_x.csv");
    if (given_loc_x_csv.is_open()) {
        for (const auto& val : given_loc_x) {
            given_loc_x_csv << val << "\n";
        }
        given_loc_x_csv.close();
    } else {
        std::cerr << "Failed to open given_loc_x.csv for writing" << std::endl;
    }

    std::ofstream given_loc_y_csv(output_dir + "/given_loc_y.csv");
    if (given_loc_y_csv.is_open()) {
        for (const auto& val : given_loc_y) {
            given_loc_y_csv << val << "\n";
        }
        given_loc_y_csv.close();
    } else {
        std::cerr << "Failed to open given_loc_y.csv for writing" << std::endl;
    }

    cudaFree(d_roi);
    cudaFree(d_peak_x);
    cudaFree(d_peak_y);
    cudaFree(d_loc_x);
    cudaFree(d_loc_y);

    return 0;
}
