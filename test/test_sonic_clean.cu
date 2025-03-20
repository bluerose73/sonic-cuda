#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <sonic-cuda/core/sonic_clean.h>
#include <sonic-cuda/core/sonic_clean_v2.h>
#include <cmath>
#include <filesystem>

// Test data are in ./test/data folder
// imdata.bin float
// bg.bin float
// peak_x.bin int
// peak_y.bin int
// F.bin int
// threshold = 100
// ignore_border_px = 15

void read_data(const std::string& filename, std::vector<float>& data, int size) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        data.resize(size);
        file.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));
        file.close();
    } else {
        std::cerr << "Failed to open " << filename << std::endl;
    }
}

void read_data(const std::string& filename, std::vector<int>& data, int size) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        data.resize(size);
        file.read(reinterpret_cast<char*>(data.data()), size * sizeof(int));
        file.close();
    } else {
        std::cerr << "Failed to open " << filename << std::endl;
    }
}

struct Peak {
    int x, y, f;
    bool operator<(const Peak& other) const {
        if (f != other.f) return f < other.f;
        if (y != other.y) return y < other.y;
        return x < other.x;
    }
    bool operator==(const Peak& other) const {
        return x == other.x && y == other.y && f == other.f;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <version> <output-dir>" << std::endl;
        return 1;
    }

    std::string version = argv[1];
    std::string output_dir = argv[2];

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);

    const int height = 64;
    const int width = 64;
    const int frames = 2500;
    const float threshold = 100.0f;
    const int ignore_border_px = 15;
    const int peak_size = static_cast<int>(std::ceil(static_cast<float>(width * height) / 49)) * frames;

    std::vector<float> h_data;
    std::vector<float> h_background;
    std::vector<int> h_peak_x;
    std::vector<int> h_peak_y;
    std::vector<int> h_peak_f;
    std::vector<float> h_blurred_data;
    std::vector<float> h_local_max_data;
    int n_locs = 0;

    read_data("./test/data/imdata.bin", h_data, height * width * frames);
    read_data("./test/data/bg.bin", h_background, height * width * frames);

    // Print top-left 3x3 tile of imdata
    std::cout << "Top-left 3x3 tile of imdata:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << h_data[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Print top-left 3x3 tile of background
    std::cout << "Top-left 3x3 tile of background:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << h_background[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    h_peak_x.resize(peak_size);
    h_peak_y.resize(peak_size);
    h_peak_f.resize(peak_size);
    h_blurred_data.resize(height * width * frames);
    h_local_max_data.resize(height * width * frames);

    float *d_data, *d_background;
    int *d_peak_x, *d_peak_y, *d_peak_f;
    float *d_blurred_data, *d_local_max_data;  // for debugging

    cudaMalloc(&d_data, h_data.size() * sizeof(float));
    cudaMalloc(&d_background, h_background.size() * sizeof(float));
    cudaMalloc(&d_peak_x, h_peak_x.size() * sizeof(int));
    cudaMalloc(&d_peak_y, h_peak_y.size() * sizeof(int));
    cudaMalloc(&d_peak_f, h_peak_f.size() * sizeof(int));
    cudaMalloc(&d_blurred_data, h_data.size() * sizeof(float));
    cudaMalloc(&d_local_max_data, h_data.size() * sizeof(float));

    cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_background, h_background.data(), h_background.size() * sizeof(float), cudaMemcpyHostToDevice);

    if (version == "v1") {
        sonic_clean(d_data, height, width, frames, d_background, threshold, ignore_border_px, d_peak_x, d_peak_y, d_peak_f, &n_locs,
                    d_blurred_data, d_local_max_data);
    } else if (version == "v2") {
        sonic_clean_v2(d_data, height, width, frames, d_background, threshold, ignore_border_px, d_peak_x, d_peak_y, d_peak_f, &n_locs,
                       d_blurred_data, d_local_max_data);
    } else {
        std::cerr << "Invalid version specified. Use 'v1' or 'v2'." << std::endl;
        return 1;
    }

    cudaMemcpy(h_peak_x.data(), d_peak_x, n_locs * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_peak_y.data(), d_peak_y, n_locs * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_peak_f.data(), d_peak_f, n_locs * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blurred_data.data(), d_blurred_data, h_blurred_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_local_max_data.data(), d_local_max_data, h_local_max_data.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<Peak> detected_peaks(n_locs);
    for (int i = 0; i < n_locs; ++i) {
        detected_peaks[i] = {h_peak_x[i], h_peak_y[i], h_peak_f[i]};
    }
    std::sort(detected_peaks.begin(), detected_peaks.end());

    std::vector<int> given_peak_x, given_peak_y, given_peak_f;
    read_data("./test/data/peak_x.bin", given_peak_x, n_locs);
    read_data("./test/data/peak_y.bin", given_peak_y, n_locs);
    read_data("./test/data/F.bin", given_peak_f, n_locs);

    // Print first 3 values in given peaks before sorting
    std::cout << "First 3 values in given peaks before sorting:" << std::endl;
    for (int i = 0; i < std::min(3, n_locs); ++i) {
        std::cout << "Given Peak " << i << ": (" << given_peak_x[i] << ", " << given_peak_y[i] << ", " << given_peak_f[i] << ")" << std::endl;
    }

    std::vector<Peak> given_peaks(n_locs);
    for (int i = 0; i < n_locs; ++i) {
        // matlab is column-major
        // and it uses 1-based indexing
        given_peaks[i] = {given_peak_y[i] - 1, given_peak_x[i] - 1, given_peak_f[i] - 1};
    }
    std::sort(given_peaks.begin(), given_peaks.end());

    bool match = (detected_peaks == given_peaks);
    std::cout << "Detected peaks match given peaks: " << (match ? "Yes" : "No") << std::endl;

    std::ofstream detected_peaks_csv(output_dir + "/detected_peaks.csv");
    if (detected_peaks_csv.is_open()) {
        detected_peaks_csv << "x,y,f\n";
        for (const auto& peak : detected_peaks) {
            detected_peaks_csv << peak.x << "," << peak.y << "," << peak.f << "\n";
        }
        detected_peaks_csv.close();
    } else {
        std::cerr << "Failed to open detected_peaks.csv for writing" << std::endl;
    }

    std::ofstream given_peaks_csv(output_dir + "/given_peaks.csv");
    if (given_peaks_csv.is_open()) {
        given_peaks_csv << "x,y,f\n";
        for (const auto& peak : given_peaks) {
            given_peaks_csv << peak.x << "," << peak.y << "," << peak.f << "\n";
        }
        given_peaks_csv.close();
    } else {
        std::cerr << "Failed to open given_peaks.csv for writing" << std::endl;
    }

    std::ofstream blurred_data_csv(output_dir + "/blurred_data.csv");
    if (blurred_data_csv.is_open()) {
        // dump the first frame
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                blurred_data_csv << h_blurred_data[j * height + i] << ",";
            }
            blurred_data_csv << "\n";
        }
        blurred_data_csv.close();
    } else {
        std::cerr << "Failed to open blurred_data.csv for writing" << std::endl;
    }

    std::ofstream local_max_data_csv(output_dir + "/local_max_data.csv");
    if (local_max_data_csv.is_open()) {
        // dump the first frame
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                local_max_data_csv << h_local_max_data[j * height + i] << ",";
            }
            local_max_data_csv << "\n";
        }
        local_max_data_csv.close();
    } else {
        std::cerr << "Failed to open local_max_data.csv for writing" << std::endl;
    }

    cudaFree(d_data);
    cudaFree(d_background);
    cudaFree(d_peak_x);
    cudaFree(d_peak_y);
    cudaFree(d_peak_f);

    return 0;
}