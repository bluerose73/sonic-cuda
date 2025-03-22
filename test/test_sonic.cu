#include <sonic-cuda/core/sonic.h>
#include <test/utils.h>
#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <fstream>

// Function to sort locations by frames, then by x, then by y
void sort_locations(std::vector<float>& x, std::vector<float>& y, std::vector<int>& f, size_t size) {
    // Create indices vector
    std::vector<size_t> indices(size);
    for (size_t i = 0; i < size; ++i) {
        indices[i] = i;
    }
    
    // Sort indices based on f, then x, then y
    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        if (f[i] != f[j]) return f[i] < f[j];
        if (x[i] != x[j]) return x[i] < x[j];
        return y[i] < y[j];
    });
    
    // Create temporary vectors for sorted data
    std::vector<float> sorted_x(size);
    std::vector<float> sorted_y(size);
    std::vector<int> sorted_f(size);
    
    // Rearrange the data according to sorted indices
    for (size_t i = 0; i < size; ++i) {
        sorted_x[i] = x[indices[i]];
        sorted_y[i] = y[indices[i]];
        sorted_f[i] = f[indices[i]];
    }
    
    // Copy sorted data back to original vectors
    x = sorted_x;
    y = sorted_y;
    f = sorted_f;
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
    const float threshold = 100.0f;
    const int ignore_border_px = 15;

    std::vector<float> h_data;
    std::vector<float> h_background;
    read_data("./test/data/imdata.bin", h_data, height * width * frames);
    read_data("./test/data/bg.bin", h_background, height * width * frames);

    const int max_n_locs = (height * width + 48) / 49 * frames;
    std::vector<float> loc_x(max_n_locs);
    std::vector<float> loc_y(max_n_locs);
    std::vector<int> loc_f(max_n_locs);
    int n_locs = 0;

    auto start = std::chrono::high_resolution_clock::now();

    sonic(h_data.data(), height, width, frames, h_background.data(), threshold, ignore_border_px,
          loc_x.data(), loc_y.data(), loc_f.data(), &n_locs);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Elapsed time: " << duration.count() << " ms" << std::endl;
    
    std::vector<int> given_loc_f;
    read_data("./test/data/F.bin", given_loc_f);
    int given_n_locs = given_loc_f.size();

    std::vector<float> given_loc_x;
    std::vector<float> given_loc_y;
    read_data("./test/data/X.bin", given_loc_x, given_n_locs);
    read_data("./test/data/Y.bin", given_loc_y, given_n_locs);

    std::cout << "Detected number of locations: " << n_locs << std::endl;
    std::cout << "Given number of locations: " << given_n_locs << std::endl;

    // Take account of row-major (c++) vs column-major (matlab) difference
    std::swap(loc_x, loc_y);
    // Take account of 1-based indexing in matlab
    for (int& f : loc_f) {
        f += 1;
    }

    // Sort detected locations
    sort_locations(loc_x, loc_y, loc_f, n_locs);
    
    // Sort given locations
    sort_locations(given_loc_x, given_loc_y, given_loc_f, given_n_locs);

    bool match_x = (loc_x == given_loc_x);
    bool match_y = (loc_y == given_loc_y);
    bool match_f = (loc_f == given_loc_f);
    std::cout << "Detected X locations match given X locations: " << (match_x ? "Yes" : "No") << std::endl;
    std::cout << "Detected Y locations match given Y locations: " << (match_y ? "Yes" : "No") << std::endl;
    std::cout << "Detected frames match given frames: " << (match_f ? "Yes" : "No") << std::endl;

    // Comparison within 0.0001 tolerance
    auto within_tolerance = [](const std::vector<float>& a, const std::vector<float>& b, float tolerance) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(a[i] - b[i]) > tolerance) return false;
        }
        return true;
    };

    bool match_x_tolerance = within_tolerance(loc_x, given_loc_x, 0.0001f);
    bool match_y_tolerance = within_tolerance(loc_y, given_loc_y, 0.0001f);
    std::cout << "Detected X locations match given X locations within 0.0001 tolerance: " << (match_x_tolerance ? "Yes" : "No") << std::endl;
    std::cout << "Detected Y locations match given Y locations within 0.0001 tolerance: " << (match_y_tolerance ? "Yes" : "No") << std::endl;

    std::ofstream detected_locs_csv(output_dir + "/detected_locs.csv");
    if (detected_locs_csv.is_open()) {
        detected_locs_csv << "x,y,f\n";
        for (int i = 0; i < n_locs; ++i) {
            detected_locs_csv << loc_x[i] << "," << loc_y[i] << "," << loc_f[i] << "\n";
        }
        detected_locs_csv.close();
    } else {
        std::cerr << "Failed to open detected_locs.csv for writing" << std::endl;
        exit(1);
    }

    std::ofstream given_locs_csv(output_dir + "/given_locs.csv");
    if (given_locs_csv.is_open()) {
        given_locs_csv << "x,y,f\n";
        for (int i = 0; i < given_n_locs; ++i) {
            given_locs_csv << given_loc_x[i] << "," << given_loc_y[i] << "," << given_loc_f[i] << "\n";
        }
        given_locs_csv.close();
    } else {
        std::cerr << "Failed to open given_locs.csv for writing" << std::endl;
        exit(1);
    }


}