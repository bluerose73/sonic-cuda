#include <test/utils.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

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

void read_data(const std::string& filename, std::vector<int>& data) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        data.resize(size / sizeof(int));
        file.read(reinterpret_cast<char*>(data.data()), size);
        file.close();
    } else {
        std::cerr << "Failed to open " << filename << std::endl;
    }
}

void convert_to_zero_indexed(std::vector<int>& data) {
    for (auto& val : data) {
        --val;
    }
}