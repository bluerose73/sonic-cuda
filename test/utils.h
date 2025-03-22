#ifndef SONIC_CUDA_TEST_UTILS_H
#define SONIC_CUDA_TEST_UTILS_H


#include <vector>
#include <string>


void read_data(const std::string& filename, std::vector<float>& data, int size);

void read_data(const std::string& filename, std::vector<int>& data);

void convert_to_zero_indexed(std::vector<int>& data);


#endif // SONIC_CUDA_TEST_UTILS_H