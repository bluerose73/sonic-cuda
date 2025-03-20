cmake_minimum_required(VERSION 3.20)

project(sonic-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

set(SOURCE_FILES
    sonic-cuda/core/sonic.cu
    sonic-cuda/core/sonic_clean.cu
    sonic-cuda/core/sonic_clean_v2.cu
)

add_library(sonic_cuda_lib ${SOURCE_FILES})
target_include_directories(sonic_cuda_lib PUBLIC .)

add_executable(test_sonic_clean test/test_sonic_clean.cu)
target_include_directories(test_sonic_clean PUBLIC .)
target_link_libraries(test_sonic_clean sonic_cuda_lib ${CUDA_LIBRARIES})
