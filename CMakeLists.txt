cmake_minimum_required(VERSION 3.20)

project(sonic-cuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Matlab REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

set(SOURCE_FILES
    sonic-cuda/core/sonic.cu
    sonic-cuda/core/sonic_clean.cu
    sonic-cuda/core/sonic_clean_v2.cu
    sonic-cuda/core/sonic_roi.cu
    sonic-cuda/core/sonic_localization.cu
)


add_library(sonic_cuda_lib ${SOURCE_FILES})
target_include_directories(sonic_cuda_lib PUBLIC .)


add_executable(test_sonic_clean test/test_sonic_clean.cu)
target_include_directories(test_sonic_clean PUBLIC .)
target_link_libraries(test_sonic_clean sonic_cuda_lib CUDA::cudart CUDA::cufft)

add_executable(test_sonic_roi test/test_sonic_roi.cu
                              test/utils.cpp)
target_include_directories(test_sonic_roi PUBLIC .)
target_link_libraries(test_sonic_roi sonic_cuda_lib CUDA::cudart CUDA::cufft)

add_executable(test_sonic_localization test/test_sonic_localization.cu
                                       test/utils.cpp)
target_include_directories(test_sonic_localization PUBLIC .)
target_link_libraries(test_sonic_localization sonic_cuda_lib CUDA::cudart CUDA::cufft)

add_executable(test_sonic test/test_sonic.cu
                          test/utils.cpp)
target_include_directories(test_sonic PUBLIC .)
target_link_libraries(test_sonic sonic_cuda_lib CUDA::cudart CUDA::cufft)

matlab_add_mex(NAME sonic_cuda
               SRC sonic-cuda/mex/sonic_cuda.cpp
               R2018a)
target_include_directories(sonic_cuda PRIVATE .)
target_link_libraries(sonic_cuda sonic_cuda_lib CUDA::cudart CUDA::cufft)