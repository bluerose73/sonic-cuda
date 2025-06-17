# sonic-cuda

Sonic emitter localization using CUDA.

This repository includes a C++ library sonic_cuda_lib and a MATLAB mex function sonic_cuda.

sonic-cuda only supports Windows platform and Nvidia GPUs.
To build the mex function, sonic-cuda requires MATLAB 2024a or higher.

## Build

sonic-cuda uses CMake build system.

The build script is as below.

```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

This will build both the library and the mex function.

## Test

The build script also builds the test.

Please download the test data from Releases. Unzip and put the `.bin` files in `test_data` directory.

To run the end-to-end test for sonic_cuda_lib,
find the test_sonic executable in the build directory, and run

```powershell
test_sonic.exe test_data output_results 1024 1024 1 100.0 15
```

For more information, please read the comments in `test_sonic.cpp`.