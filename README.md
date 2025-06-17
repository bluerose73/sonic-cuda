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