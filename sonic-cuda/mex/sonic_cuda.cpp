#include <mex.h>
#include <matrix.h>
#include <sonic-cuda/core/sonic.h>
#include <vector>

/*
* sonic_cuda
* 
* This function is the entry point for the MATLAB MEX function sonic_cuda.
* sonic_cuda performs emitter localization using phase.
* sonic_cuda uses NVIDIA CUDA to accelerate the computation.
* An NVIDIA GPU is required.
*
* Usage:
* [loc_x, loc_y, loc_f] = sonic_cuda(img, background, threshold, ignore_border_px)
*
* Arguments:
* img (single)                - 3D array of background-removed image frames (frames x height x width)
* background (single)         - 3D array of background image (frames x height x width)
* threshold (double)          - threshold for emitter detection
* ignore_border_px (double)   - number of pixels to ignore at the border of the image. Must be an non-negative integer.
*
* Returns:
* loc_x (single) - x location of the emitter
* loc_y (single) - y location of the emitter
* loc_f (int)    - frame id of the emitter
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check number of inputs
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("sonic_cuda:invalidNumInputs", "Four inputs required: img, background, threshold, ignore_border_px");
    }

    // Check number of outputs
    if (nlhs != 3) {
        mexErrMsgIdAndTxt("sonic_cuda:invalidNumOutputs", "Three outputs required: loc_x, loc_y, loc_f");
    }

    // Validate input types
    if (!mxIsSingle(prhs[0])) {
        mexErrMsgIdAndTxt("sonic_cuda:invalidInputType", "Input img must be of type single");
    }

    if (!mxIsSingle(prhs[1])) {
        mexErrMsgIdAndTxt("sonic_cuda:invalidInputType", "Input background must be of type single");
    }

    if (!mxIsDouble(prhs[2])) {
        mexErrMsgIdAndTxt("sonic_cuda:invalidInputType", "Input threshold must be of type double");
    }

    if (!mxIsDouble(prhs[3])) {
        mexErrMsgIdAndTxt("sonic_cuda:invalidInputType", "Input ignore_border_px must be of type double");
    }

    // Get dimensions of img
    int height = 0, width = 0, frames = 0;
    const mwSize num_dims = mxGetNumberOfDimensions(prhs[0]);
    const mwSize* dims = mxGetDimensions(prhs[0]);
    if (num_dims == 2) {
        mexWarnMsgIdAndTxt("sonic_cuda:invalidInputSize", "Input img is 2D, assuming 1 frame");
        height = dims[0];
        width = dims[1];
        frames = 1;
    } else if (num_dims == 3) {
        height = dims[0];
        width = dims[1];
        frames = dims[2];
    } else {
        mexErrMsgIdAndTxt("sonic_cuda:invalidInputSize", "Input img must be 3D");
    }

    // Get dimensions of background
    const mwSize num_dims_bg = mxGetNumberOfDimensions(prhs[1]);
    const mwSize* dims_bg = mxGetDimensions(prhs[1]);
    int bg_frames = 0;
    if (num_dims_bg == 2) {
        mexWarnMsgIdAndTxt("sonic_cuda:invalidInputSize", "Input background is 2D, assuming 1 frame");
        bg_frames = 1;
    } else if (num_dims_bg == 3) {
        bg_frames = dims_bg[2];
    } else {
        mexErrMsgIdAndTxt("sonic_cuda:invalidInputSize", "Input background must be 3D");
    }
    if (dims_bg[0] != height || dims_bg[1] != width || bg_frames != frames) {
        mexErrMsgIdAndTxt("sonic_cuda:invalidInputSize", "Input img and background must have the same dimensions");
    }

    // Get input data
    const float* img = mxGetSingles(prhs[0]);
    const float* background = mxGetSingles(prhs[1]);

    if (mxGetNumberOfElements(prhs[2]) != 1) {
        mexErrMsgIdAndTxt("sonic_cuda:invalidInputSize", "Input threshold must be scalar");
    }
    if (mxGetNumberOfElements(prhs[3]) != 1) {
        mexErrMsgIdAndTxt("sonic_cuda:invalidInputSize", "Input ignore_border_px must be scalar");
    }
    const float threshold = mxGetDoubles(prhs[2])[0];
    const int ignore_border_px = mxGetDoubles(prhs[3])[0];

    // Validate ignore_border_px
    if (ignore_border_px < 0) {
        mexErrMsgIdAndTxt("sonic_cuda:invalidInputValue", "Input ignore_border_px must be non-negative");
    }

    // Print debug information
    // mexPrintf("height: %d, width: %d, frames: %d\n", height, width, frames);
    // mexPrintf("threshold: %f, ignore_border_px: %d\n", threshold, ignore_border_px);

    // Allocate memory for output locations
    int max_locs = (height * width + 48) / 49 * frames;
    std::vector<float> loc_x(max_locs);
    std::vector<float> loc_y(max_locs);
    std::vector<int> loc_f(max_locs);
    int n_locs = 0;

    // Call the core sonic function
    int error_code = 0;
    error_code = sonic(img, height, width, frames, background, threshold,
        ignore_border_px, loc_x.data(), loc_y.data(), loc_f.data(), &n_locs);
    if (error_code != 0) {
        mexErrMsgIdAndTxt("sonic_cuda:executionError", "Error in sonic function: %d", error_code);
    }
    
    // Create output arrays
    plhs[0] = mxCreateNumericMatrix(n_locs, 1, mxSINGLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(n_locs, 1, mxSINGLE_CLASS, mxREAL);
    plhs[2] = mxCreateNumericMatrix(n_locs, 1, mxINT32_CLASS, mxREAL);
    float* loc_x_out = mxGetSingles(plhs[0]);
    float* loc_y_out = mxGetSingles(plhs[1]);
    int* loc_f_out = mxGetInt32s(plhs[2]);

    // Copy output data
    // Note: loc_x and loc_y are swapped because MATLAB uses column-major order
    memcpy(loc_x_out, loc_y.data(), n_locs * sizeof(float));
    memcpy(loc_y_out, loc_x.data(), n_locs * sizeof(float));
    memcpy(loc_f_out, loc_f.data(), n_locs * sizeof(int));

    // Print number of locations found
    // mexPrintf("n_locs: %d\n", n_locs);
}