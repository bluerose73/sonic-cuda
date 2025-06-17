#ifndef SONIC_CUDA_CORE_SONIC_CLEAN_V2_H
#define SONIC_CUDA_CORE_SONIC_CLEAN_V2_H

/*
* sonic_clean_v2 - Image clean and peak detection
*
* This function is similar to sonic_clean, but it is ~30% faster.
* However, it needs more GPU VRAM - 2 * image_stack_size more than sonic_clean.
* 
* Arguments:
* d_data - 3D array of image frames (frames x height x width)
* height - height of each frame
* width - width of each frame
* frames - number of frames
* d_background - 3D array of background image (frame x height x width)
* threshold - threshold value for peak detection
* ignore_border_px - number of pixels to ignore at the border of the image
* d_peak_x - output x location of the emitter
* d_peak_y - output y location of the emitter
* d_peak_f - output frame id of the emitter
* n_locs - output count of detected peaks
*
* Returns:
* error code. 0 if successful, non-zero if an error occurred.
*/
int sonic_clean_v2(const float* d_data, int height, int width, int frames,
    const float* d_background, float threshold, int ignore_border_px,
    int* d_peak_x, int* d_peak_y, int* d_peak_f, int* n_locs
);

#endif // SONIC_CUDA_CORE_SONIC_CLEAN_V2_H