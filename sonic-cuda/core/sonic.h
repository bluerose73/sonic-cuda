#ifndef SONIC_CUDA_CORE_SONIC_H
#define SONIC_CUDA_CORE_SONIC_H

/*
* Sonic - Emitter localization using phase
* 
* Arguments:
* data - 3D array of image frames (frames x height x width)
* height - height of each frame
* width - width of each frame
* frames - number of frames
* background - 3D array of background image (frames x height x width)
* ignore_border_px - number of pixels to ignore at the border of the image
* loc_x - output x location of the emitter
* loc_y - output y location of the emitter
* loc_f - output frame id of the emitter, guaranteed to be an integer
* n_loc - output count of detected emitters
*/
void sonic(const float* data, int height, int width, int frames,
    const float* background, float threshold, int ignore_border_px,
    float* loc_x, float* loc_y, float* loc_f, int* n_loc);

#endif // SONIC_CUDA_CORE_SONIC_H