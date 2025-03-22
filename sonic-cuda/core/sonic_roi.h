#ifndef SONIC_CUDA_CORE_SONIC_ROI_H
#define SONIC_CUDA_CORE_SONIC_ROI_H

#define ROI_WIDTH 7

/*
* sonic_roi - Region of interest (ROI) extraction
* 
* Arguments:
* d_data - 3D array of image frames (frames x height x width)
* height - height of each frame
* width - width of each frame
* frames - number of frames
* d_peak_x - x locations of the emitter
* d_peak_y - y locations of the emitter
* d_peak_f - frame id of the emitter
* n_locs - number of detected peaks
* d_roi - output ROI (n_locs x 7 x 7)
*/
void sonic_roi(const float* d_data, int height, int width, int frames,
    const int *d_peak_x, const int *d_peak_y, const int *d_peak_f, int n_locs,
    float* d_roi
);


#endif // SONIC_CUDA_CORE_SONIC_ROI_H