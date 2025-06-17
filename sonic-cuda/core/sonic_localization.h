#ifndef SONIC_CUDA_CORE_SONIC_LOCALIZATION_H
#define SONIC_CUDA_CORE_SONIC_LOCALIZATION_H

/*
* sonic_localization - Localize the emitter in the ROI
*
* Arguments:
* d_roi - ROI (n_locs x 7 x 7)
* d_peak_x - x locations of the emitter
* d_peak_y - y locations of the emitter
* n_locs - number of detected peaks
* d_loc_x - output x locations of the emitter
* d_loc_y - output y locations of the emitter
*/
int sonic_localization(const float* d_roi, const int* d_peak_x, const int* d_peak_y, int n_locs,
    float* d_loc_x, float* d_loc_y
);


#endif // SONIC_CUDA_CORE_SONIC_LOCALIZATION_H