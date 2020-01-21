#pragma once

#include "core/imageBlender.h"

namespace sis {

/*
    LinearAlphaImageBlender: pixels in the overlapping region are 
                             blending with values of two images using 
                             x-direction linear weighted interpolation.
*/
class LinearAlphaImageBlender : public ImageBlender {
public:
    LinearAlphaImageBlender();

private:
    void _blendImpl(
        const std::vector<cv::Mat>&   images,
        const std::vector<cv::Point>& imageAlignments,
        const std::vector<cv::Mat>&   warpImageIndices,
        cv::Mat* const                out_blendImage) const override;
};

} // namespace sis