#pragma once

#include "core/imageWarpper.h"

namespace sis {

class CylindricalImageWarpper : public ImageWarpper {
public:
    CylindricalImageWarpper();

private:
    void _warpImpl(
        const std::vector<cv::Mat>& images,
        const std::vector<float>&   focalLengths,
        std::vector<cv::Mat>* const out_warpImages,
        std::vector<cv::Mat>* const out_warpImageIndices) const override;

    bool _isOutOfBound(const float x,
                       const float y,
                       const int   width,
                       const int   height) const;
};

} // namespace sis