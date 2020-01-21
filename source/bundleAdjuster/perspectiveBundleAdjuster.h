#pragma once

#include "core/bundleAdjuster.h"

namespace sis {

/*
    PerspectiveBundleAdjuster is simply used OpenCV's warpPerspective
    function to do bundle adjustment.
*/
class PerspectiveBundleAdjuster : public BundleAdjuster {
public:
    PerspectiveBundleAdjuster();

    void adjust(
        const cv::Mat& panorama,
        cv::Mat* const out_adjustedPanorama) const override;
};

} // namespace sis