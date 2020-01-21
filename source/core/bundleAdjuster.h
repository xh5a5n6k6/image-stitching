#pragma once

#include <opencv2/opencv.hpp>

namespace sis {

/*
    BundleAdjuster: it is used for solving panorama image drifting problem.                
*/
class BundleAdjuster {
public:
    virtual void adjust(
        const cv::Mat& panorama,
        cv::Mat* const out_adjustedPanorama) const = 0;
};

} // namespace sis