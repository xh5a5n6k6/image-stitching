#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace sis {

/*
    FeatureDescriptor is used for calculating feature descriptor,
    and dimension depends on which algorithm is used.

    out_featureDescriptors: It stores all descriptors (using float vector) 
                            of all features of all images
*/
class FeatureDescriptor {
public:
    virtual void calculate(
        const std::vector<cv::Mat>&                         images, 
        const std::vector<std::vector<cv::Point>>&          featurePositions,
        std::vector<std::vector<std::vector<float>>>* const out_featureDescriptors) const = 0;
};

} // namespace sis