#pragma once

#include "core/featureDescriptor.h"

namespace sis {

/*
    SIFT feature descriptor calculates main orientation 
    for each feature, and then use local orientational 
    window to calculate descriptor to make sure its 
    sensitivity to affine transform.
    (including rotation, translation, etc)
*/
class SiftFeatureDescriptor : public FeatureDescriptor {
public:
    SiftFeatureDescriptor();

    void calculate(
        const std::vector<cv::Mat>&                         images,
        const std::vector<std::vector<cv::Point>>&          featurePositions,
        std::vector<std::vector<std::vector<float>>>* const out_featureDescriptors) const override;
};

} // namespace sis