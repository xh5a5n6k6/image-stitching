#pragma once

#include "core/featureDescriptor.h"

namespace sis {

class SIFTDescriptor : public FeatureDescriptor {
public:
    SIFTDescriptor();

    std::vector<std::vector<std::vector<float> > > 
    solve(
        std::vector<cv::Mat> images, 
        std::vector<std::vector<cv::Point> > featurePositions) override;
};

} // namespace sis