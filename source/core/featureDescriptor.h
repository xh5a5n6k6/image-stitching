#pragma once

#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

namespace sis {

class FeatureDescriptor {
public:
    virtual std::vector<std::vector<std::vector<float> > > 
    solve(
        std::vector<cv::Mat> images, 
        std::vector<std::vector<cv::Point> > featurePositions) = 0;
};

} // namespace sis