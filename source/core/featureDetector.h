#pragma once

#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

namespace sis {

class FeatureDetector {
public:
    virtual std::vector<std::vector<cv::Point> > solve(std::vector<cv::Mat> images) = 0;
};

} // namespace sis