#pragma once

#include "core/featureDetector.h"

namespace sis {

class HarrisCornerDetector : public FeatureDetector {
public:
    HarrisCornerDetector();

    std::vector<std::vector<cv::Point> > solve(std::vector<cv::Mat> images) override;

private:
    float _k, _threshold;
};

} // namespace sis