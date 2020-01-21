#pragma once

#include "core/featureDetector.h"

namespace sis {

class HarrisFeatureDetector : public FeatureDetector {
public:
    HarrisFeatureDetector();
    HarrisFeatureDetector(const float k, const float threshold);

private:
    void _detectImpl(
        const std::vector<cv::Mat>&                images,
        std::vector<std::vector<cv::Point>>* const out_featurePositions) const override;

    bool _isLocalMaximum(
        const int                   x,
        const int                   y,
        const cv::Mat&              image,
        const std::vector<cv::Mat>& shiftImages) const;

    float _k;
    float _threshold;
};

} // namespace sis