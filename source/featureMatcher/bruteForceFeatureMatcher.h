#pragma once

#include "core/featureMatcher.h"

namespace sis {

class BruteForceFeatureMatcher : public FeatureMatcher {
public:
    BruteForceFeatureMatcher();
    BruteForceFeatureMatcher(const float threshold);

private:
    void _matchImpl(
        const std::vector<cv::Mat>&                          images,
        const std::vector<std::vector<cv::Point>>&           featurePositions,
        const std::vector<std::vector<std::vector<float>>>&  featureDescriptors,
        std::vector<std::vector<std::pair<int, int>>>* const out_featureMatches) const override;

    float _threshold;
};

} // namespace sis