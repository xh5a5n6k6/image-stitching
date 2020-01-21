#pragma once

#include "core/imageMatcher.h"

namespace sis {

class RansacImageMatcher : public ImageMatcher {
public:
    RansacImageMatcher();

    void match(
        const std::vector<cv::Mat>&                          images,
        const std::vector<std::vector<cv::Point>>&           featurePositions,
        const std::vector<std::vector<std::pair<int, int>>>& featureMatchings,
        std::vector<cv::Point>* const                        out_imageAlignments) const override;
};

} // namespace sis