#pragma once

#include "config.h"

#include <cstdio>
#include <opencv2/opencv.hpp>
#include <vector>

namespace sis {

/*
    FeatureMatcher is used for feature matching between image pairs.

    out_featureMatchings: It stores all feature index matching between
                          image pairs.

                          take image pair 1-2 for example, it will store
                          all feature index matching in std::pair vector.
                          each std::pair records corresponding feature
                          indices in this image pair.

                          Ex. std::pair<int, int>(3, 10)
                              it means image2's feature 3 matches image1's feature 10
*/
class FeatureMatcher {
public:
    void match(
        const std::vector<cv::Mat>&                          images,
        const std::vector<std::vector<cv::Point>>&           featurePositions,
        const std::vector<std::vector<std::vector<float>>>&  featureDescriptors,
        std::vector<std::vector<std::pair<int, int>>>* const out_featureMatchings) const;

private:
    virtual void _matchImpl(
        const std::vector<cv::Mat>&                          images,
        const std::vector<std::vector<cv::Point>>&           featurePositions,
        const std::vector<std::vector<std::vector<float>>>&  featureDescriptors,
        std::vector<std::vector<std::pair<int, int>>>* const out_featureMatchings) const = 0;

    void _writeImages(
        const std::vector<cv::Mat>&                          images,
        const std::vector<std::vector<cv::Point>>&           featurePositions,
        const std::vector<std::vector<std::pair<int, int>>>& featureMatchings) const;
};

// header implementation

inline void FeatureMatcher::match(
    const std::vector<cv::Mat>&                          images,
    const std::vector<std::vector<cv::Point>>&           featurePositions,
    const std::vector<std::vector<std::vector<float>>>&  featureDescriptors,
    std::vector<std::vector<std::pair<int, int>>>* const out_featureMatchings) const {

    _matchImpl(images, featurePositions, featureDescriptors, out_featureMatchings);

#ifdef DRAW_FEATURE_MATCHING_IMAGES
    _writeImages(images, featurePositions, *out_featureMatchings);

#endif
}

inline void FeatureMatcher::_writeImages(
    const std::vector<cv::Mat>&                          images,
    const std::vector<std::vector<cv::Point>>&           featurePositions,
    const std::vector<std::vector<std::pair<int, int>>>& featureMatchings) const {

    for (std::size_t n = 0; n < images.size() - 1; ++n) {
        const std::vector<cv::Point>& feaPos1 = featurePositions[n];
        const std::vector<cv::Point>& feaPos2 = featurePositions[n + 1];
        const std::vector<std::pair<int, int>>& matchings = featureMatchings[n];
        const std::size_t numMatchings = matchings.size();

        cv::Mat concateImage;
        cv::hconcat(images[n], images[n + 1], concateImage);

        for (std::size_t i = 0; i < numMatchings; ++i) {
            const std::pair<int, int>& matching = matchings[i];
            const cv::Point& point1 = feaPos1[matching.second];
            const cv::Point& point2 = feaPos2[matching.first];

            cv::line(concateImage, point1, point2 + cv::Point(images[n].cols, 0), cv::Scalar(0, 255, 0), 2);
            cv::circle(concateImage, point1, 1, cv::Scalar(0, 0, 255), cv::FILLED);
            cv::circle(concateImage, point2 + cv::Point(images[n].cols, 0), 1, cv::Scalar(0, 0, 255), cv::FILLED);
        }

        char filename[100];
        sprintf(filename, "./result/matching/matching%02zd.png", n);
        cv::imwrite(filename, concateImage);
    }
}

} // namespace sis