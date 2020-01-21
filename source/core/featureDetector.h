#pragma once

#include "config.h"

#include <cstdio>
#include <opencv2/opencv.hpp>
#include <vector>

namespace sis {

/*
    FeatureDetector is used for feature detection.

    out_featurePositions: It stores all feature positions (x, y) of
                          input images
*/
class FeatureDetector {
public:
    void detect(
        const std::vector<cv::Mat>&                images,
        std::vector<std::vector<cv::Point>>* const out_featurePositions) const;

private:
    virtual void _detectImpl(
        const std::vector<cv::Mat>&                images,
        std::vector<std::vector<cv::Point>>* const out_featurePositions) const = 0;

    void _writeImages(
        const std::vector<cv::Mat>&                images,
        const std::vector<std::vector<cv::Point>>& featurePositions) const;
};

// header implementation

inline void FeatureDetector::detect(
    const std::vector<cv::Mat>&                images,
    std::vector<std::vector<cv::Point>>* const out_featurePositions) const {

    _detectImpl(images, out_featurePositions);

#ifdef DRAW_FEATURE_IMAGES
    _writeImages(images, *out_featurePositions);

#endif
}

inline void FeatureDetector::_writeImages(
    const std::vector<cv::Mat>&                images,
    const std::vector<std::vector<cv::Point>>& featurePositions) const {
    
    for (std::size_t n = 0; n < images.size(); ++n) {
        const std::vector<cv::Point>& features = featurePositions[n];
        const std::size_t numFeatures = features.size();

        cv::Mat tmpImage = images[n].clone();
        for (std::size_t i = 0; i < numFeatures; ++i) {
            cv::circle(tmpImage, cv::Point(features[i].x, features[i].y), 2, cv::Scalar(0, 0, 255), cv::FILLED);
        }

        char filename[100];
        sprintf(filename, "./result/feature/feature%02zd.png", n);
        cv::imwrite(filename, tmpImage);
    }
}

} // namespace sis