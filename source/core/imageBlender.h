#pragma once

#include "config.h"

#include <cstdio>
#include <opencv2/opencv.hpp>
#include <vector>

namespace sis {

/*
    ImageBlender is used for image blending between image pairs.

    out_blendImage: it stores stitched image result, image pair
                    overlapping regions use blend function to blend.
                    
*/
class ImageBlender {
public:
    void blend(
        const std::vector<cv::Mat>&   images, 
        const std::vector<cv::Point>& imageAlignments, 
        const std::vector<cv::Mat>&   warpImageIndices,
        cv::Mat* const                out_blendImage) const;

private:
    virtual void _blendImpl(
        const std::vector<cv::Mat>&   images,
        const std::vector<cv::Point>& imageAlignments,
        const std::vector<cv::Mat>&   warpImageIndices,
        cv::Mat* const                out_blendImage) const = 0;

    void _writeImage(const cv::Mat& blendImage) const;
};

// header implementation

inline void ImageBlender::blend(
    const std::vector<cv::Mat>&   images,
    const std::vector<cv::Point>& imageAlignments,
    const std::vector<cv::Mat>&   warpImageIndices,
    cv::Mat* const                out_blendImage) const {

    _blendImpl(images, imageAlignments, warpImageIndices, out_blendImage);

#ifdef DRAW_BLEND_IMAGES
    _writeImage(*out_blendImage);

#endif
}

inline void ImageBlender::_writeImage(const cv::Mat& blendImage) const {
    cv::imwrite("./result/blend/blendImage_result.png", blendImage);
}

} // namespace sis