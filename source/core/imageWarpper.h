#pragma once

#include "config.h"

#include <cstdio>
#include <opencv2/opencv.hpp>
#include <vector>

namespace sis {

/*
    ImageWarpper is used for image warpping.

    It's an interface which only defines warp function,
    and user can control if it needs to output warpped images
    in the config.h file.

    out_warpImageIndices: It records if a pixel succeeds in
                          inverse warpping interpolation.
                          (1 for success, 0 for failure, and 
                           it would be used in image blending.)
*/
class ImageWarpper {
public:
    void warp(
        const std::vector<cv::Mat>& images,
        const std::vector<float>&   focalLengths,
        std::vector<cv::Mat>* const out_warpImages,
        std::vector<cv::Mat>* const out_warpImageIndices) const;

private:
    virtual void _warpImpl(
        const std::vector<cv::Mat>& images,
        const std::vector<float>&   focalLengths, 
        std::vector<cv::Mat>* const out_warpImages,
        std::vector<cv::Mat>* const out_warpImageIndices) const = 0;

    void _writeImages(const std::vector<cv::Mat>& images) const;
};

// header implementation

inline void ImageWarpper::warp(
    const std::vector<cv::Mat>& images,
    const std::vector<float>&   focalLengths,
    std::vector<cv::Mat>* const out_warpImages,
    std::vector<cv::Mat>* const out_warpImageIndices) const {

    _warpImpl(images, focalLengths, out_warpImages, out_warpImageIndices);

#ifdef DRAW_WARP_IMAGES
    _writeImages(*out_warpImages);

#endif 
}

inline void ImageWarpper::_writeImages(const std::vector<cv::Mat>& images) const {
    for (std::size_t i = 0; i < images.size(); ++i) {
        char filename[100];
        sprintf(filename, "./result/warp/warp%02zd.png", i);
        cv::imwrite(filename, images[i]);
    }
}

} // namespace sis