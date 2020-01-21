#include "imageWarpper/cylindricalImageWarpper.h"

#include <iostream>

namespace sis {

CylindricalImageWarpper::CylindricalImageWarpper() = default;

void CylindricalImageWarpper::_warpImpl(
    const std::vector<cv::Mat>& images,
    const std::vector<float>&   focalLengths,
    std::vector<cv::Mat>* const out_warpImages,
    std::vector<cv::Mat>* const out_warpImageIndices) const {

    std::cout << "# Begin to warp images using cylindrical projection"
              << std::endl
              << "\r    Progress of cylindrical warpping: 0/" << images.size();

    const std::size_t numImages = images.size();
    out_warpImages->reserve(numImages);
    out_warpImageIndices->reserve(numImages);

    for (std::size_t n = 0; n < numImages; ++n) {
        cv::Mat image;
        images[n].convertTo(image, CV_32FC3);

        const int width   = image.cols;
        const int height  = image.rows;
        const int xCenter = width / 2;
        const int yCenter = height / 2;

        const float f    = focalLengths[n];
        const float invS = 1.0f / f;

        /*
            HACK here,
            make warpped image without x-direction black edges,
            it is convenient to image blending

            xBound    : x-direction bounding coordinate with origin
                        at the image center

            xWarpBound: warpped cylindrical coordinate of xBound

            offset    : because we need to remove x-direction black edges,
                        we let warpped image's size is a little bit smaller
                        than original one. To make our result correct, we 
                        need to add offset first when calculating inverse warpping.

                              xBound
            |---------------------------------------|
            |------------------------------|--------|
                       xWarpBound            offset
        */
        const int xBound     = (width - 1) - xCenter;
        const int xWarpBound = static_cast<int>(f * std::atan2(static_cast<float>(xBound), f));
        const int offset     = xBound - xWarpBound;

        cv::Mat warpImage      = cv::Mat::zeros(cv::Size(2 * xWarpBound, height), image.type());
        cv::Mat warpImageIndex = cv::Mat::zeros(cv::Size(2 * xWarpBound, height), CV_32FC1);

        /*
            cylindrical projection transform

            x' = s * arctan(x / f)
            y' = s * (y / sqrt(x^2 + f^2))

            (x, y)  : original image coordinate
            (x', y'): warpped coordinate of (x, y)
            use s = f here, it gives less distortion

            because using inverse warping interpolation,
            we need to calculate inverse transform to
            know warpped image's original coordinate

            x = tan(x' / s) * f
            y = y' / s * sqrt(x^2 + f^2)
        */
        for (int iy = 0; iy < warpImage.rows; ++iy) {
            for (int ix = 0; ix < warpImage.cols; ++ix) {
                /*
                    It needs to make image center be the origin,
                    so we need to substract center first, and why
                    x-direction needs to add extra offset is 
                    explained recently.
                */
                const float xCylindrical = static_cast<float>(ix + offset - xCenter);
                const float yCylindrical = static_cast<float>(iy - yCenter);

                /*
                    Because y will use x to calculate, x needs to be
                    calculated before y
                */
                float xOriginal = f * std::tan(xCylindrical * invS);
                float yOriginal = std::sqrt(xOriginal * xOriginal + f * f) * invS * yCylindrical;

                /*
                    Add center back
                */
                xOriginal += xCenter;
                yOriginal += yCenter;

                if (_isOutOfBound(xOriginal, yOriginal, width, height)) {
                    continue;
                }

                /*
                    As we use bi-linear interpolation, we need to
                    calculate its inverse corresponding four border coordinates,
                    and then use these coordinates' values to calculate the result
                */
                const int xFloor = static_cast<int>(std::floor(xOriginal));
                const int xCeil  = static_cast<int>(std::ceil(xOriginal));
                const int yFloor = static_cast<int>(std::floor(yOriginal));
                const int yCeil  = static_cast<int>(std::ceil(yOriginal));

                const float xt = xCeil - xOriginal;
                const float yt = yCeil - yOriginal;

                const cv::Vec3f t1 = image.at<cv::Vec3f>(yFloor, xFloor) * xt + 
                                     image.at<cv::Vec3f>(yFloor, xCeil) * (1.0f - xt);
                const cv::Vec3f t2 = image.at<cv::Vec3f>(yCeil, xFloor) * xt +
                                     image.at<cv::Vec3f>(yCeil, xCeil) * (1.0f - xt);

                warpImage.at<cv::Vec3f>(iy, ix) = t1 * yt + t2 * (1.0f - yt);
                warpImageIndex.at<float>(iy, ix) = 1.0f;
            }
        }
        warpImage.convertTo(warpImage, CV_8UC3);

        out_warpImages->push_back(warpImage);
        out_warpImageIndices->push_back(warpImageIndex);

        std::cout << "\r    Progress of cylindrical warpping: " << (n + 1) << "/" << numImages;
    }

    std::cout << std::endl
              << "# Finish image warpping"
              << std::endl;
}

bool CylindricalImageWarpper::_isOutOfBound(const float x,
                                            const float y,
                                            const int   width,
                                            const int   height) const {

    return x < 0.0f ||
           x > static_cast<float>(width - 1) ||
           y < 0.0f ||
           y > static_cast<float>(height - 1);
}

} // namespace sis