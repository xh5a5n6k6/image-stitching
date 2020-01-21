#include "bundleAdjuster/perspectiveBundleAdjuster.h"

namespace sis {

PerspectiveBundleAdjuster::PerspectiveBundleAdjuster() = default;

void PerspectiveBundleAdjuster::adjust(
    const cv::Mat& panorama,
    cv::Mat* const out_adjustedPanorama) const {

    std::cout << "# Begin to do bundle adjustment using perspective warpping"
              << std::endl;

    const int   width        = panorama.cols;
    const int   height       = panorama.rows;
    const float borderWidth  = static_cast<float>(width);
    const float borderHeight = static_cast<float>(height);

    cv::Point2i tmpPoint;
    cv::Point2f border1[4];
    cv::Point2f border2[4];

    border2[0] = cv::Point2f(0.0f, 0.0f);
    border2[1] = cv::Point2f(borderWidth - 1.0f, 0.0f);
    border2[2] = cv::Point2f(0.0f, borderHeight - 1.0f);
    border2[3] = cv::Point2f(borderWidth - 1.0f, borderHeight - 1.0f);

    for (int iy = 0; iy < height; ++iy) {
        if (panorama.at<cv::Vec3b>(iy, 0) != cv::Vec3b(0, 0, 0)) {
            tmpPoint   = cv::Point2i(0, iy);
            border1[0] = cv::Point2f(tmpPoint);
            break;
        }
    }

    for (int iy = 0; iy < height; ++iy) {
        if (panorama.at<cv::Vec3b>(iy, width - 1) != cv::Vec3b(0, 0, 0)) {
            tmpPoint   = cv::Point2i(width - 1, iy);
            border1[1] = cv::Point2f(tmpPoint);
            break;
        }
    }

    for (int iy = height - 1; iy >= 0; --iy) {
        if (panorama.at<cv::Vec3b>(iy, 0) != cv::Vec3b(0, 0, 0)) {
            tmpPoint   = cv::Point2i(0, iy);
            border1[2] = cv::Point2f(tmpPoint);
            break;
        }
    }

    for (int iy = height - 1; iy >= 0; --iy) {
        if (panorama.at<cv::Vec3b>(iy, width - 1) != cv::Vec3b(0, 0, 0)) {
            tmpPoint   = cv::Point2i(width - 1, iy);
            border1[3] = cv::Point2f(tmpPoint);
            break;
        }
    }

    const cv::Mat perspectMatrix = cv::getPerspectiveTransform(border1, border2);
    cv::Mat panoramaAdjusted;
    cv::warpPerspective(panorama, panoramaAdjusted, perspectMatrix, panorama.size());

    *out_adjustedPanorama = panoramaAdjusted;

    std::cout << "# Finish bundle adjustment"
              << std::endl;
}

} // namespace sis