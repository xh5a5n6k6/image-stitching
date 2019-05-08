#pragma once

#include <opencv2/opencv.hpp>

#define BIN_NUMBER 18

namespace sis {

inline constexpr float PI = 3.14159265f;

inline cv::Mat GetTranslationMatrix(int tx, int ty) {
    cv::Mat m = cv::Mat::zeros(cv::Size(3, 2), CV_32FC1);
    m.at<float>(0, 0) = 1.0f;
    m.at<float>(0, 2) = float(tx);
    m.at<float>(1, 1) = 1.0f;
    m.at<float>(1, 2) = float(ty);

    return m;
}

} // namespace sis