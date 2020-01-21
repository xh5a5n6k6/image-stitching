#pragma once

/*
    It stores some math utilities including
    constants and functions.
*/

#include <opencv2/opencv.hpp>
#include <random>

namespace sis::mathUtils {

inline constexpr int BIN_NUMBER
    = 18;

inline constexpr float PI
    = 3.14159265358979323846f;

inline void getTranslationMatrix(int            tx, 
                                 int            ty,
                                 cv::Mat* const out_mat) {

    cv::Mat mat = cv::Mat::zeros(cv::Size(3, 2), CV_32FC1);
    mat.at<float>(0, 0) = 1.0f;
    mat.at<float>(0, 2) = static_cast<float>(tx);
    mat.at<float>(1, 1) = 1.0f;
    mat.at<float>(1, 2) = static_cast<float>(ty);

    *out_mat = mat;
}


static std::random_device rd;

inline int nextInt(const int min, const int max) {
    std::default_random_engine generator = std::default_random_engine(rd());

    std::uniform_int_distribution<int> distribution(min, max - 1);

    return distribution(generator);
}

} // namespace sis::mathUtils