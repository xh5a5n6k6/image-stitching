#include "featureDetector/harrisFeatureDetector.h"

#include "mathUtils.h"

#include <iostream>

namespace sis {

HarrisFeatureDetector::HarrisFeatureDetector() :
    HarrisFeatureDetector(0.04f, 4000.0f) {
}

HarrisFeatureDetector::HarrisFeatureDetector(const float k, const float threshold) :
    _k(k),
    _threshold(threshold) {
}

void HarrisFeatureDetector::_detectImpl(
    const std::vector<cv::Mat>&                images,
    std::vector<std::vector<cv::Point>>* const out_featurePositions) const {

    std::cout << "# Begin to detect features using Harris corner detector"
              << std::endl
              << "\r    Progress of feature detection: 0/" << images.size();
       
    const int numImages = static_cast<int>(images.size());
    out_featurePositions->reserve(numImages);

    /*
        For each image, follow 6 steps of Harris Corner Detection
        to generate features 
        (all operations are done in gray scale image)
    */
    int numAllFeatures = 0;
    for (int n = 0; n < numImages; ++n) {
        /*
            Change image to gray scale
        */
        cv::Mat image;
        cv::cvtColor(images[n], image, CV_BGR2GRAY);
        image.convertTo(image, CV_32FC1);

        /*
            Step 1
            Compute x and y derivatives of smooth image
        */
        cv::Mat translationMatrix;
        cv::Mat smoothImage;
        cv::GaussianBlur(image, smoothImage, cv::Size(5, 5), 3);

        cv::Mat rightImage;
        mathUtils::getTranslationMatrix(-1, 0, &translationMatrix);
        cv::warpAffine(smoothImage, rightImage, translationMatrix, smoothImage.size());

        cv::Mat leftImage; 
        mathUtils::getTranslationMatrix(1, 0, &translationMatrix);
        cv::warpAffine(smoothImage, leftImage, translationMatrix, smoothImage.size());

        cv::Mat upImage;
        mathUtils::getTranslationMatrix(0, 1, &translationMatrix);
        cv::warpAffine(smoothImage, upImage, translationMatrix, smoothImage.size());

        cv::Mat downImage;
        mathUtils::getTranslationMatrix(0, -1, &translationMatrix);
        cv::warpAffine(smoothImage, downImage, translationMatrix, smoothImage.size());
        
        const cv::Mat Ix = (rightImage - leftImage) * 0.5f;
        const cv::Mat Iy = (downImage - upImage) * 0.5f;

        /*
            Step 2
            Compute products of derivatives at every pixel
        */
        const cv::Mat Ix2 = Ix.mul(Ix); 
        const cv::Mat Iy2 = Iy.mul(Iy);
        const cv::Mat Ixy = Ix.mul(Iy);;

        /*
            Step 3
            Compute the sums of the products of derivatives at each pixel
        */
        cv::Mat Sx2;
        cv::Mat Sy2;
        cv::Mat Sxy;
        cv::GaussianBlur(Ix2, Sx2, cv::Size(5, 5), 3);
        cv::GaussianBlur(Iy2, Sy2, cv::Size(5, 5), 3);
        cv::GaussianBlur(Ixy, Sxy, cv::Size(5, 5), 3);

        /*
            Step 4
            Define the matrix at each pixel

            M = [Sx2 Sxy]
                [Sxy Sy2]

            As we have calculated Sx2, Sy2, Sxy at previous step,
            just skip this step and go forward to step 5
        */

        /*
            Step 5
            Compute the response of the detector at each pixel

            R = det(M) - k * (trace(M))^2
        */
        const cv::Mat R = 
            (Sx2.mul(Sy2) - Sxy.mul(Sxy)) - _k * (Sx2 + Sy2).mul(Sx2 + Sy2);
 
        /*
            Step 6-1
            Threshold on response R

            HACK here,
            because SIFT algorithm would use local 16x16 window
            to calculate its feature discriptor, we just neglect
            pixels around borders.
        */
        const int siftHack = 8;
        cv::Mat featureIndexMat = cv::Mat::zeros(R.size(), CV_8UC1);
        for (int iy = siftHack; iy < R.rows - siftHack; ++iy) {
            for (int ix = siftHack; ix < R.cols - siftHack; ++ix) {
                if (R.at<float>(iy, ix) > _threshold) {
                    featureIndexMat.at<uchar>(iy, ix) = 1;
                }
            }
        }

        /*
            Step 6-2
            Compute non-maximum suppresion

            Only retain local maximum
        */
        std::vector<cv::Mat> shiftImages;
        shiftImages.reserve(8);
        int dx[8] = { 1, 1, 0, -1, -1, -1,  0,  1 };
        int dy[8] = { 0, 1, 1,  1,  0, -1, -1, -1 };
        for (int shift = 0; shift < 8; ++shift) {
            cv::Mat tmpImage;
            cv::Mat translationMatrix;
            mathUtils::getTranslationMatrix(dx[shift], dy[shift], &translationMatrix);

            cv::warpAffine(R, tmpImage, translationMatrix, R.size());
            shiftImages.push_back(tmpImage);
        }

        int sumFeatures = 0;
        std::vector<cv::Point> featurePos;
        for (int iy = 0; iy < R.rows; ++iy) {
            for (int ix = 0; ix < R.cols; ++ix) {
                if (featureIndexMat.at<uchar>(iy, ix) == 1 &&
                    _isLocalMaximum(ix, iy, R, shiftImages)) {

                    featurePos.push_back(cv::Point(ix, iy));
                    ++sumFeatures;
                }
            }
        }

        out_featurePositions->push_back(featurePos);

        numAllFeatures += sumFeatures;

        std::cout << "\r    Progress of feature detection: " << (n + 1) << "/" << numImages; 
    }

    std::cout << std::endl
              << "# Finish feature detection of all images, avg: "
              << (numAllFeatures / static_cast<float>(numImages)) << " features"
              << std::endl;
}

bool HarrisFeatureDetector::_isLocalMaximum(
    const int                   x,
    const int                   y,
    const cv::Mat&              image,
    const std::vector<cv::Mat>& shiftImages) const {

    /*
        Check all values of 8 neighbors
    */
    const float value = image.at<float>(y, x);
    for (auto& shiftImg : shiftImages) {
        if (value <= shiftImg.at<float>(y, x)) {
            return false;
        }
    }

    return true;
}

} // namespace sis