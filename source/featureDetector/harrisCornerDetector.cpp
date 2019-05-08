#include "featureDetector/harrisCornerDetector.h"

#include "shared.h"

namespace sis {

HarrisCornerDetector::HarrisCornerDetector() :
    _k(0.04f), _threshold(4000.0f) {
}

std::vector<std::vector<cv::Point> > 
HarrisCornerDetector::solve(std::vector<cv::Mat> images) {
    std::vector<cv::Mat> imagesCopy;
    int num = images.size();
    for (auto img : images) {
        imagesCopy.push_back(img.clone());
    }

    /*
        For each image, follow 6 steps of Harris Corner Detection
        to generate features 
        (all operations are done in gray scale image)
    */
    std::vector<std::vector<cv::Point> > featurePositions;
    for (int n = 0; n < num; n++) {
        /*
            Change image to gray scale
        */
        cv::Mat image;
        cv::cvtColor(images.at(n), image, CV_BGR2GRAY);
        image.convertTo(image, CV_32FC1);

        /*
            Step 1
            Compute x and y derivatives of smooth image
        */
        cv::Mat smoothImage, Ix, Iy;
        cv::GaussianBlur(image, smoothImage, cv::Size(5, 5), 3);

        cv::Mat rightImage, leftImage, upImage, downImage;
        cv::warpAffine(smoothImage, rightImage, GetTranslationMatrix(-1, 0), smoothImage.size());
        cv::warpAffine(smoothImage, leftImage, GetTranslationMatrix(1, 0), smoothImage.size());
        cv::warpAffine(smoothImage, downImage, GetTranslationMatrix(0, -1), smoothImage.size());
        cv::warpAffine(smoothImage, upImage, GetTranslationMatrix(0, 1), smoothImage.size());
        Ix = (rightImage - leftImage) / 2.0f;
        Iy = (downImage - upImage) / 2.0f;

        /*
            Step 2
            Compute products of derivatives at every pixel
        */
        cv::Mat Ix2, Iy2, Ixy;
        Ix2 = Ix.mul(Ix);
        Iy2 = Iy.mul(Iy);
        Ixy = Ix.mul(Iy);

        /*
            Step 3
            Compute the sums of the products of derivatives at each pixel
        */
        cv::Mat Sx2, Sy2, Sxy;
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
        cv::Mat R;
        R = (Sx2.mul(Sy2) - Sxy.mul(Sxy)) - _k * (Sx2 + Sy2).mul(Sx2 + Sy2);
 
        /*
            Step 6-1
            Threshold on response R
        */
        int siftHack = 8;
        cv::Mat featureIndexMat = cv::Mat::zeros(R.size(), CV_8UC1);
        for (int j = siftHack; j < R.rows-siftHack; j++) {
            for (int i = siftHack; i < R.cols-siftHack; i++) {
                if (R.at<float>(j, i) > _threshold) {
                    featureIndexMat.at<uchar>(j, i) = 1;
                }
            }
        }

        /*
            Step 6-2
            Compute non-maximum suppresion

            Only retain local maximum
        */
        std::vector<cv::Mat> shiftImage;
        int dx[8] = { 1, 1, 0, -1, -1, -1,  0,  1 };
        int dy[8] = { 0, 1, 1,  1,  0, -1, -1, -1 };
        for (int shift = 0; shift < 8; shift++) {
            cv::Mat tmpImage;
            cv::Mat translationMatrix = GetTranslationMatrix(dx[shift], dy[shift]);
            cv::warpAffine(R, tmpImage, translationMatrix, R.size());
            shiftImage.push_back(tmpImage);
        }

        int sum = 0;
        std::vector<cv::Point> featurePos;
        for (int j = 0; j < R.rows; j++) {
            for (int i = 0; i < R.cols; i++) {
                if (featureIndexMat.at<uchar>(j, i) == 1) {
                    /*
                        Check all values of 8 neighbors
                    */
                    int maxFlag = 1;
                    for (auto shiftImg : shiftImage) {
                        if (R.at<float>(j, i) <= shiftImg.at<float>(j, i)) {
                            maxFlag = 0;
                            break;
                        }
                    }
                    if (maxFlag == 1) {
                        featurePos.push_back(cv::Point(i, j));
                        cv::circle(imagesCopy.at(n), cv::Point(i, j), 2, cv::Scalar(0, 0, 255), CV_FILLED);
                        sum += 1;
                    }
                }
            }
        }
        featurePositions.push_back(featurePos);

        fprintf(stderr, "\rFinish harris corner detection of image %d", n + 1);
    }
    fprintf(stderr, "\n");

    return featurePositions;
}

} // namespace sis