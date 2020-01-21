#include "featureDescriptor/siftFeatureDescriptor.h"

#include "mathUtils.h"

#include <cmath>

namespace sis {

SiftFeatureDescriptor::SiftFeatureDescriptor() = default;

void SiftFeatureDescriptor::calculate(
    const std::vector<cv::Mat>&                         images,
    const std::vector<std::vector<cv::Point>>&          featurePositions,
    std::vector<std::vector<std::vector<float>>>* const out_featureDescriptors) const {

    std::cout << "# Begin to calculate descriptor vector using SIFT feature descriptor"
              << std::endl
              << "\r    Progress of calculating feature descriptors: 0/" << images.size();

    const int numImages = static_cast<int>(images.size());
    out_featureDescriptors->reserve(numImages);

    /*
        Calculate feature descriptor for each feature

        Use 16x16 window to calculate 128-dimentional vectors,
        every 4x4 pixels calculate 8 orientations
        total 8 x 4 x 4 = 128 (128-dimentional vector)


        Here is a 8x8 example, split it to four 4x4 size orientations
        +---------+---------+
        |  \ | /  |  \ | /  |
        | -- * -- | -- * -- |
        |  / | \  |  / | \  |
        +---------+---------+
        |  \ | /  |  \ | /  |
        | -- * -- | -- * -- |
        |  / | \  |  / | \  |
        +---------+---------+
    */
    const float binSize = 360.0f / mathUtils::BIN_NUMBER;
    for (int n = 0; n < numImages; ++n) {
        /*
            Change image to gray scale
        */
        cv::Mat image;
        cv::cvtColor(images[n], image, cv::COLOR_BGR2GRAY);
        image.convertTo(image, CV_32FC1);

        /*
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
            Compute products of derivatives at every pixel
        */
        const cv::Mat Ix2 = Ix.mul(Ix); 
        const cv::Mat Iy2 = Iy.mul(Iy);

        cv::Mat magnitude;
        cv::pow(Ix2 + Iy2, 0.5f, magnitude);

        /*
            For each bin, build its index map
        */
        cv::Mat binIndex[mathUtils::BIN_NUMBER];
        cv::Mat orientation[mathUtils::BIN_NUMBER];
        for (int i = 0; i < mathUtils::BIN_NUMBER; ++i) {
            binIndex[i]    = cv::Mat::zeros(image.size(), CV_32FC1);
            orientation[i] = cv::Mat::zeros(image.size(), CV_32FC1);
        }
        
        /*
            For SIFT local descriptor, it uses 8 orientations.
            It may different from previous BIN_NUMBER, so we 
            also need to build alternative descriptorBinIndex
            and descriptorOrientation
        */
        cv::Mat descriptorBinIndex[8];
        cv::Mat descriptorOrientation[8];
        for (int i = 0; i < 8; ++i) {
            descriptorBinIndex[i]    = cv::Mat::zeros(image.size(), CV_32FC1);
            descriptorOrientation[i] = cv::Mat::zeros(image.size(), CV_32FC1);
        }

        /*
            For each pixel, calculate its orientation
            and assign to its corresponding binIndex
            (both binIndex and descriptorBinIndex)
        */
        const int width  = image.cols;
        const int height = image.rows;
        for (int iy = 0; iy < height; ++iy) {
            for (int ix = 0; ix < width; ++ix) {
                const float thetaRaw = 
                    std::atan2(Iy.at<float>(iy, ix), Ix.at<float>(iy, ix) + 1e-8f)
                    * (180.0f / mathUtils::PI);
                const float theta    = (thetaRaw >= 0.0f) ? thetaRaw : thetaRaw + 360.0f;

                const int bin = static_cast<int>((theta + 0.5f * binSize) / binSize) % mathUtils::BIN_NUMBER;
                binIndex[bin].at<float>(iy, ix) = 1.0f;

                const int descriptorBin = static_cast<int>((theta + 0.5f * 45.0f) / 45.0f) % 8;
                descriptorBinIndex[descriptorBin].at<float>(iy, ix) = 1.0f;
            }
        }

        /*
            For each bin index map, calculate its influence on
            pixels around
        */
        for (int bin = 0; bin < mathUtils::BIN_NUMBER; ++bin) {
            cv::GaussianBlur(binIndex[bin], orientation[bin], cv::Size(7, 7), 3);
            orientation[bin] = orientation[bin].mul(magnitude);
        }
        for (int bin = 0; bin < 8; ++bin) {
            cv::GaussianBlur(descriptorBinIndex[bin], descriptorOrientation[bin], cv::Size(7, 7), 3);
            descriptorOrientation[bin] = descriptorOrientation[bin].mul(magnitude);
        }

        /*
            For each pixel, calculate its main orientation

            mainOrientation: the main orientation index with 
                             the largest orientation value of a pixel
        */
        cv::Mat mainOrientation           = cv::Mat::zeros(image.size(), CV_8UC1);
        cv::Mat descriptorMainOrientation = cv::Mat::zeros(image.size(), CV_8UC1);
        for (int iy = 0; iy < height; ++iy) {
            for (int ix = 0; ix < width; ++ix) {
                /*
                    Find main orientation from BIN_NUMBER orientations,
                    and then add in orientation vector
                */
                float maxOrientationValue = 0.0f;
                int   maxOrientationIndex = 0;

                for (int bin = 0; bin < mathUtils::BIN_NUMBER; ++bin) {
                    const float value = orientation[bin].at<float>(iy, ix);
                    if (value > maxOrientationValue) {
                        maxOrientationValue = value;
                        maxOrientationIndex = bin;
                    }
                }
                mainOrientation.at<uchar>(iy, ix) = maxOrientationIndex;

                /*
                    Find main orientation from 8 orientations,
                    and then add in descriptor orientation vector
                */
                maxOrientationValue = 0.0f;
                maxOrientationIndex = 0;
                for (int bin = 0; bin < 8; ++bin) {
                    const float value = descriptorOrientation[bin].at<float>(iy, ix);
                    if (value > maxOrientationValue) {
                        maxOrientationValue = value;
                        maxOrientationIndex = bin;
                    }
                }
                descriptorMainOrientation.at<uchar>(iy, ix) = maxOrientationIndex;
            }
        }

        /*
            For each feature point, calculate its local descriptor
            based on its main orientation
        */
        std::vector<std::vector<float>> descriptors;
        for (auto& pos : featurePositions[n]) {
            const int   x     = pos.x;
            const int   y     = pos.y;
            const float angle = static_cast<float>(mainOrientation.at<uchar>(y, x)) * binSize;


            const int descriptorRotateBin = (angle < 22.5f) ?
                                            0 : 1 + static_cast<int>(angle - 22.5f) / 45;

            const cv::Point2i point(x, y);
            const cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point2f(point), -angle, 1);
            cv::Mat rotationDescriptorMainOrientation;
            cv::warpAffine(descriptorMainOrientation, rotationDescriptorMainOrientation, rotationMatrix, image.size());

            /*
                There are 16 4x4 size local pixels
                needed to calculate 8-orientation histogram
            */
            std::vector<float> descriptor;
            descriptor.reserve(16 * 8);

            const int offsetX[4] = { -8, -4, 0, 4 };
            const int offsetY[4] = { -8, -4, 0, 4 };
            for (auto& yo : offsetY) {
                for (auto& xo : offsetX) {
                    /*
                        For each 4x4 size, calculate its 8-orientation histogram
                    */
                    float orientationHistogram[8] = { 0.0f };

                    const int beginX = x + xo;
                    const int beginY = y + yo;
                    for (int iy = beginY; iy < beginY + 4; ++iy) {
                        for (int ix = beginX; ix < beginX + 4; ++ix) {
                            int mainBin = rotationDescriptorMainOrientation.at<uchar>(iy, ix);

                            // rotationDescriptorMainOrientation stores un-rotated bin,
                            // so we need to subtract descriptorRotateBin so that
                            // mainBin would be local bin
                            mainBin -= descriptorRotateBin;
                            mainBin  = (mainBin + 8) % 8;

                            orientationHistogram[mainBin] += 1.0f / 16.0f;
                        }
                    }

                    /*
                        Clip valus larger than 0.2, and then normalize

                        At last, push_back to descriptor
                    */
                    float sumHistogram = 0.0f;
                    for (int i = 0; i < 8; ++i) {
                        if (orientationHistogram[i] > 0.2f) {
                            orientationHistogram[i] = 0.2f;
                        }

                        sumHistogram += orientationHistogram[i];
                    }

                    for (int i = 0; i < 8; ++i) {
                        orientationHistogram[i] /= sumHistogram;
                        descriptor.push_back(orientationHistogram[i]);
                    }
                }
            }

            descriptors.push_back(descriptor);
        }

        out_featureDescriptors->push_back(descriptors);

        std::cout << "\r    Progress of calculating feature descriptors: " << (n + 1) << "/" << numImages;
    }

    std::cout << std::endl
              << "# Finish calculating feature descriptors"
              << std::endl;
}

} // namespace sis