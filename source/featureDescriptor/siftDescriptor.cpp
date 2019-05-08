#include "featureDescriptor/siftDescriptor.h"

#include "shared.h"

namespace sis {

SIFTDescriptor::SIFTDescriptor() {
}

std::vector<std::vector<std::vector<float> > > 
SIFTDescriptor::solve(
    std::vector<cv::Mat> images, 
    std::vector<std::vector<cv::Point> > featurePositions) {

    std::vector<cv::Mat> imagesCopy;
    int num = images.size();
    for (auto img : images) {
        imagesCopy.push_back(img.clone());
    }
    
    /*
        SIFT descriptor calculates main orientation for each feature,
        and then use local orientational window to calculate descriptor to 
        make sure its sensitivity to affine transform.  
        (including rotation, translation, etc)
    */

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
    std::vector<std::vector<std::vector<float> > > allDescriptors;
    int binSize = int(360.0f / BIN_NUMBER);
    for (int n = 0; n < num; n++) {
        /*
            Change image to gray scale
        */
        cv::Mat image;
        cv::cvtColor(images.at(n), image, CV_BGR2GRAY);
        image.convertTo(image, CV_32FC1);

        /*
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
            Compute products of derivatives at every pixel
        */
        cv::Mat Ix2, Iy2;
        Ix2 = Ix.mul(Ix);
        Iy2 = Iy.mul(Iy);

        cv::Mat magnitude;
        cv::pow(Ix2 + Iy2, 0.5f, magnitude);

        /*
            For each bin, build its index map
        */
        cv::Mat binIndex[BIN_NUMBER];
        cv::Mat orientation[BIN_NUMBER];
        for (int i = 0; i < BIN_NUMBER; i++) {
            binIndex[i] = cv::Mat::zeros(image.size(), CV_32FC1);
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
        for (int i = 0; i < 8; i++) {
            descriptorBinIndex[i] = cv::Mat::zeros(image.size(), CV_32FC1);
            descriptorOrientation[i] = cv::Mat::zeros(image.size(), CV_32FC1);
        }


        /*
            For each pixel, calculate its orientation
            and assign to its corresponding binIndex
            (both binIndex and descriptorBinIndex)
        */
        for (int j = 0; j < image.rows; j++) {
            for (int i = 0; i < image.cols; i++) {
                float theta = std::atan2f(Iy.at<float>(j, i), Ix.at<float>(j, i) + 1e-8f)
                              * (180.0f / PI);
                theta = int(theta + 360.0f) % 360;

                int bin = (int(theta + 0.5f * binSize) / binSize) % BIN_NUMBER;
                binIndex[bin].at<float>(j, i) = 1.0f;

                int descriptorBin = (int(theta + 0.5f * 45) / 45) % 8;
                descriptorBinIndex[descriptorBin].at<float>(j, i) = 1.0f;
            }
        }

        /*
            For each bin index map, calculate its influence on
            pixels around
        */
        for (int bin = 0; bin < BIN_NUMBER; bin++) {
            cv::GaussianBlur(binIndex[bin], orientation[bin], cv::Size(7, 7), 3);
            orientation[bin] = orientation[bin].mul(magnitude);
        }
        for (int bin = 0; bin < 8; bin++) {
            cv::GaussianBlur(descriptorBinIndex[bin], descriptorOrientation[bin], cv::Size(7, 7), 3);
            descriptorOrientation[bin] = descriptorOrientation[bin].mul(magnitude);
        }

        /*
            For each pixel, calculate its main orientation

            mainOrientation : the main orientation index with 
                              the largest orientation value of a pixel
        */
        cv::Mat mainOrientation = cv::Mat::zeros(image.size(), CV_8UC1);
        cv::Mat descriptorMainOrientation = cv::Mat::zeros(image.size(), CV_8UC1);
        for (int j = 0; j < image.rows; j++) {
            for (int i = 0; i < image.cols; i++) {
                /*
                    Find main orientation from _binSize orientations,
                    and then add in orientation vector
                */
                float maxOrientationValue = 0.0f;
                int maxOrientationIndex = 0;
                for (int bin = 0; bin < BIN_NUMBER; bin++) {
                    float value = orientation[bin].at<float>(j, i);
                    if (value > maxOrientationValue) {
                        maxOrientationValue = value;
                        maxOrientationIndex = bin;
                    }
                }
                mainOrientation.at<uchar>(j, i) = maxOrientationIndex;

                /*
                    Find main orientation from 8 orientations,
                    and then add in orientation vector
                */
                maxOrientationValue = 0.0f;
                maxOrientationIndex = 0;
                for (int bin = 0; bin < 8; bin++) {
                    float value = descriptorOrientation[bin].at<float>(j, i);
                    if (value > maxOrientationValue) {
                        maxOrientationValue = value;
                        maxOrientationIndex = bin;
                    }
                }
                descriptorMainOrientation.at<uchar>(j, i) = maxOrientationIndex;
            }
        }

        /*
            For each feature point, calculate its local descriptor
            based on its main orientation
        */
        std::vector<std::vector<float> > descriptors;
        for (auto pos : featurePositions.at(n)) {
            int x = pos.x;
            int y = pos.y;
            float angle = float(mainOrientation.at<uchar>(y, x)) * binSize;


            int descriptorRotateBin = (angle < 22.5f) ?
                                      0 : 1 + int(angle - 22.5f) / 45;

            cv::Mat rotationDescriptorMainOrientation;
            cv::Mat rotation = cv::getRotationMatrix2D(cv::Point2f(x, y), angle, 1);
            cv::warpAffine(descriptorMainOrientation, rotationDescriptorMainOrientation, rotation, image.size());

            /*
                There are 16 4x4 size local pixels
                needed to calculate 8-orientation histogram
            */
            std::vector<float> descriptor;
            int offsetX[4] = { -8, -4, 0, 4 };
            int offsetY[4] = { -8, -4, 0, 4 };
            for (auto yo : offsetY) {
                for (auto xo : offsetX) {
                    /*
                        For each 4x4 size, calculate its 8-orientation histogram
                    */
                    float orientationHistogram[8] = { 0.0f };
                    int beginX = x + xo;
                    int beginY = y + yo;
                    for (int j = beginY; j < beginY + 4; j++) {
                        for (int i = beginX; i < beginX + 4; i++) {
                            int mainBin = rotationDescriptorMainOrientation.at<uchar>(j, i);
                            mainBin -= descriptorRotateBin;
                            mainBin = (mainBin + 8) % 8;

                            orientationHistogram[mainBin] += 1.0f / 16.0f;
                        }
                    }

                    /*
                        Clip valus larger than 0.2, and renormalize

                        At last, push_back to descriptor
                    */
                    float sum = 0.0f;
                    for (int i = 0; i < 8; i++) {
                        if (orientationHistogram[i] > 0.2f) {
                            orientationHistogram[i] = 0.2f;
                        }

                        sum += orientationHistogram[i];
                    }

                    for (int i = 0; i < 8; i++) {
                        orientationHistogram[i] /= sum;
                        descriptor.push_back(orientationHistogram[i]);
                    }
                }
            }

            descriptors.push_back(descriptor);
        }

        fprintf(stderr, "\rFinish SIFT feature descriptor of image %d", n + 1);
        allDescriptors.push_back(descriptors);
    }
    fprintf(stderr, "\n");

    return allDescriptors;
}

} // namespace sis