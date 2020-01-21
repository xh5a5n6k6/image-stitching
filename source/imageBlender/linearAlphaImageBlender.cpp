#include "imageBlender/linearAlphaImageBlender.h"

namespace sis {

LinearAlphaImageBlender::LinearAlphaImageBlender() = default;

void LinearAlphaImageBlender::_blendImpl(
    const std::vector<cv::Mat>&   images,
    const std::vector<cv::Point>& imageAlignments,
    const std::vector<cv::Mat>&   warpImageIndices,
    cv::Mat* const                out_blendImage) const {

    std::cout << "# Begin to blend images using x-direction alpha blending"
              << std::endl;

    const int numImages     = static_cast<int>(images.size());
    const int numAlignments = static_cast<int>(imageAlignments.size());

    /*
        Because we calculate alignment of every two-image pair,
        we need to add alignment globally, called accumulateAlignment

        ex. alignment1 is (-3,  4) of image2-1
            alignment2 is (-5, -2) of image3-2

            after adding alignment globally
            accumulateAlignment2 = alignment2 + alignment1

            accumulateAlignment1 is (-3, 4) of image2-1
            accumulateAlignment2 is (-8, 2) of image3-1
    */
    std::vector<cv::Point> accumulateAlignments;
    accumulateAlignments.assign(imageAlignments.begin(), imageAlignments.end());

    int minDy = (accumulateAlignments[0].y < 0) ? accumulateAlignments[0].y : 0;
    int maxDy = (accumulateAlignments[0].y > 0) ? accumulateAlignments[0].y : 0;
    for (int n = 1; n < numAlignments; ++n) {
        accumulateAlignments[n] += accumulateAlignments[n - 1];

        minDy = (accumulateAlignments[n].y < minDy) ? accumulateAlignments[n].y : minDy;
        maxDy = (accumulateAlignments[n].y > maxDy) ? accumulateAlignments[n].y : maxDy;
    }

    /*
        Build final panorama image with correct size

        allWidth : sum of total images' widths,
                   and then add last image's x-alignment
        allHeight: image's height
                    if(minDy < 0)
                        allHeight += -minDy
                    if(maxDy > 0)
                        allHeight += maxDy

        offsetX  : starting x-position for image-2 to image-N
    */
    int allWidth  = 0;
    int allHeight = 0;
    std::vector<int> offsetX;
    offsetX.reserve(images.size());

    for (auto& img : images) {
        allWidth += img.cols;
        offsetX.push_back(allWidth);
    }
    allWidth += accumulateAlignments[numAlignments - 1].x;

    allHeight = images[0].rows;
    allHeight += (minDy < 0) ? -minDy : 0;
    allHeight += (maxDy > 0) ? maxDy : 0;

    cv::Mat panorama      = cv::Mat::zeros(cv::Size(allWidth, allHeight), CV_32FC3);
    cv::Mat panoramaIndex = cv::Mat::zeros(panorama.size(), CV_8UC1);

    /*
        Stitch each image
    */
    for (int n = 0; n < numImages; ++n) {
        const cv::Mat& image = images[n];
        const int width  = image.cols;
        const int height = image.rows;

        const int beginX = (n == 0) ?
                           0 : offsetX[n - 1] + accumulateAlignments[n - 1].x;
        const int beginY = (n == 0) ?
                           -minDy : -minDy + accumulateAlignments[n - 1].y;

        /*
            From the second image, we need to use origin alignment
            to build x-linear blending weight
        */
        const float intersectionRegion = (n > 0) ?
                                         -imageAlignments[n - 1].x : 0.0f;

        for (int iy = 0; iy < height; ++iy) {
            for (int ix = 0; ix < width; ++ix) {
                if (warpImageIndices[n].at<float>(iy, ix) > 0) {
                    const cv::Vec3f originValue = panorama.at<cv::Vec3f>(iy + beginY, ix + beginX);
                    const cv::Vec3f addValue    = cv::Vec3f(image.at<cv::Vec3b>(iy, ix));
                    
                    if (panoramaIndex.at<uchar>(iy + beginY, ix + beginX) == 0) {
                        panoramaIndex.at<uchar>(iy + beginY, ix + beginX) = 1;
                        panorama.at<cv::Vec3f>(iy + beginY, ix + beginX)  = addValue;
                    }
                    else {
                        const float addWeight = ix / intersectionRegion;

                        panorama.at<cv::Vec3f>(iy + beginY, ix + beginX)
                            = (1.0f - addWeight) * originValue + addWeight * addValue;
                    }
                }
            }
        }
    }

    panorama.convertTo(panorama, CV_8UC3);

    *out_blendImage = panorama;

    std::cout << "# Finish image blending"
              << std::endl;
}

} // namespace sis