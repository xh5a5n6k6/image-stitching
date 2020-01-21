#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace sis {

/*
    ImageMatcher is used for image matching according to 
    featureMatchings between image pairs.

    out_imageAlignments: It stores all image matching alignments
                         between image pairs.

                         take image 1-2 pair for example, cv::Pointt
                         would store alignment from image2 to image1

                         Ex. cv::Point(-5, 3)
                             it means image2 needs to move x-direction -5
                             and y-direction 3 so that it would overlap
                             with image1

                         Note: alignment is calculated from the concate image
                               (Remain: image order is LEFT-TO-RIGHT,
                                        so movement is from right to left)

                            image1     image2               cv::Point(-4, 2)
                                                            (openCV's y-direction is up-to-down)
                         +----------+----------+           +----------+
                         |          |          |           |      +---+------+
                         |          |          |   ---->   |      |   |      |
                         |          |          |           |      |   |      |
                         |          |          |           |      |   |      |
                         +----------+----------+           +------+---+      |
                                                                  +----------+
*/
class ImageMatcher {
public:
    virtual void match(
        const std::vector<cv::Mat>&                          images,
        const std::vector<std::vector<cv::Point>>&           featurePositions,
        const std::vector<std::vector<std::pair<int, int>>>& featureMatchings,
        std::vector<cv::Point>* const                        out_imageAlignments) const = 0;
};

} // namespace sis