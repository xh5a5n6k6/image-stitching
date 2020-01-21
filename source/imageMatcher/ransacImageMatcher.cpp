#include "imageMatcher/ransacImageMatcher.h"

#include "mathUtils.h"

#include <cmath>

namespace sis {

RansacImageMatcher::RansacImageMatcher() = default;

void RansacImageMatcher::match(
    const std::vector<cv::Mat>&                          images,
    const std::vector<std::vector<cv::Point>>&           featurePositions,
    const std::vector<std::vector<std::pair<int, int>>>& featureMatchings,
    std::vector<cv::Point>* const                        out_imageAlignments) const {

    std::cout << "# Begin to match images between image pairs"
              << std::endl
              << "\r    Progress of image matching: 0/" << featureMatchings.size();

    const int numImageMatchings = static_cast<int>(featureMatchings.size());
    out_imageAlignments->reserve(numImageMatchings);

    /*
        Use RANSAC algorithm to calculate the best alignment
        of every two-image pair. To simplify this, we just
        use K = 500 to run RANSAC algorithm

        Remain : matchings is all feature matching pairs for
                 every two-image pair, and it is important that the
                 pair order is from n+1 to n

                 ex. std::pair<int, int>(3, 10)
                     it means image2's feature 3 matches image1's feature 10
    */
    const int K = 500;
    for (int n = 0; n < numImageMatchings; ++n) {
        const std::vector<std::pair<int, int>>& matching = featureMatchings[n];
        const int numMatchings = static_cast<int>(matching.size());

        const std::vector<cv::Point>& feaPos1 = featurePositions[n];
        const std::vector<cv::Point>& feaPos2 = featurePositions[n + 1];

        /*
            RANSAC algorithm needs to run K times,
            and there are three steps in each time
        */
        float     minDifference = std::numeric_limits<float>::max();
        cv::Point alignment     = cv::Point(0, 0);
        for (int k = 0; k < K; ++k) {
            /*
                Step 1
                Select n samples randomly

                In this case, we only need one sample
                to calculate its feature alignment
            */
            const int sample = mathUtils::nextInt(0, numMatchings);

            /*
                Step 2
                Calculate parameters with n samples

                In this case, there are two parameters,
                x-alignment and y-alignment
            */
            const std::pair<int, int>& sampleMatching = matching[sample];
            const cv::Point& point1 = feaPos1[sampleMatching.second];
            const cv::Point& point2 = feaPos2[sampleMatching.first];

            /*
                We have to calculate concate image's offset,
                so it needs to add x-offset of point2,
                and then use point1-point2 to represent alignment
            */
            const cv::Point offset(images[n].cols, 0);
            const cv::Point offsetPoint2 = point2 + offset;

            const cv::Point sampleAlignment = point1 - offsetPoint2;
            const float sampleDist2 = static_cast<float>(sampleAlignment.x * sampleAlignment.x +
                                                         sampleAlignment.y * sampleAlignment.y);

            // neglect bad matching which distance is larger
            // than image width
            if (sampleDist2 > images[n].cols * images[n].cols) {
                continue;
            }

            /*
                Step 3
                For each other N-n points, calculate its distance
                to the fitted model, count the number of inlier points

                In this case, we ignore calculation of the number
                of inlier points, we just calculate the sum of two-norm 
                difference of other features with sampleAlignment
            */
            float difference = 0.0f;
            for (auto& pair : matching) {
                const cv::Point& p1 = feaPos1[pair.second];
                const cv::Point& p2 = feaPos2[pair.first];
                const cv::Point moveP2 = p2 + offset + sampleAlignment;

                const cv::Point pointDiff = p1 - moveP2;
                const float dist2 = static_cast<float>(pointDiff.x * pointDiff.x + pointDiff.y * pointDiff.y);
                if (dist2 < images[n].cols * images[n].cols) {
                    difference += std::sqrt(dist2);
                }
            }

            if (difference < minDifference) {
                minDifference = difference;
                alignment     = sampleAlignment;
            }
        }

        out_imageAlignments->push_back(alignment);

        std::cout << "\r    Progress of image matching: " << (n + 1) << "/" << numImageMatchings;
    }

    std::cout << std::endl
              << "# Finish all image matchings"
              << std::endl;
}

} // namespace sis