#include "featureMatcher/bruteForceFeatureMatcher.h"

#include <limits>

namespace sis {

BruteForceFeatureMatcher::BruteForceFeatureMatcher() :
    BruteForceFeatureMatcher(0.7f) {
}

BruteForceFeatureMatcher::BruteForceFeatureMatcher(const float threshold) :
    _threshold(threshold) {
}

void BruteForceFeatureMatcher::_matchImpl(
    const std::vector<cv::Mat>&                          images,
    const std::vector<std::vector<cv::Point>>&           featurePositions,
    const std::vector<std::vector<std::vector<float>>>&  featureDescriptors,
    std::vector<std::vector<std::pair<int, int>>>* const out_featureMatches) const {

    std::cout << "# Begin to match features between image pairs"
              << std::endl
              << "\r    Progress of feature matching: 0/" << (images.size() - 1)
              << std::flush;

    const int numImages = static_cast<int>(images.size());
    out_featureMatches->reserve(numImages - 1);

    /*
        Use brute-force feature matching method
        for every two-image pair
    */
    for (int n = 0; n < numImages - 1; ++n) {
        const std::vector<cv::Point>& feaPos1 = featurePositions[n];
        const std::vector<cv::Point>& feaPos2 = featurePositions[n + 1];
        const std::vector<std::vector<float>>& des1 = featureDescriptors[n];
        const std::vector<std::vector<float>>& des2 = featureDescriptors[n + 1];

        /*
            From n+1_th image matches n_th image
            and the ratio of first distance to second distance
            needs to be less than the threshold (default = 0.7)

            matchIndex : image2's featureIndex matches image1's featureIndex

                         ex. std::pair<int, int>(3, 10)
                             it means image2's feature 3 matches image1's feature 10
        */
        const int numDes1 = static_cast<int>(des1.size());
        const int numDes2 = static_cast<int>(des2.size());

        std::vector<std::pair<int, int>> matchingIndex;
        for (int d2i = 0; d2i < numDes2; ++d2i) {
            const cv::Mat feature2(des2[d2i]);

            float firstDist  = std::numeric_limits<float>::max();
            int   firstIndex = 0;
            float secondDist = std::numeric_limits<float>::max();

            for (int d1i = 0; d1i < numDes1; ++d1i) {
                const cv::Mat feature1(des1[d1i]);

                const float dist = static_cast<float>(cv::norm(feature2, feature1, cv::NORM_L2));
                if (dist < firstDist) {
                    secondDist = firstDist;
                    firstDist  = dist;
                    firstIndex = d1i;
                }
                else if (dist < secondDist) {
                    secondDist = dist;
                }
            }

            if (firstDist / secondDist < _threshold) {
                matchingIndex.push_back(std::make_pair(d2i, firstIndex));
            }
        }

        out_featureMatches->push_back(matchingIndex);

        std::cout << "\r    Progress of feature matching: " << (n + 1) << "/" << (numImages - 1)
                  << std::flush;
    }

    std::cout << std::endl
              << "# Finish all feature matchings"
              << std::endl;
}

} // namespace sis