#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace sis {

class FeatureDetector;
class FeatureDescriptor;

class ImageStitcher {
public:
	ImageStitcher(std::string imageDirectory, std::string focalLengthFilename);

	void solve(cv::Mat &dst);

private:
	void _readData(std::string imageDirectory, std::string focalLengthFilename);
    std::vector<cv::Mat> _cylindricalWarping(std::vector<cv::Mat> &images);
    std::vector<std::vector<std::pair<int, int> > > 
    _featureMatching(
        std::vector<std::vector<cv::Point> > featurePostions,
        std::vector<std::vector<std::vector<float> > > descriptors);
    std::vector<cv::Point> _imageMatching(
        std::vector<std::vector<cv::Point> > featurePostions,
        std::vector<std::vector<std::pair<int, int> > > matchings);
    cv::Mat _imageBlending(std::vector<cv::Point> alignments, std::vector<cv::Mat> warpImageIndices);
    cv::Mat _bundleAdjustment(cv::Mat panorama);

    std::shared_ptr<FeatureDetector> _detector;
    std::shared_ptr<FeatureDescriptor> _descriptor;

    // Input images
    // The order of input photographs needs to be left-to-right
	std::vector<cv::Mat> _images;
    std::vector<float> _focalLengths;

    // For feature matching threshold
    float _threshold;
};

} // namespace sis