#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace sis {

class BundleAdjuster;
class CommandArgument;
class FeatureDescriptor;
class FeatureDetector;
class FeatureMatcher;
class ImageBlender;
class ImageMatcher;
class ImageWarpper;

class ImageStitcher {
public:
    ImageStitcher(const CommandArgument& arguments);

    ~ImageStitcher();

    void solve(cv::Mat* const out_panorama) const;

private:
    void _readData(const std::string& imageDirectory, 
                   const std::string& focalLengthFilename,
                   const float        sizeRatio);

    // Input images
    // The order needs to be LEFT-TO-RIGHT
    std::vector<cv::Mat> _images;
    std::vector<float>   _focalLengths;

    std::unique_ptr<ImageWarpper>      _imageWarpper;
    std::unique_ptr<FeatureDetector>   _featureDetector;
    std::unique_ptr<FeatureDescriptor> _featureDescriptor;
    std::unique_ptr<FeatureMatcher>    _featureMatcher;
    std::unique_ptr<ImageMatcher>      _imageMatcher;
    std::unique_ptr<ImageBlender>      _imageBlender;
    std::unique_ptr<BundleAdjuster>    _bundleAdjuster;
};

} // namespace sis