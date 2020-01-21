#include "core/imageStitcher.h"

#include "bundleAdjuster/perspectiveBundleAdjuster.h"
#include "commandArgument.h"
#include "featureDescriptor/siftFeatureDescriptor.h"
#include "featureDetector/harrisFeatureDetector.h"
#include "featureMatcher/bruteForceFeatureMatcher.h"
#include "imageBlender/linearAlphaImageBlender.h"
#include "imageMatcher/ransacImageMatcher.h"
#include "imageWarpper/cylindricalImageWarpper.h"

#include <algorithm>
#include <cstdio>
#include <iostream>

#if (defined(_MSC_VER) || \
     (defined(__GNUC__) && (__GNUC_MAJOR__ >= 8))) 
#include <filesystem>
    namespace std_fs = std::filesystem;
#else
    #include <experimental/filesystem>
    namespace std_fs = std::experimental::filesystem;
#endif

namespace sis {

ImageStitcher::ImageStitcher(const CommandArgument& arguments) :
    _images(),
    _focalLengths(),
    _imageWarpper(nullptr),
    _featureDetector(nullptr),
    _featureDescriptor(nullptr),
    _featureMatcher(nullptr),
    _imageMatcher(nullptr),
    _imageBlender(nullptr),
    _bundleAdjuster(nullptr) {

    const std::string sizeRatio           = arguments.find("sizeRatio", "1.0");
    const std::string imageDirectory      = arguments.find("imageDirectory");
    const std::string focalLengthFilename = arguments.find("focalLengthFilename");
    const std::string imageWarpper        = arguments.find("imageWarpper", "cylindrical");
    const std::string featureDetector     = arguments.find("featureDetector", "harris");
    const std::string featureDescriptor   = arguments.find("featureDescriptor", "sift");
    const std::string featureMatcher      = arguments.find("featureMatcher", "brute-force");
    const std::string imageMatcher        = arguments.find("imageMatcher", "ransac");
    const std::string imageBlender        = arguments.find("imageBlender", "linear-alpha");
    const std::string bundleAdjuster      = arguments.find("bundleAdjuster", "perspective");

    // decide which imageWarpper to use
    if (imageWarpper == "cylindrical") {
        _imageWarpper = std::make_unique<CylindricalImageWarpper>();
    }
    else {
        std::cout << "Unknown imageWarpper type: <"
                  << imageWarpper << ">, use <cylindrical> instead"
                  << std::endl;

        _imageWarpper = std::make_unique<CylindricalImageWarpper>();
    }

    // decide which featureDetector to use
    if (featureDetector == "harris") {
        _featureDetector = std::make_unique<HarrisFeatureDetector>();
    }
    else {
        std::cout << "Unknown featureDetector type: <"
                  << featureDetector << ">, use <harris> instead"
                  << std::endl;

        _featureDetector = std::make_unique<HarrisFeatureDetector>();
    }

    // decide which featureDescriptor to use
    if (featureDescriptor == "sift") {
        _featureDescriptor = std::make_unique<SiftFeatureDescriptor>();
    }
    else {
        std::cout << "Unknown featureDescriptor type: <"
                  << featureDescriptor << ">, use <sift> instead"
                  << std::endl;

        _featureDescriptor = std::make_unique<SiftFeatureDescriptor>();
    }

    // decide which featureMatcher to use
    if (featureMatcher == "brute-force") {
        _featureMatcher = std::make_unique<BruteForceFeatureMatcher>();
    }
    else {
        std::cout << "Unknown featureMatcher type: <"
                  << featureMatcher << ">, use <brute-force> instead"
                  << std::endl;

        _featureMatcher = std::make_unique<BruteForceFeatureMatcher>();
    }

    // decide which imageMatcher to use
    if (imageMatcher == "ransac") {
        _imageMatcher = std::make_unique<RansacImageMatcher>();
    }
    else {
        std::cout << "Unknown imageMatcher type: <"
                  << imageMatcher << ">, use <ransac> instead"
                  << std::endl;

        _imageMatcher = std::make_unique<RansacImageMatcher>();
    }

    // decide which imageBlender to use
    if (imageBlender == "linear-alpha") {
        _imageBlender = std::make_unique<LinearAlphaImageBlender>();
    }
    else {
        std::cout << "Unknown imageBlender type: <"
                  << imageBlender << ">, use <linear-alpha> instead"
                  << std::endl;

        _imageBlender = std::make_unique<LinearAlphaImageBlender>();
    }

    // decide which bundleAdjuster to use
    if (bundleAdjuster == "perspective") {
        _bundleAdjuster = std::make_unique<PerspectiveBundleAdjuster>();
    }
    else {
        std::cout << "Unknown bundleAdjuster type: <"
                  << bundleAdjuster << ">, use <perspective> instead"
                  << std::endl;

        _bundleAdjuster = std::make_unique<PerspectiveBundleAdjuster>();
    }

    // read data input (images and focal lengths)
    _readData(imageDirectory, 
              focalLengthFilename,
              static_cast<float>(std::stold(sizeRatio)));
}

ImageStitcher::~ImageStitcher() = default;

void ImageStitcher::solve(cv::Mat* const out_panorama) const {
    // image warpping
    std::vector<cv::Mat> warpImages;
    std::vector<cv::Mat> warpImageIndices;
    _imageWarpper->warp(_images, _focalLengths, &warpImages, &warpImageIndices);
    
    // feature detection
    std::vector<std::vector<cv::Point>> featurePositions;
    _featureDetector->detect(warpImages, &featurePositions);

    // feature descriptor calculation
    std::vector<std::vector<std::vector<float>>> featureDescriptors;
    _featureDescriptor->calculate(warpImages, featurePositions, &featureDescriptors);

    // feature matching
    std::vector<std::vector<std::pair<int, int>>> featureMatchings;
    _featureMatcher->match(warpImages, featurePositions, featureDescriptors, &featureMatchings);

    // image matching
    std::vector<cv::Point> imageAlignments;
    _imageMatcher->match(warpImages, featurePositions, featureMatchings, &imageAlignments);

    // image blending (stitching)
    cv::Mat panorama;
    _imageBlender->blend(warpImages, imageAlignments, warpImageIndices, &panorama);

    // bundle adjustment
    cv::Mat adjustedPanorama;
    _bundleAdjuster->adjust(panorama, &adjustedPanorama);

    // writing result
    *out_panorama = adjustedPanorama;
}

void ImageStitcher::_readData(const std::string& imageDirectory, 
                              const std::string& focalLengthFilename,
                              const float        sizeRatio) {

    // clamp sizeRatio to 0.1 ~ 1.0
    const float safeSizeRatio = (sizeRatio > 1.0f) ? 1.0f :
                                (sizeRatio < 0.1f) ? 0.1f : sizeRatio;

    /*
        Read focal length file,
        and it would be used in cylindrical warping
    */
    FILE *f = fopen(focalLengthFilename.c_str(), "r");
    if (!f) {
        std::cout << "Focal length file can't open !"
                  << std::endl;

        exit(0);
    }

    char line[1024];
    while (fgets(line, 1024, f)) {
        if (line[strlen(line)] == '\n') {
            line[strlen(line) - 1] = '\0';
        }

        const float time = static_cast<float>(std::stold(line));
        _focalLengths.push_back(time);
    }

    /*
        Read input images,
        and the order of input photographs needs to be LEFT-TO-RIGHT.
    */
    std::cout << "# Begin to read images"
              << std::endl
              << "    Using image scale ratio: <"
              << safeSizeRatio << ">" << std::endl;

    std::vector<std::string> imageFilenames;
    imageFilenames.reserve(_focalLengths.size());
    for (const auto& entry : std_fs::directory_iterator(imageDirectory)) {
        imageFilenames.push_back(entry.path().string());
    }

    // Because filename loading order may be different from standard order,
    // we need to sort it first to make sure its order fits focal length's.
    std::sort(imageFilenames.begin(), imageFilenames.end());

    for (std::size_t i = 0; i < imageFilenames.size(); ++i) {
        std::cout << "    Image " << (i + 1) << ": " << imageFilenames[i]
                  << std::endl;

        const cv::Mat  image = cv::imread(imageFilenames[i]);
        const cv::Size resizeRes(static_cast<int>(image.cols * safeSizeRatio),
                                 static_cast<int>(image.rows * safeSizeRatio));

        cv::Mat resizeImage;
        cv::resize(image, resizeImage, resizeRes, cv::INTER_LINEAR);
        _images.push_back(resizeImage);
    }

    // create directory which stores result images
    std_fs::create_directory("./result");

#ifdef DRAW_WARP_IMAGES
    std_fs::create_directory("./result/warp");
#endif

#ifdef DRAW_FEATURE_IMAGES
    std_fs::create_directory("./result/feature");
#endif

#ifdef DRAW_FEATURE_MATCHING_IMAGES
    std_fs::create_directory("./result/matching");
#endif

#ifdef DRAW_BLEND_IMAGES
    std_fs::create_directory("./result/blend");
#endif

    std::cout << "# Total read " << _images.size() << " images"
              << std::endl;
}

} // namespace sis