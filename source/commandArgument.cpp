#include "commandArgument.h"

#include <iostream>
#include <utility>

namespace sis {

CommandArgument::CommandArgument(int argc, char* argv[]) :
    _isHelpMessageRequested(false),
    _arguments() {

    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        ++i;

        if (argument == "-h") {
            _isHelpMessageRequested = true;
        }
        else if (argument == "-scl") {
            _arguments.insert(std::make_pair("sizeRatio", std::string(argv[i])));
        }
        else if (argument == "-iw") {
            _arguments.insert(std::make_pair("imageWarpper", std::string(argv[i])));
        }
        else if (argument == "-fdt") {
            _arguments.insert(std::make_pair("featureDetector", std::string(argv[i])));
        }
        else if (argument == "-fdr") {
            _arguments.insert(std::make_pair("featureDescriptor", std::string(argv[i])));
        }
        else if (argument == "-fm") {
            _arguments.insert(std::make_pair("featureMatcher", std::string(argv[i])));
        }
        else if (argument == "-im") {
            _arguments.insert(std::make_pair("imageMatcher", std::string(argv[i])));
        }
        else if (argument == "-ib") {
            _arguments.insert(std::make_pair("imageBlender", std::string(argv[i])));
        }
        else if (argument == "-ba") {
            _arguments.insert(std::make_pair("bundleAdjuster", std::string(argv[i])));
        }
    }

    _arguments.insert(std::make_pair("imageDirectory", std::string(argv[argc - 2])));
    _arguments.insert(std::make_pair("focalLengthFilename", std::string(argv[argc - 1])));
}

const std::string CommandArgument::find(const std::string& key, 
                                        const std::string& defaultValue) const {

    const auto& res = _arguments.find(key);

    if (res == _arguments.end()) {
        return defaultValue;
    }
    else {
        return res->second;
    }
}

bool CommandArgument::isHelpMessageRequested() const {
    return _isHelpMessageRequested;
}

void CommandArgument::printHelpMessage() const {
    std::cout << R"(Image-Stitching, copyright (c)2019-2020 Chia-Yu, Chou

[<options>] <images directory path> <focal length file path>

Notice you need to specify images 'directory path' and focal length 'file path'.
For example:
Image-Stitching ./IMAGES/ ./FOCAL_LENGTH.txt

Options:
    -h             Print this help text.

    -scl  <ratio>  Specify scale ratio of input images used in image stitching.
                   Ratio range is from <0.1> to <1.0>                 

                   default: <1.0>

    -iw   <method> Specify imageWarpper method used for image warpping.
                   It currently only supports one method.
                   <cylindrical>

                   default: <cylindrical>
             
    -fdt  <method> Specify featureDetector method used for feature detection.
                   It currently only supports one method.
                   <harris>

                   default: <harris> 

    -fdr  <method> Specify featureDescriptor method used for feature descriptor calculation.
                   It currently only supports one method.
                   <sift>

                   default: <sift>

    -fm   <method> Specify featureMatcher method used for feature matching.
                   It currently only supports one method.
                   <brute-force>

                   default: <brute-force>

    -im   <method> Specify imageMatcher method used for image matching.
                   It currently only supports one method.
                   <ransac>

                   default: <ransac>

    -ib   <method> Specify imageBlender method used for image blending (stitching).
                   It currently only supports one method.
                   <linear-alpha>

                   default: <linear-alpha>

    -ba   <method> Specify bundleAdjuster method used for bundle adjustment.
                   It currently only supports one method.
                   <perspective>

                   default: <perspective>
)";

}

} // namespace sis