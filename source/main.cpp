#include "commandArgument.h"
#include "core/imageStitcher.h"

#include <iostream>
#include <string>

using namespace sis;

int main(int argc, char* argv[]) {
    if (argc == 1) {
        std::cout << "Image-Stitching -h for further information."
                  << std::endl;

        return EXIT_SUCCESS;
    }

    CommandArgument args(argc, argv);
    if (args.isHelpMessageRequested()) {
        args.printHelpMessage();

        return EXIT_SUCCESS;
    }

    std::cout << "Image-Stitching, copyright (c)2019-2020 Chia-Yu Chou\n"
              << std::endl;

    cv::Mat panorama;
    ImageStitcher imageStitcher(args);

    imageStitcher.solve(&panorama);
    cv::imwrite("./result/panorama_result.png", panorama);

    return EXIT_SUCCESS;
}