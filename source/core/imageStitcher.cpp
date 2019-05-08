#include "core/imageStitcher.h"

#include "featureDescriptor/siftDescriptor.h"

#include "featureDetector/harrisCornerDetector.h"

#include <filesystem>
#include <random>

namespace sis {

ImageStitcher::ImageStitcher(std::string imageDirectory, std::string focalLengthFilename) :
    _threshold(0.7f) {
    _readData(imageDirectory, focalLengthFilename);

    _detector = std::make_shared<HarrisCornerDetector>();
    _descriptor = std::make_shared<SIFTDescriptor>();
}

void ImageStitcher::solve(cv::Mat &dst) {
    std::vector<cv::Mat> warpImageIndices = _cylindricalWarping(_images);
    std::vector<std::vector<cv::Point> > featurePos = _detector->solve(_images);
    std::vector<std::vector<std::vector<float> > > descriptors = _descriptor->solve(_images, featurePos);
    std::vector<std::vector<std::pair<int, int> > > matchings = _featureMatching(featurePos, descriptors);
    std::vector<cv::Point> alignments = _imageMatching(featurePos, matchings);
    cv::Mat panorama = _imageBlending(alignments, warpImageIndices);
    dst = _bundleAdjustment(panorama);
}

void ImageStitcher::_readData(std::string imageDirectory, std::string focalLengthFilename) {
    /*
        Read input images

        The order of input photographs needs to be left-to-right
    */
	for (const auto & entry : std::filesystem::directory_iterator(imageDirectory)) {
		cv::Mat img = cv::imread(entry.path().string());
		_images.push_back(img);
	}

    fprintf(stderr, "Finish reading input images, there are total %d photographs\n", _images.size());

    /*
        Read focal length file to calculate cylindrical warping
    */
    FILE *f;
    errno_t err;

    if ((err = fopen_s(&f, focalLengthFilename.c_str(), "r")) != 0) {
        fprintf(stderr, "Focal length file can't open !\n");
        exit(0);
    }
    char line[1024];
    while (fgets(line, 1024, f)) {
        if (line[strlen(line)] == '\n')
            line[strlen(line) - 1] = '\0';

        float time = std::stof(line);
        _focalLengths.push_back(time);
    }
}

std::vector<cv::Mat> ImageStitcher::_cylindricalWarping(std::vector<cv::Mat> &images) {
    int num = images.size();

    std::vector<cv::Mat> allWarpImageIndices;
    for (int n = 0; n < num; n++) {
        cv::Mat image = images.at(n).clone();
        image.convertTo(image, CV_32FC3);

        int width = image.cols;
        int height = image.rows;

        float f = _focalLengths.at(n);
        float invS = 1.0f / f;

        int xCenter = width / 2;
        int yCenter = height / 2;

        /*
            Hack here, 
            make warpping image without x-direction black edges,
            it is convenient to image blending

            xBound : x-direction bounding coordinate with origin at the
                     image center, and then transform to cylindrical coordinate

            offset : Because we need to remove x-direction black edges, 
        */
        int xBound = (width - 1) - xCenter;
        xBound = int(f * std::atan2f(xBound, f));
        int offset = (width - 1 - 2 * xBound) / 2;
        cv::Mat warpImage = cv::Mat::zeros(cv::Size(2 * xBound, height), image.type());
        cv::Mat warpImageIndex = cv::Mat::zeros(cv::Size(2 * xBound, height), CV_32FC1);
 
        /*
            For each warping image's pixel, use inverse warping interpolation
            to calculate its value
        */
        for (int j = 0; j < warpImage.rows; j++) {
            for (int i = 0; i < warpImage.cols; i++) {
                /*
                    It needs to make image center be the origin,
                    so we need to substract offset first
                */
                float x = i + offset - xCenter;
                float y = j - yCenter;

                /*
                    Because y will use x to calculate, y needs to be
                    calculated before x
                */
                y = std::sqrtf(x * x + f * f) * invS * y;
                x = f * std::tanf(x * invS);

                /*
                    Add offset back
                */
                x += xCenter;
                y += yCenter;

                /*
                    Filter out-of-range coordinate, and then use 
                    bi-linear interpolation to calculate final value
                */
                if (x < 0.0f ||
                    x > width - 1 ||
                    y < 0.0f ||
                    y > height - 1) {
                    continue;
                }

                /*
                    As we use bi-linear interpolation, we need to 
                    calculate its inverse corresponding four border coordinates,
                    and then use these coordinates' values to calculate the result
                */
                int xFloor = std::floor(x);
                int xCeil = std::ceil(x);
                int yFloor = std::floor(y);
                int yCeil = std::ceil(y);

                float xt = xCeil - x;
                float yt = yCeil - y;
                
                cv::Vec3f t1 = image.at<cv::Vec3f>(yFloor, xFloor) * xt
                               + image.at<cv::Vec3f>(yFloor, xCeil) * (1.0f - xt);
                cv::Vec3f t2 = image.at<cv::Vec3f>(yCeil, xFloor) * xt
                               + image.at<cv::Vec3f>(yCeil, xCeil) * (1.0f - xt);

                warpImage.at<cv::Vec3f>(j, i) = t1 * yt + t2 * (1.0f - yt);
                warpImageIndex.at<float>(j, i) = 1.0f;
            }
        }
        warpImage.convertTo(images.at(n), CV_8UC3);
        allWarpImageIndices.push_back(warpImageIndex);

        fprintf(stderr, "\rFinish cylindrical warping of image %d", n + 1);
    }
    fprintf(stderr, "\n");

    return allWarpImageIndices;
}

std::vector<std::vector<std::pair<int, int> > > 
ImageStitcher::_featureMatching(
    std::vector<std::vector<cv::Point> > featurePostions,
    std::vector<std::vector<std::vector<float> > > descriptors) {

    std::vector<cv::Mat> imagesCopy;
    int num = _images.size();
    for (auto img : _images) {
        imagesCopy.push_back(img.clone());
    }

    /*
        Use brute-force feature matching method
        for every two-image pair
    */
    std::vector<std::vector<std::pair<int, int> > > allMatchingIndices;
    for (int n = 0; n < num - 1; n++) {
        std::vector<cv::Point> feaPos1 = featurePostions.at(n);
        std::vector<cv::Point> feaPos2 = featurePostions.at(n + 1);
        std::vector<std::vector<float> > des1 = descriptors.at(n);
        std::vector<std::vector<float> > des2 = descriptors.at(n + 1);

        cv::Mat concate;
        cv::hconcat(_images[n], _images[n + 1], concate);

        /*
            From n+1_th image matches n_th image
            and the ratio of first distance to second distance
            needs to be less than the threshold (default = 0.7)

            matchIndex : image2's featureIndex matches image1's featureIndex

                         ex. std::pair<int, int>(3, 10)
                         means image2's feature 3 matches image1's feature 10
        */
        std::vector<std::pair<int, int> > matchingIndex;
        for (int i = 0; i < des2.size(); i++) {
            cv::Mat feature2 = cv::Mat(des2.at(i));

            float firstDist = std::numeric_limits<float>::max();
            int firstIndex = 0;
            float secondDist = std::numeric_limits<float>::max();
            for (int j = 0; j < des1.size(); j++) {
                cv::Mat feature1 = cv::Mat(des1.at(j));
                float dist = cv::norm(feature2, feature1, cv::NORM_L2);
                if (dist < firstDist) {
                    secondDist = firstDist;
                    firstDist = dist;
                    firstIndex = j;
                }
                else if (dist < secondDist) {
                    secondDist = dist;
                }
            }
            if (firstDist / secondDist < _threshold) {
                matchingIndex.push_back(std::make_pair(i, firstIndex));

                cv::Point point1 = feaPos1.at(firstIndex);
                cv::Point point2 = feaPos2.at(i);
                cv::line(concate, point1, point2 + cv::Point(_images[n].cols, 0), cv::Scalar(0, 255, 0), 2);

                cv::circle(concate, point1, 1, cv::Scalar(0, 0, 255), CV_FILLED);
                cv::circle(concate, point2 + cv::Point(_images[n].cols, 0), 1, cv::Scalar(0, 0, 255), CV_FILLED);
            }
        }
        
        allMatchingIndices.push_back(matchingIndex);
        fprintf(stderr, "\rFinish feature matching of image %d-%d", n + 1, n + 2);
    }
    fprintf(stderr, "\n");

    return allMatchingIndices;
}

std::vector<cv::Point> ImageStitcher::_imageMatching(
    std::vector<std::vector<cv::Point> > featurePostions,
    std::vector<std::vector<std::pair<int, int> > > matchings) {
    int num = matchings.size();

    /*
        Use RANSAC algorithm to calculate the best alignment
        of every two-image pair. To simplify this, we just
        use K = 1000 to run RANSAC algorithm

        Remain : matchings is all feature matching pairs for
                 every two-image pair, and it is important that the
                 pair order is from n+1 to n

                 ex. std::pair<int, int>(3, 10)
                 means image2's feature 3 matches image1's feature 10
    */
    int K = 500;
    std::vector<cv::Point> allAlignments;
    for (int n = 0; n < num; n++) {
        std::vector<std::pair<int, int> > matching = matchings.at(n);
        std::vector<cv::Point> feaPos1 = featurePostions.at(n);
        std::vector<cv::Point> feaPos2 = featurePostions.at(n + 1);

        /*
            RANSAC algorithm needs to run K times,
            and there are three steps in each time
        */
        float minDifference = std::numeric_limits<float>::max();
        cv::Point alignment = cv::Point(0, 0);
        for (int k = 0; k < K; k++) {
            /*
                Step 1
                Select n samples randomly

                In this case, we only need one sample
                to calculate its feature alignment
            */
            std::random_device rd;
            std::default_random_engine gen = std::default_random_engine(rd());
            std::uniform_int_distribution<int> dis(0, matching.size() - 1);
            int sample = dis(gen);

            /*
                Step 2
                Calculate parameters with n samples

                In this case, there are two parameters,
                x-alignment and y-alignment
            */
            std::pair<int, int> sampleMatching = matching.at(sample);
            cv::Point point1 = feaPos1.at(sampleMatching.second);
            cv::Point point2 = feaPos2.at(sampleMatching.first);

            /*
                We have to calculate concate image's offset,
                so it needs to add x-offset of point2,
                and then use point1-point2 to represent alignment
            */
            cv::Point offset = cv::Point(_images.at(n).cols, 0);
            point2 += offset;
            cv::Point sampleAlignment = point1 - point2;
            float sampleDist = sampleAlignment.x * sampleAlignment.x +
                               sampleAlignment.y * sampleAlignment.y;
            if (std::sqrtf(sampleDist) > _images.at(n).cols) {
                continue;
            }

            /*
                Step 3
                For each other N-n points, calculate its distance
                to the fitted model, count the number of inlier points

                In this case, we ignore calculation of the number of inlier points,
                we just calculate the sum of two-norm difference
                of other features with sampleAlignment
            */
            float difference = 0.0f;
            for (auto pair : matching) {
                cv::Point p1 = feaPos1.at(pair.second);
                cv::Point p2 = feaPos2.at(pair.first);
                p2 += offset;
                p2 += sampleAlignment;

                cv::Point pointDiff = p1 - p2;
                float dist = std::sqrtf(pointDiff.x * pointDiff.x + pointDiff.y * pointDiff.y);
                if (dist < _images.at(n).cols) {
                    difference += dist;
                }
            }

            if (difference < minDifference) {
                minDifference = difference;
                alignment = sampleAlignment;
            }
        }

        allAlignments.push_back(alignment);
        fprintf(stderr, "\rFinish image matching of image %d-%d", n + 1, n + 2);
    }
    fprintf(stderr, "\n");

    return allAlignments;
}

cv::Mat ImageStitcher::_imageBlending(
    std::vector<cv::Point> alignments, 
    std::vector<cv::Mat> warpImageIndices) {
    int aNum = alignments.size();
    std::vector<cv::Point> accumulateAlignments;
    accumulateAlignments.assign(alignments.begin(), alignments.end());

    /*
        Because we calculate alignment of every two-image pair,
        we need to add alignment globally, called accumulateAlignment

        ex. alignment1 is (-3,  4) of image2-1
            alignment2 is (-5, -2) of image3-2

            after add alignment globally
            accumulateAlignment2 += alignment2 + alignment1

            accumulateAlignment1 is (-3, 4) of image2-1
            accumulateAlignment2 is (-8, 2) of image3-1
    */
    int minDy = (accumulateAlignments.at(0).y < 0) ? accumulateAlignments.at(0).y : 0;
    int maxDy = (accumulateAlignments.at(0).y > 0) ? accumulateAlignments.at(0).y : 0;
    for (int n = 1; n < aNum; n++) {
        accumulateAlignments.at(n) += accumulateAlignments.at(n - 1);

        minDy = (accumulateAlignments.at(n).y < minDy) ? accumulateAlignments.at(n).y : minDy;
        maxDy = (accumulateAlignments.at(n).y > maxDy) ? accumulateAlignments.at(n).y : maxDy;
    }

    /*
        Build final panorama image with correct size

        allWidth : sum of total images' widths,
                   and then add last image's x-alignment
        allHeight : image's height
                    if(minDy < 0)
                        allHeight += -minDy
                    if(maxDy > 0)
                        allHeight += maxDy

        offsetX : starting x-position for image-2 to image-N
    */
    int allWidth = 0;
    std::vector<int> offsetX;
    for (auto img : _images) {
        allWidth += img.cols;
        offsetX.push_back(allWidth);
    }
    allWidth += accumulateAlignments.at(aNum - 1).x;

    int allHeight = _images.at(0).rows;
    allHeight += (minDy < 0) ? -minDy : 0;
    allHeight += (maxDy > 0) ? maxDy : 0;

    cv::Mat panorama = cv::Mat::zeros(cv::Size(allWidth, allHeight), CV_32FC3);
    cv::Mat panoramaIndex = cv::Mat::zeros(panorama.size(), CV_8UC1);

    /*
        Stitch each image
    */
    int num = _images.size();
    for (int n = 0; n < num; n++) {
        cv::Mat image = _images.at(n);
        int width = image.cols;
        int height = image.rows;

        int beginX =  (n == 0) ? 
                      0 : offsetX.at(n - 1) + accumulateAlignments.at(n - 1).x;
        int beginY =  (n == 0) ? 
                      -minDy : -minDy + accumulateAlignments.at(n - 1).y;

        /*
            From the second image, we need to use origin alignment
            to build x-linear blending weight
        */
        float intersectionRegion = (n > 0) ?
                                 -alignments.at(n - 1).x : 0.0f;

        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                if (warpImageIndices.at(n).at<float>(j, i) > 0) {
                    cv::Vec3f originValue = panorama.at<cv::Vec3f>(j + beginY, i + beginX);
                    cv::Vec3f addValue = cv::Vec3f(image.at<cv::Vec3b>(j, i));
                    if (panoramaIndex.at<uchar>(j + beginY, i + beginX) == 0) {
                        panoramaIndex.at<uchar>(j + beginY, i + beginX) = 1;
                        panorama.at<cv::Vec3f>(j + beginY, i + beginX) = addValue;
                    }
                    else {
                        float addWeight = i / intersectionRegion;
                        panorama.at<cv::Vec3f>(j + beginY, i + beginX) = 
                            (1.0f - addWeight) * originValue + addWeight * addValue;
                    }
                }
            }
        }
    }

    panorama.convertTo(panorama, CV_8UC3);

    fprintf(stderr, "Finish image blending using x-linear alpha-blending\n");
    return panorama;
}

cv::Mat ImageStitcher::_bundleAdjustment(cv::Mat panorama) {
    cv::Point2f pt1[4];
    cv::Point2f pt2[4];

    pt2[0] = cv::Point2f(0, 0);
    pt2[1] = cv::Point2f(panorama.cols - 1, 0);
    pt2[2] = cv::Point2f(0, panorama.rows - 1);
    pt2[3] = cv::Point2f(panorama.cols - 1, panorama.rows - 1);


    for (int j = 0; j < panorama.rows; j++) {
        if (panorama.at<cv::Vec3b>(j, 0) != cv::Vec3b(0, 0, 0)) {
            pt1[0] = cv::Point2f(0, j);
            break;
        }
    }

    for (int j = 0; j < panorama.rows; j++) {
        if (panorama.at<cv::Vec3b>(j, panorama.cols - 1) != cv::Vec3b(0, 0, 0)) {
            pt1[1] = cv::Point2f(panorama.cols - 1, j);
            break;
        }
    }

    for (int j = panorama.rows - 1; j >= 0; j--) {
        if (panorama.at<cv::Vec3b>(j, 0) != cv::Vec3b(0, 0, 0)) {
            pt1[2] = cv::Point2f(0, j);
            break;
        }
    }

    for (int j = panorama.rows - 1; j >= 0; j--) {
        if (panorama.at<cv::Vec3b>(j, panorama.cols - 1) != cv::Vec3b(0, 0, 0)) {
            pt1[3] = cv::Point2f(panorama.cols - 1, j);
            break;
        }
    }

    cv::Mat panoramaRectangle;
    cv::Mat perspectMatrix = cv::getPerspectiveTransform(pt1, pt2);
    cv::warpPerspective(panorama, panoramaRectangle, perspectMatrix, panorama.size());

    fprintf(stderr, "Finish bundle adjustment using perspective warping\n");
    return panoramaRectangle;
}

} // namespace sis