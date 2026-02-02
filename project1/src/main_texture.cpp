#include <opencv2/opencv.hpp>
#include <cstdio>
#include "filters.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("Usage: %s image1 image2\n", argv[0]);
        return -1;
    }

    // read two texture images
    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        std::printf("Error: could not read input images\n");
        return -2;
    }

    // gradient images
    cv::Mat sx1, sy1, mag1;
    cv::Mat sx2, sy2, mag2;

    // compute gradients for image 1
    sobelX3x3(img1, sx1);
    sobelY3x3(img1, sy1);
    magnitude(sx1, sy1, mag1);

    // compute gradients for image 2
    sobelX3x3(img2, sx2);
    sobelY3x3(img2, sy2);
    magnitude(sx2, sy2, mag2);

    // show results
    cv::imshow("Image 1", img1);
    cv::imshow("Gradient Magnitude 1", mag1);

    cv::imshow("Image 2", img2);
    cv::imshow("Gradient Magnitude 2", mag2);

    cv::waitKey(0);
    return 0;
}