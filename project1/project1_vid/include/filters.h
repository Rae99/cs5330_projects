#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

// My custom grayscale implementation.
// src: input BGR image (CV_8UC3)
// dst: output grayscale-as-3channel image (CV_8UC3), each channel identical
// return 0 on success
int greyscale(cv::Mat &src, cv::Mat &dst);

#endif