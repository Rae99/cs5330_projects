#ifndef FILTERS_H
#define FILTERS_H

#pragma once
#include <opencv2/opencv.hpp>

// My custom grayscale implementation.
// src: input BGR image (CV_8UC3)
// dst: output grayscale-as-3channel image (CV_8UC3), each channel identical
// return 0 on success
int greyscale(cv::Mat &src, cv::Mat &dst);

// Sepia tone filter
// src: input BGR image (CV_8UC3)
// dst: output sepia-toned BGR image (CV_8UC3)
// return 0 on success
int sepia(cv::Mat &src, cv::Mat &dst);


int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
#endif