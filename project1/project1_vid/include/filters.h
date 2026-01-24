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

int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels = 10);

int depthGrayscale(const cv::Mat &src,     // original color image (CV_8UC3)
                   const cv::Mat &depth8,  // depth map (CV_8UC1)
                   cv::Mat &dst,           // output image (CV_8UC3)
                   unsigned char threshold // depth threshold
);
#endif