/*
    Ding, Junrui
    Februray 2026

    CS5330 Project 2 - features.h

    This header declares feature-extraction routines for each project task,
    including color, texture, and grass-detection features.
*/

#ifndef FEATURES_H
#define FEATURES_H

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

/*
    compute_task1_feature

    Task 1 feature: 7x7 center patch, RGB flattened to length 147.

    Arguments:
        const cv::Mat &img - input image (BGR or grayscale).
        std::vector<float> &feat - output feature vector.

    Returns:
        true on success, false on failure.
*/
bool compute_task1_feature(const cv::Mat &img, std::vector<float> &feat);

/*
    compute_task2_feature

    Task 2 feature using default histogram bin count.

    Arguments:
        const cv::Mat &img - input image (BGR).
        std::vector<float> &feat - output feature vector.

    Returns:
        true on success, false on failure.
*/
bool compute_task2_feature(const cv::Mat &img,
                           std::vector<float> &feat); // default bins=16
/*
    compute_task2_feature_rg_hist

    Task 2 feature using an rg-chromaticity histogram with configurable bins.

    Arguments:
        const cv::Mat &img - input image (BGR).
        std::vector<float> &feat - output feature vector.
        int bins - number of bins per channel.

    Returns:
        true on success, false on failure.
*/
bool compute_task2_feature_rg_hist(const cv::Mat &img, std::vector<float> &feat,
                                   int bins);

/*
    compute_task3_feature

    Task 3 feature: multi-histogram (whole image + center region).

    Arguments:
        const cv::Mat &img - input image (BGR).
        std::vector<float> &feat - output feature vector.

    Returns:
        true on success, false on failure.
*/
bool compute_task3_feature(const cv::Mat &img, std::vector<float> &feat);

/*
    compute_task4_feature

    Task 4 feature: whole-image color histogram plus texture histograms
    (magnitude and orientation).

    Arguments:
        const cv::Mat &img - input image (BGR).
        std::vector<float> &feat - output feature vector.

    Returns:
        true on success, false on failure.
*/
bool compute_task4_feature(const cv::Mat &img, std::vector<float> &feat);

/*
    extract_grass_features

    Task 7 feature: extract green grass features as a simple 5D vector.

    Arguments:
        const cv::Mat &img - input image (BGR).
        std::vector<float> &feat - output 5D feature vector.

    Returns:
        true on success, false on failure.
*/
bool extract_grass_features(const cv::Mat &img, std::vector<float> &feat);

#endif // FEATURES_H
