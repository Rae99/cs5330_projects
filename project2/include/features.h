#ifndef FEATURES_H
#define FEATURES_H

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

// Task 1 feature: 7x7 center patch, RGB flattened to length 147
bool compute_task1_feature(const cv::Mat &img, std::vector<float> &feat);

// Task2
bool compute_task2_feature(const cv::Mat &img,
                           std::vector<float> &feat); // default bins=16
bool compute_task2_feature_rg_hist(const cv::Mat &img, std::vector<float> &feat,
                                   int bins);

// Task3: multi-histogram (whole + center)
bool compute_task3_feature(const cv::Mat &img, std::vector<float> &feat);

// Task4: whole-image color hist + whole-image texture hists (mag + ori)
bool compute_task4_feature(const cv::Mat &img, std::vector<float> &feat);

// Task 7: Extract green grass features (simple 5D)
bool extract_grass_features(const cv::Mat &img, std::vector<float> &feat);

#endif // FEATURES_H
