#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>
#include <vector>

// Task 1 feature: 7x7 center patch, RGB flattened to length 147
bool compute_task1_feature(const cv::Mat &img, std::vector<float> &feat);

// Task2
bool compute_task2_feature(const cv::Mat& img, std::vector<float>& feat); // default bins=16
bool compute_task2_feature_rg_hist(const cv::Mat& img, std::vector<float>& feat, int bins);

#endif // FEATURES_H
