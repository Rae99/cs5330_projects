#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>
#include <vector>

// Task 1 feature: 7x7 center patch, RGB flattened to length 147
bool compute_task1_feature(const cv::Mat &img, std::vector<float> &feat);

#endif // FEATURES_H
