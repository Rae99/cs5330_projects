#ifndef TASK_REGISTRY_H
#define TASK_REGISTRY_H

#include <opencv2/opencv.hpp>
#include <vector>

// Feature function: compute feature vector from image
using FeatureFunc = bool (*)(const cv::Mat &img, std::vector<float> &feat);

// Distance function between two feature vectors
using DistFunc = float (*)(const std::vector<float> &a,
                           const std::vector<float> &b);

struct TaskSpec {
    FeatureFunc feature;
    DistFunc dist;
};

// Return TaskSpec for a given task id. Throws std::invalid_argument if unknown.
TaskSpec get_task(int task_id);

#endif // TASK_REGISTRY_H
