/*
    Ding, Junrui
    Februray 2026

    CS5330 Project 2 - task_registry.h

    This header defines function pointer types and the task registry used
    to map task ids to feature and distance implementations.
*/

#ifndef TASK_REGISTRY_H
#define TASK_REGISTRY_H

#include <opencv2/opencv.hpp>
#include <vector>

/*
    FeatureFunc

    Function pointer type for computing a feature vector from an image.

    Arguments:
        const cv::Mat &img - input image.
        std::vector<float> &feat - output feature vector.

    Returns:
        true on success, false on failure.
*/
using FeatureFunc = bool (*)(const cv::Mat &img, std::vector<float> &feat);

/*
    DistFunc

    Function pointer type for computing distance between feature vectors.

    Arguments:
        const std::vector<float> &a - first feature vector.
        const std::vector<float> &b - second feature vector.

    Returns:
        distance value.
*/
using DistFunc = float (*)(const std::vector<float> &a,
                           const std::vector<float> &b);

/*
    TaskSpec

    Bundle of feature and distance functions for a specific task.
*/
struct TaskSpec {
    FeatureFunc feature;
    DistFunc dist;
};

/*
    get_task

    Return a TaskSpec for a given task id.

    Arguments:
        int task_id - task identifier.

    Returns:
        TaskSpec with feature and distance functions.

    Throws:
        std::invalid_argument if the task id is unknown.
*/
TaskSpec get_task(int task_id);

#endif // TASK_REGISTRY_H
