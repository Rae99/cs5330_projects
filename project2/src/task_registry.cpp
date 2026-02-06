#include "../include/task_registry.h"

#include <stdexcept>

#include "../include/features.h"
#include "../include/ranking.h"

TaskSpec get_task(int task_id) {
    switch (task_id) {
    case 1:
        return {compute_task1_feature, ssd_distance};
    case 2:
        return {compute_task2_feature, hist_intersection_distance};
    case 3:
        return {compute_task3_feature,
                [](const std::vector<float> &a, const std::vector<float> &b) {
                    return task3_multi_hist_distance(a, b, 0.5f, 0.5f);
                }};

    case 4:
        return {compute_task4_feature,
                [](const std::vector<float> &a, const std::vector<float> &b) {
                    return task4_distance(a, b);
                }};
    default:
        throw std::invalid_argument("Unknown task id: " +
                                    std::to_string(task_id));
    }
}
