#include "../include/task_registry.h"

#include <stdexcept>

#include "../include/features.h"
#include "../include/ranking.h"

TaskSpec get_task(int task_id) {
    switch (task_id) {
    case 1:
        return {compute_task1_feature, ssd_distance};
    case 2: return { compute_task2_feature, hist_intersection_distance };
    default:
        throw std::invalid_argument("Unknown task id: " +
                                    std::to_string(task_id));
    }
}
