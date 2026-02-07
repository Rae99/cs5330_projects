#ifndef RANKING_H
#define RANKING_H

#include <string>
#include <vector>

struct Match {
    std::string filename;
    float dist;
};

// compute SSD distance between two feature vectors
float ssd_distance(const std::vector<float> &a, const std::vector<float> &b);

// sort matches in-place (ascending by dist)
void sort_matches(std::vector<Match> &matches);

// compute task 2 histogram intersection distance between two feature vectors
float hist_intersection_distance(const std::vector<float> &a,
                                 const std::vector<float> &b);

// compute Task 3 multi-histogram distance
float task3_multi_hist_distance(const std::vector<float> &a,
                                const std::vector<float> &b, float w_whole,
                                float w_center);

float task3_distance(const std::vector<float> &a, const std::vector<float> &b);

float task4_distance(const std::vector<float> &a, const std::vector<float> &b);

float cosine_distance(const std::vector<float> &a, const std::vector<float> &b);

// Task 7: simple distance for grass features (5D)
float grass_distance(const std::vector<float> &a, const std::vector<float> &b);

#endif // RANKING_H
