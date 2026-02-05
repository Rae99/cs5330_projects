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

#endif // RANKING_H
