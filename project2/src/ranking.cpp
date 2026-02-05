#include "../include/ranking.h"

#include <algorithm>
#include <cstddef>

float ssd_distance(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size())
        return 1e30f;
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s += d * d;
    }
    return static_cast<float>(s);
}

void sort_matches(std::vector<Match> &matches) {
    std::sort(
        matches.begin(), matches.end(),
        [](const Match &m1, const Match &m2) { return m1.dist < m2.dist; });
}

float hist_intersection_distance(const std::vector<float> &a,
                                 const std::vector<float> &b) {
    if (a.size() != b.size())
        return 1e30f;
    double sim = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        sim += std::min((double)a[i], (double)b[i]);
    }
    // distance: smaller = more similar
    return (float)(1.0 - sim);
}
