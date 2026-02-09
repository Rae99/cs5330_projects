/*
    Ding, Junrui
    Februray 2026

    CS5330 Project 2 - ranking.cpp

    This file implements distance metrics and ranking utilities for
    comparing feature vectors across tasks.
*/

#include "../include/ranking.h"

#include <algorithm>
#include <cstddef>

/*
    ssd_distance

    Compute sum of squared differences between two vectors.

    Arguments:
        const std::vector<float> &a - first feature vector.
        const std::vector<float> &b - second feature vector.

    Returns:
        SSD distance, or a large sentinel value on size mismatch.
*/
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

/*
    sort_matches

    Sort matches in ascending order by distance.

    Arguments:
        std::vector<Match> &matches - match list to sort.

    Returns:
        void.
*/
void sort_matches(std::vector<Match> &matches) {
    std::sort(
        matches.begin(), matches.end(),
        [](const Match &m1, const Match &m2) { return m1.dist < m2.dist; });
}

/*
    hist_intersection_distance

    Compute histogram intersection distance for Task 2 features.

    Arguments:
        const std::vector<float> &a - first histogram.
        const std::vector<float> &b - second histogram.

    Returns:
        histogram intersection distance, or a large sentinel on size mismatch.
*/
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

/*
    task3_multi_hist_distance

    Compute Task 3 multi-histogram distance using weighted intersection
    over whole-image and center-region histograms.

    Arguments:
        const std::vector<float> &a - first multi-histogram feature.
        const std::vector<float> &b - second multi-histogram feature.
        float w_whole - weight for whole-image histogram distance.
        float w_center - weight for center-region histogram distance.

    Returns:
        weighted multi-histogram distance.
*/
float task3_multi_hist_distance(const std::vector<float> &a,
                                const std::vector<float> &b, float w_whole,
                                float w_center) {
    if (a.size() != b.size())
        return 1e30f;

    // Expect exactly two histograms concatenated
    if (a.size() % 2 != 0)
        return 1e30f;

    const size_t seg_len = a.size() / 2;

    // segment 0: whole
    double sim0 = 0.0;
    for (size_t i = 0; i < seg_len; i++)
        sim0 += std::min((double)a[i], (double)b[i]);
    double d0 = 1.0 - sim0;

    // segment 1: center
    double sim1 = 0.0;
    for (size_t i = 0; i < seg_len; i++)
        sim1 += std::min((double)a[seg_len + i], (double)b[seg_len + i]);
    double d1 = 1.0 - sim1;

    // weighted combination
    double d = (double)w_whole * d0 + (double)w_center * d1;
    return (float)d;
}

/*
    task3_distance

    Convenience wrapper for Task 3 with recommended weights.

    Arguments:
        const std::vector<float> &a - first multi-histogram feature.
        const std::vector<float> &b - second multi-histogram feature.

    Returns:
        weighted multi-histogram distance.
*/
float task3_distance(const std::vector<float> &a, const std::vector<float> &b) {
    return task3_multi_hist_distance(a, b, 0.4f, 0.6f);
}

/*
    task4_distance

    Compute Task 4 distance using combined color and texture histograms.

    Arguments:
        const std::vector<float> &a - first Task 4 feature vector.
        const std::vector<float> &b - second Task 4 feature vector.

    Returns:
        distance value, or a large sentinel on size mismatch.
*/
float task4_distance(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size())
        return 1e30f;

    const size_t color_dim = 16 * 16; // 256 (rg 2D hist)
    const size_t mag_dim = 16;        // must match mag_bins
    const size_t ori_dim = 18;        // must match ori_bins

    if (a.size() != color_dim + mag_dim + ori_dim)
        return 1e30f;

    // color intersection distance
    double sim_c = 0.0;
    for (size_t i = 0; i < color_dim; i++)
        sim_c += std::min((double)a[i], (double)b[i]);
    double d_color = 1.0 - sim_c;

    // magnitude intersection distance
    const size_t mag_off = color_dim;
    double sim_m = 0.0;
    for (size_t i = 0; i < mag_dim; i++)
        sim_m += std::min((double)a[mag_off + i], (double)b[mag_off + i]);
    double d_mag = 1.0 - sim_m;

    // orientation intersection distance
    const size_t ori_off = color_dim + mag_dim;
    double sim_o = 0.0;
    for (size_t i = 0; i < ori_dim; i++)
        sim_o += std::min((double)a[ori_off + i], (double)b[ori_off + i]);
    double d_ori = 1.0 - sim_o;

    // texture distance: equal weight between magnitude and orientation
    double d_tex = 0.5 * d_mag + 0.5 * d_ori;

    // Task4 requirement: equal weight between color and texture
    double d = 0.5 * d_color + 0.5 * d_tex;
    return (float)d;
}

/*
    cosine_distance

    Compute cosine distance (1 - cosine similarity) for Task 5 embeddings.

    Arguments:
        const std::vector<float> &a - first embedding vector.
        const std::vector<float> &b - second embedding vector.

    Returns:
        cosine distance, or a large sentinel on size mismatch.
*/
float cosine_distance(const std::vector<float> &a,
                      const std::vector<float> &b) {
    if (a.size() != b.size() || a.empty())
        return 1e30f;

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double x = (double)a[i];
        const double y = (double)b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }

    const double denom = std::sqrt(na) * std::sqrt(nb);
    if (denom <= 1e-12)
        return 1e30f;

    const double cosv = dot / denom; // in [-1,1] typically
    return (float)(1.0 - cosv);      // cosine distance
}

/*
    grass_distance

    Compute Task 7 distance for 5D grass feature vectors.

    Arguments:
        const std::vector<float> &a - first grass feature vector.
        const std::vector<float> &b - second grass feature vector.

    Returns:
        distance value, or a large sentinel on size mismatch.
*/
float grass_distance(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != 5 || b.size() != 5)
        return 1e30f;

    float d = 0.0f;
    d += 2.0f * (a[0] - b[0]) * (a[0] - b[0]); // green_ratio (important!)
    d += 5.0f * (a[1] - b[1]) * (a[1] - b[1]); // H (color, very important!)
    d += 3.0f * (a[2] - b[2]) * (a[2] - b[2]); // S
    d += 1.0f * (a[3] - b[3]) * (a[3] - b[3]); // V
    d += 0.5f * (a[4] - b[4]) * (a[4] - b[4]); // has_green flag

    return std::sqrt(d);
}