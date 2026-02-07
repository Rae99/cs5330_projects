#include "../include/ranking.h"

#include <algorithm>
#include <cstddef>

// Task 1: SSD distances
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

// Task 2: histogram intersection distance
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

// Task 3: multi-histogram distance
// Feature format: [H_whole || H_center], each histogram length = bins*bins.
// We compute distance per segment using histogram intersection:
//   D_k = 1 - sum_i min(hq_i, hd_i)
// and combine with weights: D = w0*D_whole + w1*D_center
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

// Convenience wrapper with recommended weights (whole=0.4, center=0.6)
float task3_distance(const std::vector<float> &a, const std::vector<float> &b) {
    return task3_multi_hist_distance(a, b, 0.4f, 0.6f);
}

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

// task 5: cosine distance (1 - cosine similarity)
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

// Task 7: grass feature distance
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