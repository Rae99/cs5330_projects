/*
    Ding, Junrui
    Februray 2026

    CS5330 Project 2 - ranking.h

    This header declares distance metrics and sorting helpers for ranking
    query matches across tasks.
*/

#ifndef RANKING_H
#define RANKING_H

#include <string>
#include <vector>

/*
    Match

    Simple container storing a database filename and its distance score.
*/
struct Match {
    std::string filename;
    float dist;
};

/*
    ssd_distance

    Compute sum of squared differences between two feature vectors.

    Arguments:
        const std::vector<float> &a - first feature vector.
        const std::vector<float> &b - second feature vector.

    Returns:
        SSD distance, or a large sentinel value on size mismatch.
*/
float ssd_distance(const std::vector<float> &a, const std::vector<float> &b);

/*
    sort_matches

    Sort matches in-place in ascending order by distance.

    Arguments:
        std::vector<Match> &matches - match list to sort.

    Returns:
        void.
*/
void sort_matches(std::vector<Match> &matches);

/*
    hist_intersection_distance

    Compute Task 2 histogram intersection distance between feature vectors.

    Arguments:
        const std::vector<float> &a - first histogram.
        const std::vector<float> &b - second histogram.

    Returns:
        histogram intersection distance, or a large sentinel on size mismatch.
*/
float hist_intersection_distance(const std::vector<float> &a,
                                 const std::vector<float> &b);

/*
    task3_multi_hist_distance

    Compute Task 3 multi-histogram distance with configurable weights.

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
                                float w_center);

/*
    task3_distance

    Convenience wrapper for Task 3 with recommended weights.

    Arguments:
        const std::vector<float> &a - first multi-histogram feature.
        const std::vector<float> &b - second multi-histogram feature.

    Returns:
        weighted multi-histogram distance.
*/
float task3_distance(const std::vector<float> &a, const std::vector<float> &b);

/*
    task4_distance

    Compute Task 4 distance using combined color and texture histograms.

    Arguments:
        const std::vector<float> &a - first Task 4 feature vector.
        const std::vector<float> &b - second Task 4 feature vector.

    Returns:
        distance value, or a large sentinel on size mismatch.
*/
float task4_distance(const std::vector<float> &a, const std::vector<float> &b);

/*
    cosine_distance

    Compute cosine distance (1 - cosine similarity) for Task 5 embeddings.

    Arguments:
        const std::vector<float> &a - first embedding vector.
        const std::vector<float> &b - second embedding vector.

    Returns:
        cosine distance, or a large sentinel on size mismatch.
*/
float cosine_distance(const std::vector<float> &a, const std::vector<float> &b);

/*
    grass_distance

    Compute Task 7 distance for 5D grass features.

    Arguments:
        const std::vector<float> &a - first grass feature vector.
        const std::vector<float> &b - second grass feature vector.

    Returns:
        distance value, or a large sentinel on size mismatch.
*/
float grass_distance(const std::vector<float> &a, const std::vector<float> &b);

#endif // RANKING_H
