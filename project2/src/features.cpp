#include "../include/features.h"

#include <opencv2/opencv.hpp>

bool compute_task1_feature(const cv::Mat &img, std::vector<float> &feat) {
    if (img.empty())
        return false;

    cv::Mat bgr;
    if (img.channels() == 3)
        bgr = img;
    else if (img.channels() == 1)
        cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
    else
        return false;

    if (bgr.rows < 7 || bgr.cols < 7)
        return false;

    const int cy = bgr.rows / 2;
    const int cx = bgr.cols / 2;
    const int y0 = cy - 3;
    const int x0 = cx - 3;

    if (y0 < 0 || x0 < 0 || y0 + 6 >= bgr.rows || x0 + 6 >= bgr.cols)
        return false;

    feat.clear();
    feat.reserve(147);

    for (int y = y0; y < y0 + 7; y++) {
        const cv::Vec3b *row = bgr.ptr<cv::Vec3b>(y);
        for (int x = x0; x < x0 + 7; x++) {
            const cv::Vec3b &p = row[x];
            feat.push_back(static_cast<float>(p[0]));
            feat.push_back(static_cast<float>(p[1]));
            feat.push_back(static_cast<float>(p[2]));
        }
    }

    return (feat.size() == 147);
}

bool compute_task2_feature_rg_hist(const cv::Mat &img, std::vector<float> &feat,
                                   int bins) {
    if (img.empty() || img.channels() < 3 || bins <= 0)
        return false;

    // 2D hist bins x bins, float counts
    cv::Mat hist = cv::Mat::zeros(cv::Size(bins, bins), CV_32FC1);

    const int rows = img.rows;
    const int cols = img.cols;

    for (int i = 0; i < rows; i++) {
        const cv::Vec3b *ptr = img.ptr<cv::Vec3b>(i); // BGR row
        for (int j = 0; j < cols; j++) {
            float B = ptr[j][0];
            float G = ptr[j][1];
            float R = ptr[j][2];

            float div = R + G + B;
            if (div <= 0.0f)
                div = 1.0f;

            float r = R / div; // [0,1]
            float g = G / div; // [0,1]

            // map to bin index: 0..bins-1
            int rbin = (int)(r * bins);
            int gbin = (int)(g * bins);

            // clamp (avoid edge case r==1)
            if (rbin >= bins)
                rbin = bins - 1;
            if (gbin >= bins)
                gbin = bins - 1;
            if (rbin < 0)
                rbin = 0;
            if (gbin < 0)
                gbin = 0;

            hist.at<float>(rbin, gbin) += 1.0f;
        }
    }

    // normalize to probability (sum = 1)
    hist /= (float)(rows * cols);

    // flatten to feature vector (row-major)
    feat.clear();
    feat.reserve(bins * bins);
    for (int r = 0; r < bins; r++) {
        const float *hptr = hist.ptr<float>(r);
        for (int c = 0; c < bins; c++)
            feat.push_back(hptr[c]);
    }
    return true;
}

bool compute_task2_feature(const cv::Mat &img, std::vector<float> &feat) {
    // default to 16 bins
    return compute_task2_feature_rg_hist(img, feat, 16);
}