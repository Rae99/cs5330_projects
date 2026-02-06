#include "../include/features.h"
#include <algorithm>
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

// Compute rg chromaticity histogram on a ROI.
// ROI is [x0, x0+w) x [y0, y0+h), clamped to image bounds.
// Output is flattened bins*bins, normalized (sum=1).
static bool compute_rg_hist_roi(const cv::Mat &img, std::vector<float> &out,
                                int bins, int x0, int y0, int w, int h) {
    if (img.empty() || img.channels() < 3 || bins <= 0)
        return false;

    // clamp ROI
    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    int x1 = std::min(img.cols, x0 + w);
    int y1 = std::min(img.rows, y0 + h);

    if (x1 <= x0 || y1 <= y0)
        return false;

    cv::Mat hist = cv::Mat::zeros(cv::Size(bins, bins), CV_32FC1);

    const int roi_rows = y1 - y0;
    const int roi_cols = x1 - x0;

    for (int i = y0; i < y1; i++) {
        const cv::Vec3b *ptr = img.ptr<cv::Vec3b>(i); // BGR row
        for (int j = x0; j < x1; j++) {
            float B = ptr[j][0];
            float G = ptr[j][1];
            float R = ptr[j][2];

            float div = R + G + B;
            if (div <= 0.0f)
                div = 1.0f;

            float r = R / div; // [0,1]
            float g = G / div; // [0,1]

            int rbin = (int)(r * bins);
            int gbin = (int)(g * bins);

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

    // normalize by ROI pixel count
    hist /= (float)(roi_rows * roi_cols);

    out.clear();
    out.reserve(bins * bins);
    for (int r = 0; r < bins; r++) {
        const float *hptr = hist.ptr<float>(r);
        for (int c = 0; c < bins; c++)
            out.push_back(hptr[c]);
    }
    return true;
}

bool compute_task3_feature(const cv::Mat &img, std::vector<float> &feat) {
    // Multi-histogram: whole image + center region, rg chromaticity hist
    if (img.empty() || img.channels() < 3)
        return false;

    const int bins = 16;
    std::vector<float> h_whole;
    std::vector<float> h_center;

    // whole image ROI
    if (!compute_rg_hist_roi(img, h_whole, bins, 0, 0, img.cols, img.rows))
        return false;

    // center ROI: middle 50% x 50%
    const int cw = img.cols / 2;
    const int ch = img.rows / 2;
    const int cx0 = (img.cols - cw) / 2;
    const int cy0 = (img.rows - ch) / 2;

    if (!compute_rg_hist_roi(img, h_center, bins, cx0, cy0, cw, ch))
        return false;

    // concatenate
    feat.clear();
    feat.reserve((int)h_whole.size() + (int)h_center.size());
    feat.insert(feat.end(), h_whole.begin(), h_whole.end());
    feat.insert(feat.end(), h_center.begin(), h_center.end());

    // expected size: 2 * 16 * 16 = 512
    return (feat.size() == (size_t)(2 * bins * bins));
}

// 1D histogram of Sobel gradient magnitudes (normalized, sum=1)
static bool compute_sobel_mag_hist(const cv::Mat &img, std::vector<float> &hist,
                                   int bins) {
    if (img.empty() || bins <= 0)
        return false;

    cv::Mat bgr;
    if (img.channels() == 3)
        bgr = img;
    else if (img.channels() == 1)
        cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
    else
        return false;

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

    cv::Mat mag;
    cv::magnitude(gx, gy, mag);

    double maxv = 0.0;
    cv::minMaxLoc(mag, nullptr, &maxv);
    const float denom = (float)(maxv + 1e-6);

    hist.assign(bins, 0.0f);

    const int rows = mag.rows;
    const int cols = mag.cols;
    for (int i = 0; i < rows; i++) {
        const float *mp = mag.ptr<float>(i);
        for (int j = 0; j < cols; j++) {
            float v = mp[j] / denom; // normalize to [0,1]
            if (v < 0.0f)
                v = 0.0f;
            if (v > 1.0f)
                v = 1.0f;
            int bin = (int)(v * bins);
            if (bin >= bins)
                bin = bins - 1;
            hist[bin] += 1.0f;
        }
    }

    const float total = (float)(rows * cols);
    if (total <= 0.0f)
        return false;
    for (float &x : hist)
        x /= total; // sum=1

    return true;
}

// 1D histogram of Sobel gradient orientations (normalized, sum=1)
// Orientation is "unsigned": [0,180) degrees (so edges with opposite direction
// match)
static bool compute_sobel_ori_hist(const cv::Mat &img, std::vector<float> &hist,
                                   int bins) {
    if (img.empty() || bins <= 0)
        return false;

    cv::Mat bgr;
    if (img.channels() == 3)
        bgr = img;
    else if (img.channels() == 1)
        cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
    else
        return false;

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

    // angle in degrees [0,360)
    cv::Mat ang;
    cv::phase(gx, gy, ang, true);

    hist.assign(bins, 0.0f);

    const int rows = ang.rows;
    const int cols = ang.cols;
    for (int i = 0; i < rows; i++) {
        const float *ap = ang.ptr<float>(i);
        for (int j = 0; j < cols; j++) {
            float a = ap[j]; // [0,360)
            // fold to [0,180)
            if (a >= 180.0f)
                a -= 180.0f;

            // map to bins
            float t = a / 180.0f; // [0,1)
            if (t < 0.0f)
                t = 0.0f;
            if (t >= 1.0f)
                t = std::nextafter(1.0f, 0.0f);

            int bin = (int)(t * bins);
            if (bin >= bins)
                bin = bins - 1;

            hist[bin] += 1.0f;
        }
    }

    const float total = (float)(rows * cols);
    if (total <= 0.0f)
        return false;
    for (float &x : hist)
        x /= total; // sum=1

    return true;
}

// Task 4 feature: whole-image color hist + whole-image texture hists (mag +
// ori) Feature layout: [color(256) || mag(mag_bins) || ori(ori_bins)]
bool compute_task4_feature(const cv::Mat &img, std::vector<float> &feat) {
    if (img.empty())
        return false;

    // Color part (reuse Task2): rg chromaticity 2D hist with 16 bins => 256
    // dims
    std::vector<float> color;
    if (!compute_task2_feature_rg_hist(img, color, 16))
        return false;

    // Texture parts
    const int mag_bins = 16;
    const int ori_bins = 18; // you can change to 16 if you want symmetry

    std::vector<float> mag_hist;
    std::vector<float> ori_hist;

    if (!compute_sobel_mag_hist(img, mag_hist, mag_bins))
        return false;

    if (!compute_sobel_ori_hist(img, ori_hist, ori_bins))
        return false;

    feat.clear();
    feat.reserve(color.size() + mag_hist.size() + ori_hist.size());
    feat.insert(feat.end(), color.begin(), color.end());
    feat.insert(feat.end(), mag_hist.begin(), mag_hist.end());
    feat.insert(feat.end(), ori_hist.begin(), ori_hist.end());

    return true;
}