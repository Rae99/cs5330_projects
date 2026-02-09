#include "../include/features.h"
#include <algorithm>
#include <opencv2/opencv.hpp>

bool compute_task1_feature(const cv::Mat &img, std::vector<float> &feat) {
    // Check if input image is valid
    // Return false if empty to avoid processing invalid data
    if (img.empty())
        return false;

    // Ensure we always have a 3-channel BGR image
    cv::Mat bgr;
    if (img.channels() == 3)
        bgr = img;
    else if (img.channels() == 1)
        cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
    else
        return false;

    // Image must be at least 7×7 pixels to extract a 7×7 patch around the
    // center
    if (bgr.rows < 7 || bgr.cols < 7)
        return false;

    // Patch centered at (cx, cy) with size 7×7. Each pixel has 3 channels
    // (BGR), so total Patch spans from (cx-3, cy-3) to (cx+3, cy+3), inclusive
    const int cy = bgr.rows / 2;
    const int cx = bgr.cols / 2;
    const int y0 = cy - 3;
    const int x0 = cx - 3;

    // Boundary check: ensure the 7×7 patch is fully within the image. If not,
    // return false.
    if (y0 < 0 || x0 < 0 || y0 + 6 >= bgr.rows || x0 + 6 >= bgr.cols)
        return false;

    feat.clear();
    feat.reserve(147);

    // Iterate through 7×7 patch in row-major order
    // For each pixel: store B, G, R values (0-255 range)
    // Total 7×7×3 = 147 values
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
    // Input validation: image must be non-empty, have at least 3 channels
    // (BGR), and bins must be positive
    if (img.empty() || img.channels() < 3 || bins <= 0)
        return false;

    // Create 2D histogram:
    // Size : bins×bins(default 16×16 = 256 bins total)
    // Type: CV_32FC1(32 - bit float, single channel)
    // Initialized to zeros
    cv::Mat hist = cv::Mat::zeros(cv::Size(bins, bins), CV_32FC1);

    // Pixel iteration and extract R, G, B values
    const int rows = img.rows;
    const int cols = img.cols;

    for (int i = 0; i < rows; i++) {
        const cv::Vec3b *ptr = img.ptr<cv::Vec3b>(i); // BGR row
        for (int j = 0; j < cols; j++) {
            float B = ptr[j][0];
            float G = ptr[j][1];
            float R = ptr[j][2];

            // compute rg chromaticity: r = R/(R+G+B), g = G/(R+G+B)
            // normalize by sum to reduce sensitivity to lighting intensity
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

            // Increment count for bin (rbin, gbin) Each pixel votes for one bin
            // based on its color
            hist.at<float>(rbin, gbin) += 1.0f;
        }
    }

    // normalize to probability (sum = 1) so that comparison is invariant to
    // image size
    hist /= (float)(rows * cols);

    // flatten to feature vector (row-major)
    // Convert 2D histogram (16×16 matrix) to 1D vector (256 elements)
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

// Compute rg chromaticity histogram on a ROI(Region of Interest)
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

    // check if ROI is valid (non-empty)
    if (x1 <= x0 || y1 <= y0)
        return false;

    // Create histogram for ROI (same as whole image, but only for pixels in
    // ROI)
    cv::Mat hist = cv::Mat::zeros(cv::Size(bins, bins), CV_32FC1);

    // iterate ROI and compute histogram (same as whole image, but only for
    // pixels in ROI)
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

    // flatten to output vector
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

    // concatenate:
    // Final feature is [whole_hist(256) || center_hist(256)] = 512-dim vector
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

// Task 7: Extract green grass features (simple 5D)
bool extract_grass_features(const cv::Mat &img, std::vector<float> &feat) {
    if (img.empty())
        return false;

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // Green range for grass
    const int H_LOW = 35;  // 70 degrees
    const int H_HIGH = 85; // 170 degrees
    const int S_MIN = 20;
    const int V_MIN = 20;

    cv::Mat mask(hsv.rows, hsv.cols, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < hsv.rows; ++i) {
        const cv::Vec3b *p = hsv.ptr<cv::Vec3b>(i);
        uchar *mp = mask.ptr<uchar>(i);
        for (int j = 0; j < hsv.cols; ++j) {
            const int H = p[j][0], S = p[j][1], V = p[j][2];
            if (H >= H_LOW && H <= H_HIGH && S >= S_MIN && V >= V_MIN)
                mp[j] = 255;
        }
    }

    // Clean mask
    cv::morphologyEx(
        mask, mask, cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
    cv::morphologyEx(
        mask, mask, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));

    // Compute features
    int green_pixels = cv::countNonZero(mask);
    float green_ratio = green_pixels / (float)(mask.rows * mask.cols);

    // Color statistics in green region
    double sum_h = 0, sum_s = 0, sum_v = 0;
    int cnt = 0;
    for (int i = 0; i < hsv.rows; ++i) {
        const cv::Vec3b *hp = hsv.ptr<cv::Vec3b>(i);
        const uchar *mp = mask.ptr<uchar>(i);
        for (int j = 0; j < hsv.cols; ++j) {
            if (mp[j] > 0) {
                sum_h += hp[j][0];
                sum_s += hp[j][1];
                sum_v += hp[j][2];
                cnt++;
            }
        }
    }

    float avg_h = (cnt > 0) ? (sum_h / cnt) / 179.0f : 0.0f;
    float avg_s = (cnt > 0) ? (sum_s / cnt) / 255.0f : 0.0f;
    float avg_v = (cnt > 0) ? (sum_v / cnt) / 255.0f : 0.0f;

    // 5D feature vector
    feat = {green_ratio, avg_h, avg_s, avg_v, (cnt > 0) ? 1.0f : 0.0f};
    return true;
}