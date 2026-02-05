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
