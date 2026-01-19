#include "filters.h"
#include <cstdio>

int greyscale(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        std::printf("greyscale(): src is empty\n");
        return -1;
    }

    // allocate dst same size/type as src
    dst.create(src.rows, src.cols, src.type());
    // create() will not reallocate if already correct size/type

    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *sp =
            src.ptr<cv::Vec3b>(i); // source pointer, read-only, so const
        cv::Vec3b *dp = dst.ptr<cv::Vec3b>(i); // destination pointer

        for (int j = 0; j < src.cols; j++) {
            // BGR
            uchar B = sp[j][0];
            uchar G = sp[j][1];
            uchar R = sp[j][2];

            // Alternative grayscale idea: invert the red channel
            uchar gray = (uchar)(255 - R);

            dp[j][0] = gray;
            dp[j][1] = gray;
            dp[j][2] = gray;
        }
    }

    return 0;
}