#include "filters.h"
#include <cstdio>

// Greyscale filter
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

        // understanding .ptr<T>(i):
        // .ptr<cv::Vec3b>: Underlying memory is just bytes;
        // <cv::Vec3b> tells the compiler to interpret them as groups of 3.
        // The pointer increment is based on the type cv::Vec3b, so it moves
        // one pixel (3 bytes) at a time

        // return a pointer to cv::Vec3b array to the start of row i
        // the start of row i is at offset i * step bytes from the start of
        // data, start_of_row_i = data + i * step
        // These 3 points to the same memory location:
        // src.ptr<cv::Vec3b>(i) src.ptr<uchar>(i) src.ptr<float>(i)

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

// Sepia tone filter
// add vignetting
int sepia(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        std::printf("sepia(): src is empty\n");
        return -1;
    }

    // allocate dst same size/type as src
    dst.create(src.rows, src.cols, src.type());

    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *sp =
            src.ptr<cv::Vec3b>(i); // source pointer, read-only, so const
        cv::Vec3b *dp = dst.ptr<cv::Vec3b>(i); // destination pointer

        // .ptr<cv::Vec3b>: Underlying memory is just bytes;
        // <cv::Vec3b> tells the compiler to interpret them as groups of 3.

        for (int j = 0; j < src.cols; j++) {
            // BGR
            uchar B = sp[j][0];
            uchar G = sp[j][1];
            uchar R = sp[j][2];

            // Sepia tone calculations
            // red coefficients: 0.393, 0.769, 0.189
            // green coefficients: 0.349, 0.686, 0.168
            // blue coefficients: 0.272, 0.534, 0.131
            uchar tr =
                cv::saturate_cast<uchar>(0.393 * R + 0.769 * G + 0.189 * B);
            uchar tg =
                cv::saturate_cast<uchar>(0.349 * R + 0.686 * G + 0.168 * B);
            uchar tb =
                cv::saturate_cast<uchar>(0.272 * R + 0.534 * G + 0.131 * B);

            // understanding cv::saturate_cast<uchar>:
            // saturate_cast<uchar> ensures the value is within [0, 255]
            // before assigning to uchar
            // if a value is < 0, it becomes 0; if > 255, it becomes 255

            // if we use normal cast,  (uchar)
            // the overflow will cause unexpected results
            // the overflow wraps around using modulo 256 arithmetic

            // apply vignetting weight (linear)
            float cx = (src.cols - 1) * 0.5f; // center x
            float cy = (src.rows - 1) * 0.5f; // center y

            float dx = j - cx; // distance from center x
            float dy = i - cy; // distance from center y
            float r =
                std::sqrt(dx * dx + dy * dy); // distance to center, radius

            // max radius to a corner
            float rmax = std::sqrt(cx * cx + cy * cy);
            float ratio = r / rmax; // 0~1

            float strength = 0.6f;
            float weight = 1.0f - strength * ratio;

            if (weight < 0.0f)
                weight = 0.0f;
            if (weight > 1.0f)
                weight = 1.0f;

            dp[j][0] = (uchar)(tb * weight);
            dp[j][1] = (uchar)(tg * weight);
            dp[j][2] = (uchar)(tr * weight);
        }
    }

    return 0;
}