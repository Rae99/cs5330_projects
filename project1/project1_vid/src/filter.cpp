#include "filters.h"
#include <cstdio>
#include "faceDetect.h"

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

// blur5x5_1: naive 5x5 using at<>
int blur5x5_1(cv::Mat &src, cv::Mat &dst) {

    // kernel weights
    static const int k[5][5] = {{1, 2, 4, 2, 1},
                                {2, 4, 8, 4, 2},
                                {4, 8, 16, 8, 4},
                                {2, 4, 8, 4, 2},
                                {1, 2, 4, 2, 1}};
    const int sum = 100;
    if (src.empty())
        return -1;
    if (src.type() != CV_8UC3) {
        std::printf("blur5x5_1: expected CV_8UC3, got %d\n", src.type());
        return -2;
    }

    // We are doing a valid 5x5 blur, skip 2 pixels on each border
    src.copyTo(dst);

    int rows = src.rows;
    int cols = src.cols;

    // 5x5 kernel sum is 100
    for (int i = 2; i < rows - 2; i++) {
        for (int j = 2; j < cols - 2; j++) {
            int sumB = 0, sumG = 0, sumR = 0;

            for (int di = -2; di <= 2; di++) {
                for (int dj = -2; dj <= 2; dj++) {
                    int w = k[di + 2][dj + 2];
                    cv::Vec3b pix = src.at<cv::Vec3b>(i + di, j + dj);
                    sumB += w * pix[0];
                    sumG += w * pix[1];
                    sumR += w * pix[2];
                }
            }

            dst.at<cv::Vec3b>(i, j)[0] = (uchar)(sumB / sum);
            dst.at<cv::Vec3b>(i, j)[1] = (uchar)(sumG / sum);
            dst.at<cv::Vec3b>(i, j)[2] = (uchar)(sumR / sum);
        }
    }
    return 0;
}

// separable 5x5 blur: (1 2 4 2 1) vertical and horizontal
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    if (src.empty())
        return -1;

    if (src.type() != CV_8UC3) {
        std::printf("blur5x5_2: expected CV_8UC3, got %d\n", src.type());
        return -2;
    }

    const int rows = src.rows;
    const int cols = src.cols;

    // Initialize dst to src so that border pixels already have non-zero values
    src.copyTo(dst);

    // Temporary image to hold intermediate results after horizontal pass
    // Max per channel = 10*255 = 2550
    // 16 bits is enough.
    cv::Mat tmp;
    src.convertTo(tmp, CV_16SC3);
    // equivalent to:
    // tmp.create(src.rows, src.cols, CV_16SC3);
    // for each pixel:
    //     tmp = (short)src;

    // 1D kernel used for both horizontal and vertical passes
    static const int k[5] = {1, 2, 4, 2, 1};

    // horizontal pass
    for (int i = 0; i < rows; i++) {
        // sp points to the first pixel (Vec3b) of row i in src.
        const cv::Vec3b *sp = src.ptr<cv::Vec3b>(i);

        // tp points to the first pixel (Vec3s) of row i in tmp.
        // Vec3s = 3 signed shorts, enough to store intermediate sums like 2550.
        cv::Vec3s *tp = tmp.ptr<cv::Vec3s>(i);

        // Compute horizontal convolution for interior columns.
        for (int j = 2; j < cols - 2; j++) {
            // read 5 neighboring pixels in the same row
            const cv::Vec3b &p0 = sp[j - 2];
            const cv::Vec3b &p1 = sp[j - 1];
            const cv::Vec3b &p2 = sp[j];
            const cv::Vec3b &p3 = sp[j + 1];
            const cv::Vec3b &p4 = sp[j + 2];
            // &p is a reference, like an alias, but not a pointer

            int sumB =
                1 * p0[0] + 2 * p1[0] + 4 * p2[0] + 2 * p3[0] + 1 * p4[0];
            int sumG =
                1 * p0[1] + 2 * p1[1] + 4 * p2[1] + 2 * p3[1] + 1 * p4[1];
            int sumR =
                1 * p0[2] + 2 * p1[2] + 4 * p2[2] + 2 * p3[2] + 1 * p4[2];

            // For now, we don't divide by 10.
            tp[j][0] = (short)sumB;
            tp[j][1] = (short)sumG;
            tp[j][2] = (short)sumR;
        }
    }

    // vertical pass
    for (int i = 2; i < rows - 2; i++) {
        // dp points to row i of dst.
        cv::Vec3b *dp = dst.ptr<cv::Vec3b>(i);

        // 5 row pointers in tmp
        const cv::Vec3s *tp0 = tmp.ptr<cv::Vec3s>(i - 2);
        const cv::Vec3s *tp1 = tmp.ptr<cv::Vec3s>(i - 1);
        const cv::Vec3s *tp2 = tmp.ptr<cv::Vec3s>(i);
        const cv::Vec3s *tp3 = tmp.ptr<cv::Vec3s>(i + 1);
        const cv::Vec3s *tp4 = tmp.ptr<cv::Vec3s>(i + 2);

        for (int j = 2; j < cols - 2; j++) {
            // accumulate weighted sum for each channel at column j
            int sumB = 1 * tp0[j][0] + 2 * tp1[j][0] + 4 * tp2[j][0] +
                       2 * tp3[j][0] + 1 * tp4[j][0];
            int sumG = 1 * tp0[j][1] + 2 * tp1[j][1] + 4 * tp2[j][1] +
                       2 * tp3[j][1] + 1 * tp4[j][1];
            int sumR = 1 * tp0[j][2] + 2 * tp1[j][2] + 4 * tp2[j][2] +
                       2 * tp3[j][2] + 1 * tp4[j][2];

            // total sum = 100
            int outB = sumB / 100;
            int outG = sumG / 100;
            int outR = sumR / 100;

            dp[j][0] = cv::saturate_cast<uchar>(outB);
            dp[j][1] = cv::saturate_cast<uchar>(outG);
            dp[j][2] = cv::saturate_cast<uchar>(outR);
        }
    }

    return 0;
}

// Sobel X: positive to the right
// Separable: vertical smoothing [1 2 1], horizontal derivative [-1 0 1]
// This is a valid convolution.
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        std::printf("sobelX3x3(): src is empty\n");
        return -1;
    }
    if (src.type() != CV_8UC3) {
        std::printf("sobelX3x3(): expected CV_8UC3, got %d\n", src.type());
        return -2;
    }

    // dst must be signed short 3-channel
    dst.create(src.rows, src.cols, CV_16SC3);

    // tmp holds result after vertical smoothing; needs to store values up to
    // 1*255+2*255+1*255=1020
    cv::Mat tmp(src.rows, src.cols, CV_16SC3);

    static const int kSmooth[3] = {1, 2, 1};
    static const int kDeriv[3] = {-1, 0, 1};

    const int rows = src.rows;
    const int cols = src.cols;

    // initialize borders to 0
    tmp.setTo(cv::Scalar(0, 0, 0));
    dst.setTo(cv::Scalar(0, 0, 0));

    // 1) vertical smoothing pass: use neighbors in i direction (i-1, i, i+1)
    for (int i = 1; i < rows - 1; i++) {
        const cv::Vec3b *sp0 = src.ptr<cv::Vec3b>(i - 1);
        const cv::Vec3b *sp1 = src.ptr<cv::Vec3b>(i);
        const cv::Vec3b *sp2 = src.ptr<cv::Vec3b>(i + 1);

        cv::Vec3s *tp = tmp.ptr<cv::Vec3s>(i);

        for (int j = 0; j < cols; j++) {
            // BGR
            tp[j][0] = (short)(kSmooth[0] * sp0[j][0] + kSmooth[1] * sp1[j][0] +
                               kSmooth[2] * sp2[j][0]);
            tp[j][1] = (short)(kSmooth[0] * sp0[j][1] + kSmooth[1] * sp1[j][1] +
                               kSmooth[2] * sp2[j][1]);
            tp[j][2] = (short)(kSmooth[0] * sp0[j][2] + kSmooth[1] * sp1[j][2] +
                               kSmooth[2] * sp2[j][2]);
        }
    }

    // 2) horizontal derivative pass: use neighbors in j direction (j-1, j, j+1)
    for (int i = 1; i < rows - 1; i++) {
        const cv::Vec3s *tp = tmp.ptr<cv::Vec3s>(i);
        cv::Vec3s *dp = dst.ptr<cv::Vec3s>(i);

        for (int j = 1; j < cols - 1; j++) {
            // apply [-1 0 1] on tmp row
            dp[j][0] = (short)(kDeriv[0] * tp[j - 1][0] + kDeriv[1] * tp[j][0] +
                               kDeriv[2] * tp[j + 1][0]);
            dp[j][1] = (short)(kDeriv[0] * tp[j - 1][1] + kDeriv[1] * tp[j][1] +
                               kDeriv[2] * tp[j + 1][1]);
            dp[j][2] = (short)(kDeriv[0] * tp[j - 1][2] + kDeriv[1] * tp[j][2] +
                               kDeriv[2] * tp[j + 1][2]);
        }
    }

    return 0;
}

int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        std::printf("sobelY3x3(): src is empty\n");
        return -1;
    }
    if (src.type() != CV_8UC3) {
        std::printf("sobelY3x3(): expected CV_8UC3, got %d\n", src.type());
        return -2;
    }

    // dst must be signed short 3-channel
    dst.create(src.rows, src.cols, CV_16SC3);

    // tmp holds result after horizontal smoothing; needs to store values up to
    // 1*255+2*255+1*255=1020
    cv::Mat tmp(src.rows, src.cols, CV_16SC3);

    static const int kSmooth[3] = {1, 2, 1};
    static const int kDeriv[3] = {-1, 0, 1};

    const int rows = src.rows;
    const int cols = src.cols;

    // initialize borders to 0
    tmp.setTo(cv::Scalar(0, 0, 0));
    dst.setTo(cv::Scalar(0, 0, 0));

    // 1) horizontal smoothing pass: use neighbors in j direction (j-1, j, j+1)
    for (int i = 0; i < rows; i++) {
        const cv::Vec3b *sp = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *tp = tmp.ptr<cv::Vec3s>(i);

        for (int j = 1; j < cols - 1; j++) {
            // BGR
            tp[j][0] =
                (short)(kSmooth[0] * sp[j - 1][0] + kSmooth[1] * sp[j][0] +
                        kSmooth[2] * sp[j + 1][0]);
            tp[j][1] =
                (short)(kSmooth[0] * sp[j - 1][1] + kSmooth[1] * sp[j][1] +
                        kSmooth[2] * sp[j + 1][1]);
            tp[j][2] =
                (short)(kSmooth[0] * sp[j - 1][2] + kSmooth[1] * sp[j][2] +
                        kSmooth[2] * sp[j + 1][2]);
        }
    }
    // 2) vertical derivative pass: use neighbors in i direction (i-1, i, i+1)
    for (int i = 1; i < rows - 1; i++) {
        const cv::Vec3s *tp0 = tmp.ptr<cv::Vec3s>(i - 1);
        const cv::Vec3s *tp1 = tmp.ptr<cv::Vec3s>(i);
        const cv::Vec3s *tp2 = tmp.ptr<cv::Vec3s>(i + 1);
        cv::Vec3s *dp = dst.ptr<cv::Vec3s>(i);

        for (int j = 0; j < cols; j++) {
            // apply [-1 0 1] on tmp column
            dp[j][0] = (short)(kDeriv[0] * tp0[j][0] + kDeriv[1] * tp1[j][0] +
                               kDeriv[2] * tp2[j][0]);
            dp[j][1] = (short)(kDeriv[0] * tp0[j][1] + kDeriv[1] * tp1[j][1] +
                               kDeriv[2] * tp2[j][1]);
            dp[j][2] = (short)(kDeriv[0] * tp0[j][2] + kDeriv[1] * tp1[j][2] +
                               kDeriv[2] * tp2[j][2]);
        }
    }
    return 0;
}

// magnitude: combine sx and sy to magnitude image
// sx and sy are CV_16SC3 (from Sobel), dst is CV_8UC3
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    if (sx.empty() || sy.empty()) {
        std::printf("magnitude(): sx or sy is empty\n");
        return -1;
    }
    if (sx.type() != CV_16SC3 || sy.type() != CV_16SC3) {
        std::printf("magnitude(): expected CV_16SC3 inputs, got sx=%d sy=%d\n",
                    sx.type(), sy.type());
        return -2;
    }
    if (sx.rows != sy.rows || sx.cols != sy.cols) {
        std::printf("magnitude(): size mismatch\n");
        return -3;
    }

    dst.create(sx.rows, sx.cols, CV_8UC3);

    const int rows = sx.rows;
    const int cols = sx.cols;

    for (int i = 0; i < rows; i++) {
        const cv::Vec3s *spx = sx.ptr<cv::Vec3s>(i);
        const cv::Vec3s *spy = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b *dp = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < cols; j++) {
            // per-channel magnitude (B,G,R)
            for (int c = 0; c < 3; c++) {
                // gradient in x-direction and y-direction
                int gx = spx[j][c];
                int gy = spy[j][c];

                // sqrt(gx*gx + gy*gy)
                // use float for sqrt, then clamp to uchar
                float mag = std::sqrt((float)gx * gx + (float)gy * gy);

                dp[j][c] = cv::saturate_cast<uchar>(mag);
            }
        }
    }

    return 0;
}

int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    if (src.empty()) {
        std::printf("blurQuantize(): src is empty\n");
        return -1;
    }
    if (src.type() != CV_8UC3) {
        std::printf("blurQuantize(): expected CV_8UC3, got %d\n", src.type());
        return -2;
    }
    if (levels <= 0) {
        std::printf("blurQuantize(): levels must be > 0\n");
        return -3;
    }
    if (levels > 255) {
        std::printf("blurQuantize(): levels too large (>%d)\n", 255);
        return -4;
    }

    // 1) blur
    cv::Mat blurred;
    int rc = blur5x5_2(src, blurred);
    if (rc != 0)
        return rc;

    // 2) quantize
    dst.create(src.rows, src.cols, src.type());

    int b = 255 / levels; // bucket size

    for (int i = 0; i < blurred.rows; i++) {
        const cv::Vec3b *sp = blurred.ptr<cv::Vec3b>(i);
        cv::Vec3b *dp = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < blurred.cols; j++) {
            // BGR channels
            for (int c = 0; c < 3; c++) {
                int x = sp[j][c]; // 0..255
                int xt = x / b;   // bucket index
                int xf = xt * b;  // bucket representative value: lower bound
                dp[j][c] = (uchar)xf; // still in 0..255
            }
        }
    }

    return 0;
}