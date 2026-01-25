/*
  Ding, Junrui
  January 2026

  Include file for filters.cpp
  Custom image processing filters for CS5330 Project 1.

  This header declares a set of filters used by multiple executables:
  - imgDisplay (still image demo)
  - vid        (webcam pipeline demo)
  - timeBlur   (blur timing harness)

  Conventions:
  - Most functions return 0 on success, negative values on error.
  - Unless specified, images are assumed to be BGR (CV_8UC3).
*/

#pragma once
#include <opencv2/opencv.hpp>

/**
 * Custom grayscale filter.
 * Produces a 3-channel grayscale image (each channel identical) so it can
 * remain compatible with later BGR-based effects in the pipeline.
 *
 * @param src Input BGR image (CV_8UC3).
 * @param dst Output grayscale-as-BGR image (CV_8UC3).
 * @return 0 on success, negative on error.
 */
int greyscale(cv::Mat &src, cv::Mat &dst);

/**
 * Sepia tone filter (with vignetting in implementation).
 *
 * @param src Input BGR image (CV_8UC3).
 * @param dst Output sepia-toned image (CV_8UC3).
 * @return 0 on success, negative on error.
 */
int sepia(cv::Mat &src, cv::Mat &dst);

/**
 * 5x5 blur (naive implementation using a full 5x5 kernel).
 *
 * @param src Input BGR image (CV_8UC3).
 * @param dst Output blurred image (CV_8UC3).
 * @return 0 on success, negative on error.
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst);

/**
 * 5x5 blur (separable implementation: horizontal pass + vertical pass).
 *
 * @param src Input BGR image (CV_8UC3).
 * @param dst Output blurred image (CV_8UC3).
 * @return 0 on success, negative on error.
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

/**
 * Sobel X filter (computes horizontal intensity gradient ∂I/∂x).
 * Output is signed 16-bit to preserve negative gradients.
 *
 * @param src Input BGR image (CV_8UC3).
 * @param dst Output signed gradient image (CV_16SC3).
 * @return 0 on success, negative on error.
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

/**
 * Sobel Y filter (computes vertical intensity gradient ∂I/∂y).
 * Output is signed 16-bit to preserve negative gradients.
 *
 * @param src Input BGR image (CV_8UC3).
 * @param dst Output signed gradient image (CV_16SC3).
 * @return 0 on success, negative on error.
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

/**
 * Gradient magnitude image from Sobel X and Sobel Y.
 * Typically computes per-channel magnitude: sqrt(sx^2 + sy^2), then clamps.
 *
 * @param sx Input signed Sobel X image (CV_16SC3).
 * @param sy Input signed Sobel Y image (CV_16SC3).
 * @param dst Output magnitude image (CV_8UC3).
 * @return 0 on success, negative on error.
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

/**
 * Blur + quantize (posterization) filter.
 * Blurs first to reduce noise / small variations, then maps each channel to
 * a fixed number of discrete intensity levels.
 *
 * @param src Input BGR image (CV_8UC3).
 * @param dst Output blurred+quantized image (CV_8UC3).
 * @param levels Number of quantization levels per channel (default 10).
 * @return 0 on success, negative on error.
 */
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels = 10);

/**
 * Depth-based grayscale effect.
 * Uses an 8-bit depth/disparity map to selectively desaturate pixels based
 * on a threshold.
 *
 * @param src Input color image (CV_8UC3).
 * @param depth8 Depth/disparity map (CV_8UC1, 0..255).
 * @param dst Output image (CV_8UC3).
 * @param threshold Depth threshold (interpretation depends on depth8 meaning).
 * @return 0 on success, negative on error.
 */
int depthGrayscale(const cv::Mat &src, const cv::Mat &depth8, cv::Mat &dst,
                   unsigned char threshold);

/**
 * Emboss effect built from Sobel signed gradients.
 * Computes a directional shading term using a dot product between the gradient
 * and a chosen light direction (dirx, diry), then offsets to mid-gray.
 *
 * @param sx16 Signed Sobel X image (CV_16SC3).
 * @param sy16 Signed Sobel Y image (CV_16SC3).
 * @param dst8 Output embossed image (CV_8UC3).
 * @param dirx X component of the light direction in gradient space.
 * @param diry Y component of the light direction in gradient space.
 * @param scale Scaling factor for emboss contrast.
 * @return 0 on success, negative on error.
 */
int embossFromSobel(const cv::Mat &sx16, const cv::Mat &sy16, cv::Mat &dst8,
                    float dirx = 0.7071f, float diry = 0.7071f,
                    float scale = 0.25f);

/**
 * Depth-based fog effect.
 * Blends each pixel with a fog color using an exponential falloff model.
 *
 * @param srcBGR Input color image (CV_8UC3).
 * @param depth8 Depth/disparity map (CV_8UC1, 0..255).
 * @param dstBGR Output fogged image (CV_8UC3).
 * @param k Fog density controlling how fast fog increases with distance.
 */
void applyDepthFog(const cv::Mat &srcBGR, const cv::Mat &depth8,
                   cv::Mat &dstBGR, float k = 2.2f);