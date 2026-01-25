/*
  Ding, Junrui
  January 2026

  Face-based visual effects implementation.

  This file implements effects that depend on face detection,
  using OpenCV's Haar cascade via faceDetect.cpp.
*/

#include "effects_face.h"
#include "faceDetect.h"
#include <opencv2/opencv.hpp>

/*
  faceColorPop

  Creates a "face color pop" effect by desaturating the entire image
  to grayscale and then restoring the original color within detected
  face regions.

  The function performs the following steps:
  1. Convert the input image to grayscale.
  2. Convert the grayscale image back to 3-channel BGR so it remains
     compatible with later filters.
  3. Detect faces in the grayscale image.
  4. Copy the original color pixels back into each detected face region.
  5. Draw bounding boxes around detected faces for visualization.

  Arguments:
    const cv::Mat &srcBGR - input color image (CV_8UC3).
    cv::Mat &dstBGR       - output image with face color pop effect applied.
*/
void faceColorPop(const cv::Mat &srcBGR, cv::Mat &dstBGR) {
    cv::Mat gray;
    cv::cvtColor(srcBGR, gray, cv::COLOR_BGR2GRAY);

    // Start from a grayscale background
    cv::cvtColor(gray, dstBGR, cv::COLOR_GRAY2BGR);

    std::vector<cv::Rect> faces;
    detectFaces(gray, faces);

    // Copy original color back into each detected face region
    for (int i = 0; i < faces.size(); i++) {
        const cv::Rect &r = faces[i];

        // Ensure face rectangle lies within image bounds
        cv::Rect validFaceRect = r & cv::Rect(0, 0, srcBGR.cols, srcBGR.rows);

        if (validFaceRect.area() > 0) {
            srcBGR(validFaceRect).copyTo(dstBGR(validFaceRect));
        }
    }

    // Draw face bounding boxes
    drawBoxes(dstBGR, faces, 50, 1.0f);
}