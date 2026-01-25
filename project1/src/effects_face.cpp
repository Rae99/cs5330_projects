#include "effects_face.h"
#include "faceDetect.h"
#include <opencv2/opencv.hpp>

void faceColorPop(const cv::Mat &srcBGR, cv::Mat &dstBGR) {
    cv::Mat gray;
    cv::cvtColor(srcBGR, gray, cv::COLOR_BGR2GRAY);

    // Start from grayscale background
    cv::cvtColor(gray, dstBGR, cv::COLOR_GRAY2BGR);

    std::vector<cv::Rect> faces;
    detectFaces(gray, faces);

    // Copy original color back into each face box
    for (int i = 0; i < faces.size(); i++) {
        const cv::Rect &r = faces[i];
        cv::Rect validFaceRect = r & cv::Rect(0, 0, srcBGR.cols, srcBGR.rows);
        if (validFaceRect.area() > 0) {
            srcBGR(validFaceRect).copyTo(dstBGR(validFaceRect));
        }
    }

    // Draw boxes
    drawBoxes(dstBGR, faces, 50, 1.0f);
}