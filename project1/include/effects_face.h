/*
  Ding, Junrui
  January 2026

  Include file for effects_face.cpp
  Face-based color effects
*/

#ifndef EFFECTS_FACE_H
#define EFFECTS_FACE_H

#include <opencv2/opencv.hpp>

// Face color pop effect
// Keeps detected face regions in color while desaturating the background
void faceColorPop(const cv::Mat &srcBGR, cv::Mat &dstBGR);

#endif