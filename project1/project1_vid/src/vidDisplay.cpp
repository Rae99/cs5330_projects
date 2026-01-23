
#include "faceDetect.h"
#include "filters.h"
#include <cmath>
#include <cstdio>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev; // CV_8UC3 = 3 channels of 8-bit unsigned

    // open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return (-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // identifies a window

    cv::Mat frame;
    cv::Mat display;

    enum Mode {
        MODE_COLOR,
        MODE_GRAY,
        MODE_CUSTOM_GRAY,
        MODE_BLUR,
        MODE_FLIP,
        MODE_INVERT,
        MODE_SEPIA,
        MODE_SOBEL_X,
        MODE_SOBEL_Y,
        MODE_MAGNITUDE,
        MODE_QUANTIZE,
        MODE_FACE_DETECT
    };
    Mode mode = MODE_COLOR;

    int rotateQuarterTurns = 0; // 0,1,2,3 => 0/90/180/270 degrees
    int saveIndex = 0;

    for (;;) {
        *capdev >> frame; // get a new frame from the camera
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        // Step 1: start from the current frame (copy)
        frame.copyTo(display);

        // Step 2: apply the "mode" transformation (persistent)
        if (mode == MODE_GRAY) {
            // BGR -> Gray (1 channel), CV_8UC1
            cv::cvtColor(display, display,
                         cv::COLOR_BGR2GRAY); // cvtcolor can do in-place, it
                                              // will use a temporary buffer
        } else if (mode == MODE_CUSTOM_GRAY) {
            // My custom grayscale implementation (output is 3-channel)
            cv::Mat gray3ch;
            if (greyscale(display, gray3ch) == 0) {
                gray3ch.copyTo(display);
            }
        } else if (mode == MODE_BLUR) {
            // My custom 5x5 blur implementation
            cv::Mat blurred;
            if (blur5x5_2(display, blurred) == 0) {
                blurred.copyTo(display);
            };

            // test that the blur does work
            // capdev->set(cv::CAP_PROP_FRAME_WIDTH, 640);
            // capdev->set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            // blur5x5_1(display, blurred);
            // blur5x5_1(blurred, display);

            // built-in Gaussian blur (commented out)
            // cv::GaussianBlur(
            //     display, display, cv::Size(15, 15),
            //     0); // ksize stands for kernel size, the larger the more
            // blur;
            // BORDER_DEFAULT is default, = BORDER_REFLECT_101
        } else if (mode == MODE_FLIP) {
            cv::flip(display, display,
                     1); // 1 = horizontal flip; 0 = vertical; -1 = both
        } else if (mode == MODE_INVERT) {
            cv::bitwise_not(display, display);
        } else if (mode == MODE_SEPIA) {
            cv::Mat sepiaImg;
            if (sepia(display, sepiaImg) == 0) {
                sepiaImg.copyTo(display);
            }
        } else if (mode == MODE_SOBEL_X) {
            cv::Mat sobelx16;
            cv::Mat sobelx8;
            if (sobelX3x3(frame, sobelx16) == 0) {
                cv::convertScaleAbs(sobelx16, sobelx8);
                sobelx8.copyTo(display);
            }

        } else if (mode == MODE_SOBEL_Y) {
            cv::Mat sobely16;
            cv::Mat sobely8;
            if (sobelY3x3(frame, sobely16) == 0) {
                cv::convertScaleAbs(sobely16, sobely8);
                sobely8.copyTo(display);
            }
        } else if (mode == MODE_MAGNITUDE) {
            // compute Sobel X and Sobel Y, then combine to magnitude
            cv::Mat sobelx16, sobely16, magnitudeImg;
            if (sobelX3x3(frame, sobelx16) == 0 &&
                sobelY3x3(frame, sobely16) == 0) {
                if (magnitude(sobelx16, sobely16, magnitudeImg) == 0) {
                    magnitudeImg.copyTo(display);
                }
            }
        } else if (mode == MODE_QUANTIZE) {
            cv::Mat quantized;
            if (blurQuantize(frame, quantized, 10) == 0) {
                quantized.copyTo(display);
            }
        } else if (mode == MODE_FACE_DETECT) {

            cv::Mat grey;
            cv::cvtColor(display, grey, cv::COLOR_BGR2GRAY);

            std::vector<cv::Rect> faces;
            detectFaces(grey, faces);
            drawBoxes(display, faces, 50, 1.0f);
        }

        // Step 3: apply rotation (persistent, accumulative)
        // rotateQuarterTurns can be 0..3
        if (rotateQuarterTurns == 1) {
            cv::rotate(display, display,
                       cv::ROTATE_90_CLOCKWISE); // 90 degrees clockwise
        } else if (rotateQuarterTurns == 2) {
            cv::rotate(display, display,
                       cv::ROTATE_180); // 180 degrees
        } else if (rotateQuarterTurns == 3) {
            cv::rotate(display, display,
                       cv::ROTATE_90_COUNTERCLOCKWISE); // 90 degrees
                                                        // counterclockwise
        }

        // Step 4: show
        cv::imshow("Video", display);

        // Step 5: key handling
        char key = (char)cv::waitKey(10);

        if (key == 'q')
            break;

        // persistent modes
        if (key == 'c')
            mode = MODE_COLOR;
        if (key == 'g')
            mode = MODE_GRAY;
        if (key == 'h')
            mode = MODE_CUSTOM_GRAY; // custom grayscale (3-channel)
        if (key == 'b')
            mode = MODE_BLUR;
        if (key == 'F')
            mode = MODE_FLIP;
        if (key == 'v')
            mode = MODE_INVERT;
        if (key == 'p')
            mode = MODE_SEPIA;
        if (key == 'x')
            mode = MODE_SOBEL_X;
        if (key == 'y')
            mode = MODE_SOBEL_Y;
        if (key == 'm')
            mode = MODE_MAGNITUDE;
        // persistent rotation: each press adds 90 degrees clockwise
        if (key == 'r') {
            rotateQuarterTurns = (rotateQuarterTurns + 1) % 4;
        }
        if (key == 'i') {
            mode = MODE_QUANTIZE;
        }
        if (key == 'f') {
            mode = MODE_FACE_DETECT;
        }

        // one-shot actions (do not change mode)
        if (key == 'd') {
            int channels = frame.channels();
            int depth = frame.depth(); // CV_8U, CV_16U, etc. (numeric)
            size_t elemSize =
                frame.elemSize(); // bytes per pixel (all channels)
            printf("Frame info: %d x %d, channels=%d, depth=%d, elemSize=%zu "
                   "bytes\n",
                   frame.cols, frame.rows, channels, depth, elemSize);
        }

        if (key == 's') {
            // Save what you're currently displaying (after mode + rotation)
            char outname[256];
            std::snprintf(outname, sizeof(outname), "frame_%04d.png",
                          saveIndex++);
            bool ok = cv::imwrite(outname, display);
            if (ok)
                printf("Saved %s\n", outname);
            else
                printf("Failed to save %s\n", outname);
        }
    }

    delete capdev;
    return 0;
}
