#include "filters.h"
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
        MODE_SEPIA
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
            cv::GaussianBlur(
                display, display, cv::Size(15, 15),
                0); // ksize stands for kernel size, the larger the more blur;
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
        if (key == 'f')
            mode = MODE_FLIP;
        if (key == 'i')
            mode = MODE_INVERT;
        if (key == 'p')
            mode = MODE_SEPIA;

        // persistent rotation: each press adds 90 degrees clockwise
        if (key == 'r') {
            rotateQuarterTurns = (rotateQuarterTurns + 1) % 4;
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