#include "faceDetect.h"
#include "filters.h"
#include <cmath>
#include <cstdio>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;

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

    // View modes (mutually exclusive) press again to toggle back to ORIGINAL
    enum class ViewMode {
        ORIGINAL,    // show original color frame
        GRAY,        // show grayscale (but converted back to 3-ch for pipeline)
        CUSTOM_GRAY, // custom greyscale() output (3-ch)
        SOBEL_X,     // show X sobel (abs, displayable)
        SOBEL_Y,     // show Y sobel (abs, displayable)
        MAGNITUDE    // show gradient magnitude (displayable)
    };
    ViewMode view = ViewMode::ORIGINAL;

    // Effects (stackable toggles) press to toggle on/off
    bool blurOn = false;     // 'b'
    bool flipOn = false;     // 'F'
    bool invertOn = false;   // 'v'
    bool sepiaOn = false;    // 'p'
    bool quantizeOn = false; // 'i'
    bool faceOn = false;     // 'f'

    // rotation: persistent, each press adds 90 degrees clockwise
    int rotateQuarterTurns = 0; // 0,1,2,3 => 0/90/180/270 degrees

    int saveIndex = 0;

    // Helper: press same view key again and go back to ORIGINAL
    auto toggleView = [&](ViewMode chosen) {
        view = (view == chosen) ? ViewMode::ORIGINAL : chosen;
    };

    for (;;) {
        *capdev >> frame; // get a new frame from the camera
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        // Step 1: start from the current frame (copy)
        frame.copyTo(display);

        // Step 2: apply the view(mutually exclusive)
        // Note: keep 'display' as 3-channel most of the time so the later
        // effects (sepia/blur/etc.) can always run without type errors.
        if (view == ViewMode::GRAY) {
            // BGR -> Gray (1 channel) then back to BGR (3 channel),
            // so the rest of the pipeline can still treat it as CV_8UC3.
            cv::Mat gray1;
            cv::cvtColor(display, gray1, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray1, display, cv::COLOR_GRAY2BGR);
        } else if (view == ViewMode::CUSTOM_GRAY) {
            // My custom grayscale implementation (output is 3-channel)
            cv::Mat gray3ch;
            if (greyscale(display, gray3ch) == 0) {
                gray3ch.copyTo(display);
            }
        } else if (view == ViewMode::SOBEL_X) {
            // Keep Sobel output (16SC3) and visualization (8UC3) as separate
            // variables
            cv::Mat sobelx16, sobelx8;
            if (sobelX3x3(frame, sobelx16) == 0) {
                cv::convertScaleAbs(sobelx16, sobelx8); // abs + scale to 8-bit
                sobelx8.copyTo(display);
            }
        } else if (view == ViewMode::SOBEL_Y) {
            cv::Mat sobely16, sobely8;
            if (sobelY3x3(frame, sobely16) == 0) {
                cv::convertScaleAbs(sobely16, sobely8);
                sobely8.copyTo(display);
            }
        } else if (view == ViewMode::MAGNITUDE) {
            // compute Sobel X and Sobel Y, then combine to magnitude
            cv::Mat sobelx16, sobely16, mag8;
            if (sobelX3x3(frame, sobelx16) == 0 &&
                sobelY3x3(frame, sobely16) == 0) {
                if (magnitude(sobelx16, sobely16, mag8) == 0) {
                    mag8.copyTo(display); // mag8 should be 8UC3 for display
                }
            }
        }

        // Step 3: apply accumulative EFFECTS
        if (blurOn) {
            // My custom 5x5 blur implementation
            cv::Mat blurred;
            if (blur5x5_2(display, blurred) == 0) {
                blurred.copyTo(display);
            }
        }

        if (quantizeOn) {
            cv::Mat quantized;
            if (blurQuantize(display, quantized, 10) == 0) {
                quantized.copyTo(display);
            }
        }

        if (invertOn) {
            cv::bitwise_not(display, display);
        }

        if (sepiaOn) {
            cv::Mat sepiaImg;
            if (sepia(display, sepiaImg) == 0) {
                sepiaImg.copyTo(display);
            }
        }

        if (flipOn) {
            cv::flip(display, display,
                     1); // 1 = horizontal flip; 0 = vertical; -1 = both
        }

        if (faceOn) {
            cv::Mat grey;
            cv::cvtColor(display, grey, cv::COLOR_BGR2GRAY);

            std::vector<cv::Rect> faces;
            detectFaces(grey, faces);
            drawBoxes(display, faces, 50, 1.0f); // minWidth=50, scale=1.0
        }

        // Step 4: apply rotation (persistent, accumulative)
        // rotateQuarterTurns can be 0..3
        if (rotateQuarterTurns == 1) {
            cv::rotate(display, display,
                       cv::ROTATE_90_CLOCKWISE); // 90 degrees clockwise
        } else if (rotateQuarterTurns == 2) {
            cv::rotate(display, display, cv::ROTATE_180); // 180 degrees
        } else if (rotateQuarterTurns == 3) {
            cv::rotate(
                display, display,
                cv::ROTATE_90_COUNTERCLOCKWISE); // 90 degrees counterclockwise
        }

        // Step 5: show
        cv::imshow("Video", display);

        // Step 6: key handling
        char key = (char)cv::waitKey(10);

        if (key == 'q')
            break;

        // View toggles (press again to cancel back to ORIGINAL)
        if (key == 'o') {
            view = ViewMode::ORIGINAL;
        }
        if (key == 'g')
            toggleView(ViewMode::GRAY);
        if (key == 'h')
            toggleView(ViewMode::CUSTOM_GRAY);
        if (key == 'x')
            toggleView(ViewMode::SOBEL_X);
        if (key == 'y')
            toggleView(ViewMode::SOBEL_Y);
        if (key == 'm')
            toggleView(ViewMode::MAGNITUDE);

        // Effect toggles (press again to turn off)
        if (key == 'b')
            blurOn = !blurOn;
        if (key == 'F')
            flipOn = !flipOn;
        if (key == 'v')
            invertOn = !invertOn;
        if (key == 'p')
            sepiaOn = !sepiaOn;
        if (key == 'i')
            quantizeOn = !quantizeOn;
        if (key == 'f')
            faceOn = !faceOn;

        // Rotation: each press adds 90 degrees clockwise
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
            // Save the display(after view + effects +
            // rotation)
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