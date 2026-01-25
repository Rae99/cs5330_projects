#include "DA2Network.hpp"
#include "effects_face.h"
#include "faceDetect.h"
#include "filters.h"
#include <cmath>
#include <cstdio>
#include <exception>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;

    // open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return (-1);
    }

    // set up the DA2 network
    bool da2Ready = true;
    DA2Network *da2 = nullptr;
    try {
        // da2 = new DA2Network("../data/model_fp16.onnx");
        da2 = new DA2Network("../data/model_fp16.onnx");
    } catch (const std::exception &e) {
        printf("DA2Network init failed: %s\n", e.what());
        da2Ready = false;
    } catch (...) {
        printf("DA2Network init failed: unknown error\n");
        da2Ready = false;
    }

    // Depth output buffer (8-bit, 1-channel, 0..255)
    cv::Mat depth8;
    // It's a trade between speed vs quality
    float da2ScaleFactor = 0.4f; // smaller = faster but worse depth
    int da2EveryN = 3;           // update depth once every N frames
    int frameCount = 0;          // counter for da2EveryN

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
        MAGNITUDE,   // show gradient magnitude (displayable)
        DEPTH,       // show depth map from DA2 network
        DEPTH_GRAY_EFFECT, // an effect using depth map
        EMBOSS,
        FACE_COLOR_POP,
        DEPTH_FOG
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

    // Extension: video recording
    bool recording = false;
    cv::VideoWriter writer;
    int videoIndex = 0;

    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    double fps = 30.0;
    cv::Size recordSize = refS;

    // read camera FPS
    double camFps = capdev->get(cv::CAP_PROP_FPS);
    if (camFps > 1.0 && camFps < 240.0)
        fps = camFps;
    printf("Recording FPS set to: %.2f\n", fps);

    // press same view key again and go back to ORIGINAL
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

        // If we are using DA2 view/effect, compute depth ONCE per
        // frame depth8 will be CV_8UC1 (1-channel) sized to frame.size()
        frameCount++;
        bool needDA2 =
            (view == ViewMode::DEPTH || view == ViewMode::DEPTH_GRAY_EFFECT ||
             view == ViewMode::DEPTH_FOG);
        bool updateDepthThisFrame = false;
        if (needDA2 && da2Ready && da2 != nullptr) {
            updateDepthThisFrame =
                (depth8.empty() || (frameCount % da2EveryN == 0));
        }

        if (updateDepthThisFrame) {
            // DA2 expects a normal BGR image (CV_8UC3)
            // set_input does optional resizing by scale factor
            da2->set_input(frame, da2ScaleFactor);
            da2->run_network(depth8, frame.size());
        }

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
            // Keep Sobel output (16SC3) and visualization (8UC3) as
            // separate variables
            cv::Mat sobelx16, sobelx8;
            if (sobelX3x3(frame, sobelx16) == 0) {
                cv::convertScaleAbs(sobelx16,
                                    sobelx8); // abs + scale to 8-bit
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
        } else if (view == ViewMode::DEPTH) {
            if (!da2Ready) {
                printf("DA2 network not ready\n");
            } else if (!depth8.empty()) {
                // depth8 is CV_8UC1, convert to 3-channel for display
                // consistency
                cv::cvtColor(depth8, display, cv::COLOR_GRAY2BGR);
            }
        }
        // DA2 DEPTH_GRAY_EFFECT view
        else if (view == ViewMode::DEPTH_GRAY_EFFECT) {
            if (da2Ready && !depth8.empty()) {
                depthGrayscale(frame, depth8, display, 96);
            }
        } else if (view == ViewMode::EMBOSS) {
            cv::Mat sx16, sy16, emboss8;
            // CV_16SC3 outputs from Sobel
            if (sobelX3x3(frame, sx16) == 0 && sobelY3x3(frame, sy16) == 0) {
                if (embossFromSobel(sx16, sy16, emboss8) == 0) {
                    emboss8.copyTo(display);
                }
            }
        } else if (view == ViewMode::FACE_COLOR_POP) {
            cv::Mat tmp;
            faceColorPop(frame, tmp);
            tmp.copyTo(display);
        } else if (view == ViewMode::DEPTH_FOG) {
            if (!depth8.empty()) {
                cv::Mat fogged;
                applyDepthFog(frame, depth8, fogged, 2.2f);
                fogged.copyTo(display);
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
            cv::rotate(display, display,
                       cv::ROTATE_90_COUNTERCLOCKWISE); // 90 degrees
                                                        // counterclockwise
        }

        // Step 5: show
        cv::imshow("Video", display);

        // write frame if recording
        if (recording) {
            // writer expects fixed size + 3-channel BGR
            if (display.size() == recordSize && display.type() == CV_8UC3) {
                writer.write(display);
            } else {
                // if rotation changed the size (e.g., 90 degrees), handle it
                cv::Mat resized;
                cv::resize(display, resized, recordSize);
                if (resized.type() != CV_8UC3) {
                    cv::cvtColor(resized, resized, cv::COLOR_GRAY2BGR);
                }
                writer.write(resized);
            }
        }

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
        if (key == 'd')
            toggleView(ViewMode::DEPTH);
        if (key == 'D') {
            toggleView(ViewMode::DEPTH_GRAY_EFFECT);
            printf("Switched to DEPTH_GRAY_EFFECT\n");
        }
        if (key == 'e')
            toggleView(ViewMode::EMBOSS);
        if (key == 'c')
            toggleView(ViewMode::FACE_COLOR_POP);
        if (key == 'z')
            toggleView(ViewMode::DEPTH_FOG);

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
        // type info
        if (key == 't') {
            int channels = frame.channels();
            int depth = frame.depth(); // CV_8U, CV_16U, etc. (numeric)
            size_t elemSize =
                frame.elemSize(); // bytes per pixel (all channels)
            printf("Frame info: %d x %d, channels=%d, depth=%d, elemSize=%zu "
                   "bytes\n",
                   frame.cols, frame.rows, channels, depth, elemSize);
        }

        if (key == 's') {
            // Save the display(after view + effects + rotation)
            char outname[256];
            std::snprintf(outname, sizeof(outname), "../output/frame_%04d.png",
                          saveIndex++);
            bool ok = cv::imwrite(outname, display);
            if (ok)
                printf("Saved %s\n", outname);
            else
                printf("Failed to save %s\n", outname);
        }

        // toggle video recording
        if (key == 'V') {
            if (!recording) {
                // start recording
                char outname[256];
                std::snprintf(outname, sizeof(outname),
                              "../output/video_%02d.mp4", videoIndex++);

                // recordSize must be constant for writer
                recordSize = refS;

                bool ok = writer.open(outname, fourcc, fps, recordSize, true);
                if (!ok) {
                    printf("Failed to open VideoWriter for %s\n", outname);
                } else {
                    recording = true;
                    printf("Recording START: %s (%.2f fps, %d x %d)\n", outname,
                           fps, recordSize.width, recordSize.height);
                }
            } else {
                // stop recording
                recording = false;
                writer.release();
                printf("Recording STOP\n");
            }
        }
    }

    if (da2) {
        delete da2;
        da2 = nullptr;
    }

    if (recording) {
        recording = false;
        writer.release();
    }

    delete capdev;
    return 0;
}