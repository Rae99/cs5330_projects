/*
  Ding, Junrui
  January 2026

  CS5330 Project 1 - imgDisplay

  Simple image viewer using OpenCV. Loads a single image from the command line
  and supports basic interactive operations via keyboard shortcuts:
    - rotate, blur, flip, invert, grayscale
    - print image info
    - save image to disk
*/

#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
    // check command-line arguments
    if (argc < 2) {
        printf("Usage: %s <image_file>\n", argv[0]);
        return -1;
    }

    const char *filename = argv[1];

    // read the image (BGR, 8-bit by default)
    cv::Mat src = cv::imread(filename);

    if (src.empty()) {
        printf("Could not open or find the image: %s\n", filename);
        return -1;
    }

    // create window and display image
    cv::namedWindow("Image Display", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image Display", src);

    // event loop
    while (true) {
        char key = (char)cv::waitKey(0);

        if (key == 'q') {
            break;
        }

        else if (key == 'r') { // rotate 90 degrees clockwise
            cv::Mat rotated;
            cv::rotate(src, rotated, cv::ROTATE_90_CLOCKWISE);
            cv::imshow("Image Display", rotated);
        }

        else if (key == 'b') { // blur
            cv::Mat blurred;
            cv::GaussianBlur(src, blurred, cv::Size(15, 15), 0);
            cv::imshow("Image Display", blurred);
        }

        else if (key == 'o') { // original
            cv::imshow("Image Display", src);
        }

        else if (key == 'f') { // flip horizontally
            cv::Mat flipped;
            cv::flip(src, flipped, 1);
            cv::imshow("Image Display", flipped);
        }

        else if (key == 'i') { // invert colors
            cv::Mat inverted;
            cv::bitwise_not(src, inverted);
            cv::imshow("Image Display", inverted);
        }

        else if (key == 'g') { // grayscale
            cv::Mat gray;
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
            cv::imshow("Image Display", gray);
        }

        else if (key == 'd') { // print image info
            printf("Image info: %d x %d, channels=%d\n", src.cols, src.rows,
                   src.channels());
        }

        else if (key == 's') { // save image
            cv::imwrite("output.png", src);
            printf("Saved image as output.png\n");
        }
    }

    cv::destroyAllWindows();
    return 0;
}