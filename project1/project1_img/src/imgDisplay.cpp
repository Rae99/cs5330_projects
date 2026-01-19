/*
 Created by Junrui Ding on 2026-01-17.
 Purpose: Read an image from a file and display it
*/

#include <cstdio> // standard io functions and allocation
#include <cstring>
#include <cstdlib>
#include <opencv2/opencv.hpp> // OpenCV main header

void imgDisplay(const char* filename) {
    cv::Mat src; // Matrix to hold the image data

    // read the image
    // by default, imread converts the image to a 8 bit/channel image
    // use cv::IMREAD_UNCHANGED to read the image as is if needed
    // iread by default reads the image in BGR color format(interleaved)
    src = cv::imread(filename);
    
    // check if the image has been loaded successfully
    if (src.empty()) {
        printf("Could not open or find the image: %s\n", filename);
        exit(-1);
    }

     // display the original image
    cv::imshow(filename, src );
    
    // enter a loop, checking for a keypress. If the user types 'q', the program should quit. Feel free to add other functionality to your program by detecting other keypresses. 
    while (true){
        char key = (char) cv::waitKey(0); // wait for a keypress
        if (key == 'q'){ // if the user types 'q', quit the program
            break;
        }
        if(key == 'r'){ // if the user types 'r', rotate the image 90 degrees clockwise
            cv::Mat rotated;
            cv::rotate(src, rotated, cv::ROTATE_90_CLOCKWISE);
            cv::imshow("Rotated Image", rotated); //imshow(windowName, imageMat)
        }
        if (key == 'b'){ // if the user types 'b', blur the image
            cv::Mat blurred;
            cv::GaussianBlur(src, blurred, cv::Size(15, 15), 0);
            cv::imshow("Blurred Image", blurred);
        }
        if (key == 'o'){ // if the user types 'o', show original image
            cv::imshow("Original Image", src);
        }
        if (key == 'f'){ // if the user types 'f', flip the image horizontally
            cv::Mat flipped;
            cv::flip(src, flipped, 1); // flip the image horizontally
            cv::imshow("Flipped Image", flipped);
        }
        if (key == 'i'){ // if the user types 'i', invert the image colors
            cv::Mat inverted;
            cv::bitwise_not(src, inverted);
            cv::imshow("Inverted Image", inverted);
        }
        if (key == 'g'){ // if the user types 'g', convert the image to grayscale
            cv::Mat gray;
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
            cv::imshow("Grayscale Image", gray);
        }
        if (key == 'd') { // if the user types 'd', display image dimensions
            printf("Image dimensions: %d rows x %d cols x %d channels\n", src.rows, src.cols, src.channels());
        }
        if (key == 's'){ // if the user types 's', save the image
            cv::imwrite("output.png", src);
            printf("Image saved as output.png\n");
        }
    }
}

    


    // // print out some information about the image
    // printf( "filename:        %s\n", filename);
    // printf( "Image size:      %d rows x %d cols\n", (int)src.size().height, (int)src.size().width);  // also use src.rows, src.cols
    // printf( "Image size:      %d rows x %d cols\n", src.rows, src.cols );
    // printf( "Image dimensions %d\n", (int)src.channels() );
    // printf( "Image depth:     %d\n", (int)src.elemSize() / src.channels() );

    // // create a window to display the image
    // cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE );
    // // show the image in the created window
    // cv::imshow("Display window", src);
    // cv::waitKey(0);
    // // wait for a key press indefinitely
