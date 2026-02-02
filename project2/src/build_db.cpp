/*
  Build DB for Task 1 (baseline):
  - Feature: 7x7 square in the middle of the image (RGB flattened)
  - Output CSV: filename, f1, f2, ... f147

  Usage:
    ./build_db <image_dir> <output_csv>

  Example:
    ./build_db ./olympus ./features_task1.csv
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

// ---------- helpers ----------

// check if the file is an image (simple substring check, same idea as instructor code)
static bool is_image_file(const char* name) {
  return (strstr(name, ".jpg") || strstr(name, ".png") || strstr(name, ".ppm") || strstr(name, ".tif"));
}

// Task 1 feature: 7x7 center patch, RGB flattened to length 147
static bool compute_task1_feature(const cv::Mat& img, std::vector<float>& feat) {
  if (img.empty()) return false;

  // Ensure 3-channel BGR input (OpenCV default). If grayscale, convert to BGR.
  cv::Mat bgr;
  if (img.channels() == 3) {
    bgr = img;
  } else if (img.channels() == 1) {
    cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
  } else {
    return false;
  }

  if (bgr.rows < 7 || bgr.cols < 7) return false;

  const int cy = bgr.rows / 2;
  const int cx = bgr.cols / 2;

  // Top-left corner of 7x7 patch
  const int y0 = cy - 3;
  const int x0 = cx - 3;

  // Guard for very small images where center-3 might be negative (just in case)
  if (y0 < 0 || x0 < 0 || y0 + 6 >= bgr.rows || x0 + 6 >= bgr.cols) return false;

  feat.clear();
  feat.reserve(7 * 7 * 3);

  // Flatten in a fixed order: row-major, then channels (B,G,R) as stored in OpenCV
  for (int y = y0; y < y0 + 7; y++) {
    const cv::Vec3b* row = bgr.ptr<cv::Vec3b>(y);
    for (int x = x0; x < x0 + 7; x++) {
      const cv::Vec3b& p = row[x];
      feat.push_back(static_cast<float>(p[0])); // B
      feat.push_back(static_cast<float>(p[1])); // G
      feat.push_back(static_cast<float>(p[2])); // R
    }
  }

  // Should be 147 dims for color
  return (feat.size() == 147);
}

static void write_csv_row(std::ofstream& out, const std::string& filename, const std::vector<float>& feat) {
  out << filename;
  for (float v : feat) out << "," << v;
  out << "\n";
}

/*
  Given a directory on the command line, scans through the directory for image files.

  Prints out the full path name for each file.  This can be used as an argument to fopen or to cv::imread.
 */
int main(int argc, char *argv[]) {
  char dirname[256];
  char buffer[256];
  DIR *dirp;
  struct dirent *dp;

  // check for sufficient arguments
  if( argc < 3) {
    printf("usage: %s <directory path> <output csv>\n", argv[0]);
    exit(-1);
  }

  // get the directory path
  strcpy(dirname, argv[1]);
  printf("Processing directory %s\n", dirname );

  const std::string out_csv = argv[2];
  std::ofstream out(out_csv);
  if (!out.is_open()) {
    std::cerr << "Cannot open output csv: " << out_csv << "\n";
    return -1;
  }

  // open the directory
  dirp = opendir( dirname );
  if( dirp == NULL) {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  int written = 0;
  int skipped = 0;

  // loop over all the files in the image file listing
  while( (dp = readdir(dirp)) != NULL ) {

    // check if the file is an image
    if( is_image_file(dp->d_name) ) {

      printf("processing image file: %s\n", dp->d_name);

      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "/");
      strcat(buffer, dp->d_name);

      printf("full path name: %s\n", buffer);

      cv::Mat img = cv::imread(buffer, cv::IMREAD_UNCHANGED);
      std::vector<float> feat;
      if (!compute_task1_feature(img, feat)) {
        std::cerr << "  [skip] failed to compute feature for " << dp->d_name << "\n";
        skipped++;
        continue;
      }

      // store just the filename (not full path) to match typical assignment outputs
      write_csv_row(out, dp->d_name, feat);
      written++;
    }
  }

  closedir(dirp);
  out.close();

  printf("Wrote %d feature rows to %s (skipped %d)\n", written, out_csv.c_str(), skipped);
  printf("Terminating\n");

  return(0);
}