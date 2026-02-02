/*
  Task 1 Query Program (baseline)
  - Reads feature CSV produced by build_db
  - Computes Task1 feature for target image
  - Uses SSD distance (write it yourself, no OpenCV distance function)
  - Sort and print top N matches

  Usage:
    ./query_db <target_image> <image_dir> <feature_csv> <topN>

  Notes:
  - image_dir is used only to print full path if you want; CSV stores filenames.
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>

// Task 1 feature: 7x7 center patch, RGB flattened to length 147
static bool compute_task1_feature(const cv::Mat& img, std::vector<float>& feat) {
  if (img.empty()) return false;

  cv::Mat bgr;
  if (img.channels() == 3) bgr = img;
  else if (img.channels() == 1) cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
  else return false;

  if (bgr.rows < 7 || bgr.cols < 7) return false;

  const int cy = bgr.rows / 2;
  const int cx = bgr.cols / 2;
  const int y0 = cy - 3;
  const int x0 = cx - 3;

  if (y0 < 0 || x0 < 0 || y0 + 6 >= bgr.rows || x0 + 6 >= bgr.cols) return false;

  feat.clear();
  feat.reserve(147);

  for (int y = y0; y < y0 + 7; y++) {
    const cv::Vec3b* row = bgr.ptr<cv::Vec3b>(y);
    for (int x = x0; x < x0 + 7; x++) {
      const cv::Vec3b& p = row[x];
      feat.push_back(static_cast<float>(p[0])); // B
      feat.push_back(static_cast<float>(p[1])); // G
      feat.push_back(static_cast<float>(p[2])); // R
    }
  }
  return (feat.size() == 147);
}

// SSD distance
static float ssd_distance(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return 1e30f;
  double s = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    s += d * d;
  }
  return static_cast<float>(s);
}

struct Match {
  std::string filename;
  float dist;
};

static bool parse_csv_row(const std::string& line, std::string& filename, std::vector<float>& feat) {
  std::stringstream ss(line);
  std::string token;

  // first token: filename
  if (!std::getline(ss, token, ',')) return false;
  filename = token;

  feat.clear();
  while (std::getline(ss, token, ',')) {
    try {
      feat.push_back(std::stof(token));
    } catch (...) {
      return false;
    }
  }
  return !filename.empty() && !feat.empty();
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: " << argv[0] << " <target_image> <image_dir> <feature_csv> <topN>\n";
    return -1;
  }

  const std::string target_path = argv[1];
  const std::string image_dir   = argv[2];
  const std::string csv_path    = argv[3];
  const int topN = std::max(1, std::atoi(argv[4]));

  // compute target feature
  cv::Mat target_img = cv::imread(target_path, cv::IMREAD_UNCHANGED);
  std::vector<float> target_feat;
  if (!compute_task1_feature(target_img, target_feat)) {
    std::cerr << "Failed to compute target feature for: " << target_path << "\n";
    return -1;
  }

  // read db csv and compute distances
  std::ifstream in(csv_path);
  if (!in.is_open()) {
    std::cerr << "Cannot open csv: " << csv_path << "\n";
    return -1;
  }

  std::vector<Match> matches;
  std::string line;

  while (std::getline(in, line)) {
    if (line.empty()) continue;

    std::string fname;
    std::vector<float> feat;
    if (!parse_csv_row(line, fname, feat)) continue;

    // sanity: feature dimension should match (147)
    if (feat.size() != target_feat.size()) continue;

    float d = ssd_distance(target_feat, feat);
    matches.push_back({fname, d});
  }

  in.close();

  // sort ascending by distance (smaller = more similar)
  std::sort(matches.begin(), matches.end(),
            [](const Match& m1, const Match& m2){ return m1.dist < m2.dist; });

  std::cout << "Top " << topN << " matches for target: " << target_path << "\n";
  for (int i = 0; i < topN && i < (int)matches.size(); i++) {
    // print filename + distance; you can also print full path if you want
    std::cout << (i+1) << ") " << matches[i].filename
              << "  dist=" << matches[i].dist
              << "  fullpath=" << (image_dir + "/" + matches[i].filename)
              << "\n";
  }

  return 0;
}