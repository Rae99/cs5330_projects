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

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../include/csv_io.h"
#include "../include/features.h"
#include "../include/ranking.h"
#include "../include/utils.h"

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "usage: " << argv[0]
                  << " <target_image> <image_dir> <feature_csv> <topN>\n";
        return -1;
    }

    const std::string target_path = argv[1];
    const std::string image_dir = argv[2];
    const std::string csv_path = argv[3];
    const int topN = std::max(1, std::atoi(argv[4]));

    const std::string target_name = basename_only(target_path);

    // compute target feature
    cv::Mat target_img = cv::imread(target_path, cv::IMREAD_UNCHANGED);
    std::vector<float> target_feat;
    if (!compute_task1_feature(target_img, target_feat)) {
        std::cerr << "Failed to compute target feature for: " << target_path
                  << "\n";
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
        if (line.empty())
            continue;

        std::string fname;
        std::vector<float> feat;
        if (!parse_csv_row(line, fname, feat))
            continue;

        // skip the target image itself if it's in the database
        if (fname == target_name)
            continue;

        // sanity: feature dimension should match (147)
        if (feat.size() != target_feat.size())
            continue;

        float d = ssd_distance(target_feat, feat);
        matches.push_back({fname, d});
    }

    in.close();

    // sort ascending by distance (smaller = more similar)
    sort_matches(matches);

    std::cout << "Top " << topN << " matches for target: " << target_path
              << "\n";
    for (int i = 0; i < topN && i < (int)matches.size(); i++) {
        // print filename + distance; you can also print full path if you want
        std::cout << (i + 1) << ") " << matches[i].filename
                  << "  dist=" << matches[i].dist
                  << "  fullpath=" << (image_dir + "/" + matches[i].filename)
                  << "\n";
    }

    return 0;
}