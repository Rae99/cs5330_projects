/*
    Ding, Junrui
    CS5330 Project 2 - Task 7: Lawn/Grass Detection

    Uses DNN embedding + green grass features
*/

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "../include/csv_io.h"
#include "../include/features.h"
#include "../include/ranking.h"

static std::string basename_only(const std::string &path) {
    // Find the last occurrence of '/' or '\\' in the path
    const size_t pos = path.find_last_of("/\\");
    // If not found, return the whole path; otherwise, return the substring
    // after the last separator
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

int main(int argc, char **argv) {
    // validate arguments
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <target_image> <image_dir> <emb_csv> <topN>\n";
        return -1;
    }

    // parse arguments
    const std::string target_path = argv[1];
    const std::string image_dir = argv[2];
    const std::string emb_csv = argv[3];
    const int topN =
        std::max(1,
                 std::atoi(argv[4])); // atoi: convert string to int
    const std::string target_name = basename_only(target_path);

    // Read embeddings
    std::ifstream in(emb_csv); // input file stream
    if (!in.is_open()) {
        std::cerr << "Cannot open " << emb_csv << "\n";
        return -1;
    }

    // Parallel arrays: names[i] corresponds to embs[i]
    std::vector<std::string> names;
    std::vector<std::vector<float>> embs;
    std::vector<float> target_emb;
    bool found_target = false;

    std::string line;
    while (std::getline(in, line)) { // read line by line until EOF(end of file)
        if (line.empty())
            continue; // skip empty lines
        std::string fname;
        std::vector<float> feat;
        if (!parse_csv_row(line, fname, feat))
            continue;
        if (feat.size() != 512)
            continue;

        names.push_back(fname);
        embs.push_back(feat);
        if (fname == target_name) {
            target_emb = feat;
            found_target = true;
        }
    }
    in.close();

    if (!found_target) {
        std::cerr << "Target not found in CSV\n";
        return -1;
    }

    // Extract target features
    cv::Mat target_img = cv::imread(target_path);
    if (target_img.empty()) {
        std::cerr << "Cannot read target\n";
        return -1;
    }

    std::vector<float> target_feat;
    if (!extract_grass_features(target_img, target_feat)) {
        std::cerr << "Failed to extract target features\n";
        return -1;
    }

    std::cout << "Target green ratio: " << target_feat[0] << "\n";

    // Compute distances
    std::vector<Match> matches;
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == target_name)
            continue;

        cv::Mat img = cv::imread(image_dir + "/" + names[i]);
        if (img.empty())
            continue;

        std::vector<float> db_feat;
        if (!extract_grass_features(img, db_feat))
            continue;

        // Skip images with very little green
        if (db_feat[0] < 0.05)
            continue;

        float d_emb = cosine_distance(target_emb, embs[i]);
        float d_grass = grass_distance(target_feat, db_feat);

        // Fusion: 40% DNN + 60% grass features
        float d = 0.4f * d_emb + 0.6f * d_grass;

        matches.push_back({names[i], d});
    }

    // Sort and print
    std::sort(matches.begin(), matches.end(),
              [](const Match &a, const Match &b) { return a.dist < b.dist; });

    std::cout << "\nTask 7: Grass/Lawn Detection - Top " << topN
              << " matches\n";
    std::cout << "Target: " << target_path << "\n\n";

    for (int k = 0; k < topN && k < (int)matches.size(); ++k) {
        std::cout << (k + 1) << ". " << matches[k].filename
                  << " (distance: " << matches[k].dist << ")\n";
    }

    for (int k = matches.size() - 5; k < matches.size(); k++) {
        std::cout << "Bottom " << k << ": " << matches[k].filename
                  << " (distance: " << matches[k].dist << ")\n";
    }

    return 0;
}