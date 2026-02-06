/*
  Task 5 Query Program (deep embeddings)
  - Reads embedding CSV (filename + 512 floats)
  - Finds target embedding by target filename
  - Computes cosine distance to all embeddings
  - Sorts and prints top N (excluding itself)

  Usage:
    ./query_task5 <target_filename> <embedding_csv> <topN>
*/

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../include/csv_io.h"
#include "../include/ranking.h"


int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0]
                  << " <target_filename> <embedding_csv> <topN>\n";
        return -1;
    }

    const std::string target_name = argv[1];   // e.g., pic.0535.jpg
    const std::string csv_path = argv[2];
    const int topN = std::max(1, std::atoi(argv[3]));

    std::ifstream in(csv_path);
    if (!in.is_open()) {
        std::cerr << "Cannot open csv: " << csv_path << "\n";
        return -1;
    }

    // 1) read all embeddings
    std::vector<std::pair<std::string, std::vector<float>>> rows;
    rows.reserve(2000);

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::string fname;
        std::vector<float> feat;
        if (!parse_csv_row(line, fname, feat))
            continue;
        rows.push_back({fname, feat});
    }
    in.close();

    // 2) find target embedding
    std::vector<float> target_feat;
    bool found = false;
    for (const auto &r : rows) {
        if (r.first == target_name) {
            target_feat = r.second;
            found = true;
            break;
        }
    }
    if (!found) {
        std::cerr << "Target filename not found in embedding csv: " << target_name << "\n";
        return -1;
    }

    // 3) compute distances (exclude itself)
    std::vector<Match> matches;
    matches.reserve(rows.size());

    for (const auto &r : rows) {
        if (r.first == target_name)
            continue;

        // sanity: should be 512 dims
        if (r.second.size() != target_feat.size())
            continue;

        float d = cosine_distance(target_feat, r.second);
        matches.push_back({r.first, d});
    }

    // 4) sort ascending
    std::sort(matches.begin(), matches.end(),
              [](const Match &m1, const Match &m2) { return m1.dist < m2.dist; });

    std::cout << "Top " << topN << " matches (Task5 cosine) for target: "
              << target_name << "\n";
    for (int i = 0; i < topN && i < (int)matches.size(); i++) {
        std::cout << (i + 1) << ") " << matches[i].filename
                  << "  dist=" << matches[i].dist << "\n";
    }

    return 0;
}