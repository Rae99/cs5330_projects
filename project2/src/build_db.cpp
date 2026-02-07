/*
  Build DB for Task 1 (baseline):
  - Feature: 7x7 square in the middle of the image (RGB flattened)
  - Output CSV: filename, f1, f2, ... f147

  Usage:
  cd to build/

  ./build_db <image_dir> <output_csv>

  Example:
    ./build_db ../data/olympus ../output/features_task1.csv
*/


#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../include/csv_io.h"
#include "../include/dir_scan.h"
#include "../include/features.h"
#include "../include/task_registry.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <directory path> <output csv>\n",
                     argv[0]);
        return -1;
    }

    const std::string dirname = argv[1];
    const std::string out_csv = argv[2];

    std::ofstream out(out_csv);
    if (!out.is_open()) {
        std::cerr << "Cannot open output csv: " << out_csv << "\n";
        return -1;
    }

    // task id optional (default = 1)
    const int task_id = (argc > 3) ? std::atoi(argv[3]) : 1;
    TaskSpec spec;
    try {
        spec = get_task(task_id);
    } catch (const std::exception &e) {
        std::cerr << "Invalid task id: " << task_id << " (" << e.what()
                  << ")\n";
        return -1;
    }

    std::vector<std::string> files;
    if (!list_image_files(dirname, files)) {
        std::cerr << "Cannot open directory " << dirname << "\n";
        return -1;
    }

    int written = 0;
    int skipped = 0;

    for (const std::string &name : files) {
        std::printf("processing image file: %s\n", name.c_str());

        const std::string full = dirname + "/" + name;
        std::printf("full path name: %s\n", full.c_str());

        cv::Mat img = cv::imread(full, cv::IMREAD_UNCHANGED);
        std::vector<float> feat;
        if (!spec.feature(img, feat)) {
            std::cerr << "  [skip] failed to compute feature for " << name
                      << "\n";
            skipped++;
            continue;
        }

        write_csv_row(out, name, feat);
        written++;
    }

    out.close();
    std::printf("Wrote %d feature rows to %s (skipped %d)\n", written,
                out_csv.c_str(), skipped);
    std::printf("Terminating\n");
    return 0;
}