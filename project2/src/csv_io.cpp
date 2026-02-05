#include "../include/csv_io.h"

#include <iostream>
#include <sstream>

void write_csv_row(std::ofstream &out, const std::string &filename,
                   const std::vector<float> &feat) {
    out << filename;
    for (float v : feat)
        out << "," << v;
    out << "\n";
}

bool parse_csv_row(const std::string &line, std::string &filename,
                   std::vector<float> &feat) {
    std::stringstream ss(line);
    std::string token;

    if (!std::getline(ss, token, ','))
        return false;
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
