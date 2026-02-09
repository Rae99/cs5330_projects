/*
    Ding, Junrui
    Februray 2026

    CS5330 Project 2 - csv_io.cpp

    This file implements CSV serialization helpers for feature vectors.
*/

#include "../include/csv_io.h"

#include <iostream>
#include <sstream>

/*
    write_csv_row

    Write a single CSV row in the format: filename,f1,f2,...

    Arguments:
        std::ofstream &out - output file stream.
        const std::string &filename - image filename.
        const std::vector<float> &feat - feature vector values.

    Returns:
        void.
*/
void write_csv_row(std::ofstream &out, const std::string &filename,
                   const std::vector<float> &feat) {
    out << filename;
    for (float v : feat)
        out << "," << v;
    out << "\n";
}

/*
    parse_csv_row

    Parse a CSV row produced by write_csv_row into filename and feature vector.

    Arguments:
        const std::string &line - CSV row string.
        std::string &filename - output filename field.
        std::vector<float> &feat - output feature vector.

    Returns:
        true on success, false on failure.
*/
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
