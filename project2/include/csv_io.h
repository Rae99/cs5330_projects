/*
    Ding, Junrui
    Februray 2026

    CS5330 Project 2 - csv_io.h

    This header declares CSV serialization helpers for feature vectors
    used by the build and query tools.
*/

#ifndef CSV_IO_H
#define CSV_IO_H

#include <fstream>
#include <string>
#include <vector>

/*
    write_csv_row

    Write a CSV row in the format: filename,f1,f2,... to the output stream.

    Arguments:
        std::ofstream &out - output file stream.
        const std::string &filename - image filename.
        const std::vector<float> &feat - feature vector values.

    Returns:
        void.
*/
void write_csv_row(std::ofstream &out, const std::string &filename,
                   const std::vector<float> &feat);

/*
    parse_csv_row

    Parse a CSV row produced by write_csv_row into filename and feature list.

    Arguments:
        const std::string &line - CSV row string.
        std::string &filename - output filename field.
        std::vector<float> &feat - output feature vector.

    Returns:
        true on success, false on failure.
*/
bool parse_csv_row(const std::string &line, std::string &filename,
                   std::vector<float> &feat);

#endif // CSV_IO_H
