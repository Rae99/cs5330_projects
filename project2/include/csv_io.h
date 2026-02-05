#ifndef CSV_IO_H
#define CSV_IO_H

#include <fstream>
#include <string>
#include <vector>

// write a CSV row: filename,f1,f2,...
void write_csv_row(std::ofstream &out, const std::string &filename,
                   const std::vector<float> &feat);

// parse a CSV row produced by write_csv_row
bool parse_csv_row(const std::string &line, std::string &filename,
                   std::vector<float> &feat);

#endif // CSV_IO_H
