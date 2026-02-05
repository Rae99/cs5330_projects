#ifndef DIR_SCAN_H
#define DIR_SCAN_H

#include <string>
#include <vector>

// List image filenames (not full paths) in `dir`. Returns true on success.
bool list_image_files(const std::string &dir, std::vector<std::string> &files);

// Check by filename suffix whether this looks like an image file.
bool is_image_filename(const std::string &name);

#endif // DIR_SCAN_H
