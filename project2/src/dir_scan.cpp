/*
    Ding, Junrui
    Februray 2026

    CS5330 Project 2 - dir_scan.cpp

    This file implements directory scanning and filename filtering helpers
    for image datasets.
*/

#include "../include/dir_scan.h"

#include <cstring>
#include <dirent.h>

/*
    has_suffix

    Return true if string s ends with suffix suf.

    Arguments:
        const std::string &s - input string.
        const char *suf - suffix to check.

    Returns:
        true if s ends with suf, false otherwise.
*/
static bool has_suffix(const std::string &s, const char *suf) {
    if (s.size() < strlen(suf))
        return false;
    return s.compare(s.size() - strlen(suf), strlen(suf), suf) == 0;
}

/*
    is_image_filename

    Check common file extensions to determine if a filename is an image.

    Arguments:
        const std::string &name - filename to check.

    Returns:
        true if the filename has a supported image extension.
*/
bool is_image_filename(const std::string &name) {
    // check common extensions (lowercase)
    std::string n = name;
    for (char &c : n)
        c = static_cast<char>(std::tolower(c));
    return has_suffix(n, ".jpg") || has_suffix(n, ".png") ||
           has_suffix(n, ".ppm") || has_suffix(n, ".tif") ||
           has_suffix(n, ".jpeg");
}

/*
    list_image_files

    Scan a directory and collect image filenames into `files`.

    Arguments:
        const std::string &dir - directory path to scan.
        std::vector<std::string> &files - output list of filenames.

    Returns:
        true on success, false on failure.
*/
bool list_image_files(const std::string &dir, std::vector<std::string> &files) {
    DIR *dp = opendir(dir.c_str());
    if (!dp)
        return false;
    files.clear();
    struct dirent *ent;
    while ((ent = readdir(dp)) != NULL) {
        std::string name(ent->d_name);
        if (is_image_filename(name))
            files.push_back(name);
    }
    closedir(dp);
    return true;
}
