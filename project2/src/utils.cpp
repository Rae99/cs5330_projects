/*
    Ding, Junrui
    Februray 2026

    CS5330 Project 2 - utils.cpp

    This file implements small utility helpers used by multiple tools.
*/

#include "../include/utils.h"

#include <cstring>

/*
    basename_only

    Return the filename portion of a path (strip parent directories).

    Arguments:
        const std::string &path - input path.

    Returns:
        filename without parent directories.
*/
std::string basename_only(const std::string &path) {
    const char *s = path.c_str();
    const char *p = std::strrchr(s, '/');
    return p ? std::string(p + 1) : path;
}
