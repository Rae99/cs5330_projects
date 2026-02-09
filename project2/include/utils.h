/*
    Ding, Junrui
    Februray 2026

    CS5330 Project 2 - utils.h

    This header declares small utility helpers shared across tools.
*/

#ifndef UTILS_H
#define UTILS_H

#include <string>

/*
    basename_only

    Return basename (filename only) for a path (e.g. "/a/b.jpg" -> "b.jpg").

    Arguments:
        const std::string &path - input path.

    Returns:
        filename without parent directories.
*/
std::string basename_only(const std::string &path);

#endif // UTILS_H
