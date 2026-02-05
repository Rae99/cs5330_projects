#include "../include/utils.h"

#include <cstring>

std::string basename_only(const std::string &path) {
    const char *s = path.c_str();
    const char *p = std::strrchr(s, '/');
    return p ? std::string(p + 1) : path;
}
