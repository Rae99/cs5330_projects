#include "../include/dir_scan.h"

#include <cstring>
#include <dirent.h>

static bool has_suffix(const std::string &s, const char *suf) {
    if (s.size() < strlen(suf))
        return false;
    return s.compare(s.size() - strlen(suf), strlen(suf), suf) == 0;
}

bool is_image_filename(const std::string &name) {
    // check common extensions (lowercase)
    std::string n = name;
    for (char &c : n)
        c = static_cast<char>(std::tolower(c));
    return has_suffix(n, ".jpg") || has_suffix(n, ".png") ||
           has_suffix(n, ".ppm") || has_suffix(n, ".tif") ||
           has_suffix(n, ".jpeg");
}

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
