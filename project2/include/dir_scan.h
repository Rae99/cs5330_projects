/*
	Ding, Junrui
	Februray 2026

	CS5330 Project 2 - dir_scan.h

	This header declares directory-scanning helpers for discovering
	image files used by the project’s database build and query tools.
*/

#ifndef DIR_SCAN_H
#define DIR_SCAN_H

#include <string>
#include <vector>

/*
	list_image_files

	List image filenames (not full paths) in `dir` and store them in `files`.

	Arguments:
		const std::string &dir - directory path to scan.
		std::vector<std::string> &files - output list of filenames.

	Returns:
		true on success, false on failure.
*/
bool list_image_files(const std::string &dir, std::vector<std::string> &files);

/*
	is_image_filename

	Check by filename suffix whether the name appears to be an image file.

	Arguments:
		const std::string &name - filename to test.

	Returns:
		true if the filename has a supported image extension.
*/
bool is_image_filename(const std::string &name);

#endif // DIR_SCAN_H
