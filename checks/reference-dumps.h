#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "model-file.h"

/**
 * The folders the reference/<architecture>/dump.py of a recognition model wrote under `root` for the model of `model`,
 * whose general.name is the name dump.py was given: those under <root>/<name>/ that hold features.npy, sorted.
 */
inline std::vector<std::filesystem::path> reference_dumps(const std::string & root, const ModelFile & model) {
    const std::filesystem::path dir = std::filesystem::u8path(root) / std::filesystem::u8path(model.str("general.name"));
    std::vector<std::filesystem::path> dumps;
    if (std::filesystem::is_directory(dir)) {
        for (const auto & e : std::filesystem::recursive_directory_iterator(dir)) {
            if (e.is_regular_file() && e.path().filename() == "features.npy") dumps.push_back(e.path().parent_path());
        }
    }
    std::sort(dumps.begin(), dumps.end());
    if (dumps.empty()) {
        throw std::runtime_error("no dump of reference/" + model.str("general.architecture") + "/dump.py is under " + dir.u8string());
    }
    return dumps;
}

/** A text file of a dump, such as text.txt with the official text, in UTF-8. */
inline std::string dump_text(const std::filesystem::path & file) {
    std::ifstream f(file, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + file.u8string());
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
