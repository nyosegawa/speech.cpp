#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

/** The folders of reference/fastconformer/dump.py under `root`: those that hold features.npy, sorted. */
inline std::vector<std::filesystem::path> fastconformer_dumps(const std::string & root) {
    std::vector<std::filesystem::path> dumps;
    for (const auto & e : std::filesystem::recursive_directory_iterator(std::filesystem::u8path(root))) {
        if (e.is_regular_file() && e.path().filename() == "features.npy") dumps.push_back(e.path().parent_path());
    }
    std::sort(dumps.begin(), dumps.end());
    if (dumps.empty()) throw std::runtime_error("no dump of reference/fastconformer/dump.py is under " + root);
    return dumps;
}

/** A dump's ctc_text.txt: the official CTC text in UTF-8. */
inline std::string fastconformer_text(const std::filesystem::path & dump) {
    std::ifstream f(dump / "ctc_text.txt", std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + (dump / "ctc_text.txt").u8string());
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
