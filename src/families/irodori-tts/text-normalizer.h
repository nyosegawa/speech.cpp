#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace irodori {

/** NFKC of `text`, with the tables of Unicode 13.0, the version of the Python the official runtime runs on. */
std::vector<uint32_t> nfkc(const std::vector<uint32_t> & text);

/**
 * The official runtime's normalize_text() followed by its strip(): symbol replacements, removal of
 * brackets that enclose the whole text, NFKC, and "..." written as "…".
 */
std::string normalize_text(const std::string & text);

}  // namespace irodori
