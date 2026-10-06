#pragma once

#include <string>

namespace irodori {

/**
 * The official runtime's normalize_text() followed by its strip(): symbol replacements, removal of
 * brackets that enclose the whole text, NFKC, and "..." written as "…".
 */
std::string normalize_text(const std::string & text);

}  // namespace irodori
