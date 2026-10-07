#pragma once

#include <string>

namespace irodori {

/**
 * The official runtime's normalize_text() followed by its strip(): symbol replacements, removal of
 * brackets that enclose the whole text, NFKC, and "..." written as "…".
 */
std::string normalize_text(const std::string & text);

/**
 * A caption as the official runtime takes it: Python's str.strip() alone, which removes the characters Python counts
 * as whitespace (U+3000 among them) from both ends. A caption that strips to nothing is no caption.
 */
std::string strip_caption(const std::string & caption);

}  // namespace irodori
