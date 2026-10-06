#pragma once

#include <cstdint>
#include <vector>

/**
 * The version of Unicode whose tables a normalization follows, each that of a reference. One set of tables holds both,
 * since a character's decomposition, combining class and composition never change once it is assigned: those of 9.0
 * are those of 13.0 without the characters assigned since.
 */
enum class UnicodeVersion : uint8_t {
    /** The tables of the unicode-normalization-alignments crate 0.1.12, which the tokenizers library normalizes with. */
    V9 = 9,
    /** The tables of Python 3.10's unicodedata. */
    V13 = 13,
};

/** The NFC of `text` by the tables of `version`. */
std::vector<uint32_t> nfc(const std::vector<uint32_t> & text, UnicodeVersion version);

/** The NFKC of `text` by the tables of `version`. */
std::vector<uint32_t> nfkc(const std::vector<uint32_t> & text, UnicodeVersion version);

/** Whether Python's str.isspace() takes `cp` for whitespace, which str.strip() removes; Python 3.10 and 3.12 agree. */
bool python_space(uint32_t cp);
