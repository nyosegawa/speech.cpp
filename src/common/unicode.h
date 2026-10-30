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

/**
 * `text` as Python 3.10's str.lower() writes it: each code point as its full lowercase mapping, which is two code points
 * for U+0130, and a capital sigma as the final sigma where CPython takes it to end a word, after a cased character and
 * before none, with the case-ignorable characters between them looked past.
 */
std::vector<uint32_t> python_lower(const std::vector<uint32_t> & text);

/** Whether Python 3.10's str.isdigit() takes `cp` for a digit, a character whose numeric type is Digit or Decimal. */
bool python_digit(uint32_t cp);

/**
 * The general categories of Unicode 15.0, as Python 3.12's unicodedata.category() gives them and the tokenizers library's
 * regular expressions match them: letters (L), numbers (N), marks (M), the whitespace of \s, uppercase and titlecase
 * letters (Lu, Lt), lowercase letters (Ll), and modifier and other letters (Lm, Lo).
 */
bool unicode_letter(uint32_t cp);
bool unicode_number(uint32_t cp);
bool unicode_mark(uint32_t cp);
bool unicode_space(uint32_t cp);
bool unicode_upper(uint32_t cp);
bool unicode_lower(uint32_t cp);
bool unicode_other_letter(uint32_t cp);
