#pragma once

#include <cstddef>
#include <string>
#include <vector>

/**
 * Where the segments of a recognition's tokens end, the tokens' texts `texts` as they stand in the text: after a token
 * whose text ends in one of `breaks`, wherever it stands, for a text written without spaces, in which NeMo's words run to
 * the end; after the last token of a word (`word_starts` marks the first token of each) whose text ends in one of
 * `separators`, as NeMo's get_segment_offsets() ends it, its leading spaces left out: the word's last character is a
 * separator of one character, or the word is a separator, so that "1.000", "z.B." and "?!" are not cut within; before
 * each token that `turns`, where given, marks as the start of a turn; and after the last token. Returns the end, past
 * the last token, of each segment in order.
 */
std::vector<size_t> segment_ends(const std::vector<std::string> & texts, const std::vector<bool> & word_starts,
                                 const std::vector<std::string> & separators, const std::vector<std::string> & breaks,
                                 const std::vector<bool> & turns = {});
