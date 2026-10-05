#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "decoder.h"
#include "detokenizer.h"

namespace fastconformer {

/** A start and an end in encoder frames. */
struct Span {
    int64_t start, end;
};

/**
 * The span of each token of `decoding` as compute_rnnt_timestamps() (nemo/collections/asr/parts/submodules/
 * rnnt_decoding.py) gives it: from the frame the token was emitted on to the frame its predicted duration reaches, or
 * to the next frame for a decoding that predicts no durations. With durations, a token that is a punctuation mark
 * spans nothing where the token before it ends (_refine_timestamps_tdt()), since TDT can emit a mark long after the
 * word it ends when silence follows. NeMo's beam search gives the step of its search where this gives the frame
 * (AlsdDecoder).
 */
std::vector<Span> token_spans(const Decoding & decoding, const Detokenizer & detokenizer);

/** A run of tokens [first, end), its span and its text. */
struct Segment {
    size_t first, end;
    Span span;
    std::string text;
};

/**
 * The segments of tokens with the texts `texts`, as they stand in the text, the word starts `word_starts`
 * (Detokenizer::word_starts()) and the spans `spans`. A segment ends with a word whose text ends in one of
 * `separators`, as get_segment_offsets() ends it: the word's last character is a separator of one character, or the
 * word is a separator; so "1.000", "z.B." and "?!" are not cut within. It also ends with any token whose text ends in
 * one of `breaks`, wherever it stands, for a text written without spaces, in which NeMo's words run to the end. The
 * last token ends the last segment. A segment spans from the start of its first token to the end of its last, and its
 * text is theirs, so that the texts of the segments joined are the text.
 */
std::vector<Segment> segments(const std::vector<std::string> & texts, const std::vector<bool> & word_starts,
                              const std::vector<Span> & spans, const std::vector<std::string> & separators,
                              const std::vector<std::string> & breaks);

}  // namespace fastconformer
