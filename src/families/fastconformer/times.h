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
 * The segments of tokens with the texts `texts`, as they stand in the text, and the spans `spans`: runs that end with
 * a token whose text ends in one of `separators`, or with the last token. A segment spans from the start of its first
 * token to the end of its last, and its text is theirs, so that the texts of the segments joined are the text.
 *
 * NeMo's get_segment_offsets() ends a segment at a word, a run of tokens up to a space, that ends in a separator. The
 * two agree where a separator ends a word; a separator within a word, as in "1.000", "z.B." or "?!", ends a segment
 * here and not in NeMo, and in Japanese, which has no spaces, NeMo ends none before the last token.
 */
std::vector<Segment> segments(const std::vector<std::string> & texts, const std::vector<Span> & spans,
                              const std::vector<std::string> & separators);

}  // namespace fastconformer
