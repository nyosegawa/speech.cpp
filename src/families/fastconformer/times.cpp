#include "times.h"

#include <algorithm>
#include <stdexcept>

#include "sentence-segments.h"

namespace fastconformer {

std::vector<Span> token_spans(const Decoding & decoding, const Detokenizer & detokenizer) {
    const size_t n = decoding.ids.size();
    const bool durations = !decoding.durations.empty();
    if (decoding.frames.size() != n || (durations && decoding.durations.size() != n)) {
        throw std::runtime_error("a decoding does not have a frame and a duration for each token");
    }
    std::vector<Span> spans(n);
    for (size_t i = 0; i < n; i++) {
        const int64_t start = decoding.frames[i];
        spans[i] = {start, start + (durations ? decoding.durations[i] : 1)};
        if (durations && i > 0 && detokenizer.punctuation(decoding.ids[i])) spans[i] = {spans[i - 1].end, spans[i - 1].end};
    }
    return spans;
}

std::vector<Segment> segments(const std::vector<std::string> & texts, const std::vector<bool> & word_starts,
                              const std::vector<Span> & spans, const std::vector<std::string> & separators,
                              const std::vector<std::string> & breaks) {
    if (spans.size() != texts.size()) throw std::runtime_error("segments need a span for each token's text");
    std::vector<Segment> out;
    size_t first = 0;
    for (size_t end : segment_ends(texts, word_starts, separators, breaks)) {
        Segment s{first, end, {spans[first].start, spans[end - 1].end}, ""};
        for (size_t k = first; k < end; k++) s.text += texts[k];
        out.push_back(std::move(s));
        first = end;
    }
    return out;
}

}  // namespace fastconformer
