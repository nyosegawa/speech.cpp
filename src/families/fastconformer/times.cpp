#include "times.h"

#include <stdexcept>

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

std::vector<Segment> segments(const std::vector<std::string> & texts, const std::vector<Span> & spans,
                              const std::vector<std::string> & separators) {
    if (spans.size() != texts.size()) throw std::runtime_error("segments need a span for each token's text");
    for (const std::string & s : separators) {
        if (s.empty()) throw std::runtime_error("a segment separator is empty");
    }
    auto ends_sentence = [&](const std::string & text) {
        for (const std::string & s : separators) {
            if (text.size() >= s.size() && text.compare(text.size() - s.size(), s.size(), s) == 0) return true;
        }
        return false;
    };
    std::vector<Segment> out;
    size_t first = 0;
    for (size_t i = 0; i < texts.size(); i++) {
        if (i + 1 < texts.size() && !ends_sentence(texts[i])) continue;
        Segment s{first, i + 1, {spans[first].start, spans[i].end}, ""};
        for (size_t k = first; k <= i; k++) s.text += texts[k];
        out.push_back(std::move(s));
        first = i + 1;
    }
    return out;
}

}  // namespace fastconformer
