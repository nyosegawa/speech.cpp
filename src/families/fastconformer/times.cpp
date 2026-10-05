#include "times.h"

#include <algorithm>
#include <stdexcept>

namespace fastconformer {

namespace {

/** Whether `s` is one UTF-8 character. */
bool one_character(const std::string & s) {
    if (s.empty()) return false;
    const unsigned char lead = (unsigned char) s[0];
    const size_t n = lead < 0x80 ? 1 : lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : 2;
    return s.size() == n;
}

}  // namespace

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
    if (spans.size() != texts.size() || word_starts.size() != texts.size()) {
        throw std::runtime_error("segments need a span and a word start for each token's text");
    }
    for (const std::vector<std::string> * marks : {&separators, &breaks}) {
        for (const std::string & m : *marks) {
            if (m.empty()) throw std::runtime_error("a mark that ends a segment is empty");
        }
    }
    auto ends_with = [](const std::string & text, const std::string & end) {
        return text.size() >= end.size() && text.compare(text.size() - end.size(), end.size(), end) == 0;
    };
    // word[-1] in segment_delimiter_tokens or word in segment_delimiter_tokens, on NeMo's word, which has no leading
    // space.
    auto ends_sentence = [&](const std::string & word) {
        const std::string trimmed = word.substr(std::min(word.find_first_not_of(' '), word.size()));
        for (const std::string & s : separators) {
            if (trimmed == s || (one_character(s) && ends_with(trimmed, s))) return true;
        }
        return false;
    };
    auto ends_break = [&](const std::string & text) {
        for (const std::string & b : breaks) {
            if (ends_with(text, b)) return true;
        }
        return false;
    };
    std::vector<Segment> out;
    size_t first = 0, word = 0;
    for (size_t i = 0; i < texts.size(); i++) {
        const bool last = i + 1 == texts.size();
        bool cut = last || ends_break(texts[i]);
        if (last || word_starts[i + 1]) {
            std::string text;
            for (size_t k = word; k <= i; k++) text += texts[k];
            cut = cut || ends_sentence(text);
            word = i + 1;
        }
        if (!cut) continue;
        Segment s{first, i + 1, {spans[first].start, spans[i].end}, ""};
        for (size_t k = first; k <= i; k++) s.text += texts[k];
        out.push_back(std::move(s));
        first = i + 1;
    }
    return out;
}

}  // namespace fastconformer
