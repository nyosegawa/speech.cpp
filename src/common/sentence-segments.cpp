#include "sentence-segments.h"

#include <algorithm>
#include <stdexcept>

namespace {

/** Whether `s` is one UTF-8 character. */
bool one_character(const std::string & s) {
    if (s.empty()) return false;
    const unsigned char lead = (unsigned char) s[0];
    const size_t n = lead < 0x80 ? 1 : lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : 2;
    return s.size() == n;
}

bool ends_with(const std::string & text, const std::string & end) {
    return text.size() >= end.size() && text.compare(text.size() - end.size(), end.size(), end) == 0;
}

}  // namespace

std::vector<size_t> segment_ends(const std::vector<std::string> & texts, const std::vector<bool> & word_starts,
                                 const std::vector<std::string> & separators, const std::vector<std::string> & breaks,
                                 const std::vector<bool> & turns) {
    if (word_starts.size() != texts.size() || (!turns.empty() && turns.size() != texts.size())) {
        throw std::runtime_error("segments need a word start and a turn for each token's text");
    }
    for (const std::vector<std::string> * marks : {&separators, &breaks}) {
        for (const std::string & m : *marks) {
            if (m.empty()) throw std::runtime_error("a mark that ends a segment is empty");
        }
    }
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
    std::vector<size_t> out;
    size_t word = 0;
    for (size_t i = 0; i < texts.size(); i++) {
        const bool last = i + 1 == texts.size();
        bool cut = last || ends_break(texts[i]) || (!turns.empty() && turns[i + 1]);
        if (last || word_starts[i + 1]) {
            std::string text;
            for (size_t k = word; k <= i; k++) text += texts[k];
            cut = cut || ends_sentence(text);
            word = i + 1;
        }
        if (cut) out.push_back(i + 1);
    }
    return out;
}
