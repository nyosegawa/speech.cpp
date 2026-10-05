#include "ctc.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastconformer {

namespace {

const std::string kSpaceSymbol = "\xe2\x96\x81";

std::string replace_all(std::string s, const std::string & from, const std::string & to) {
    for (size_t at = s.find(from); at != std::string::npos; at = s.find(from, at + to.size())) s.replace(at, from.size(), to);
    return s;
}

}  // namespace

CtcHead::CtcHead(const ModelFile & m) : m_(m), blank_((int) m.u32("fastconformer.ctc.blank_id")) {
    if (m.tensor("ctc.weight")->ne[1] != blank_ + 1) throw std::runtime_error("ctc.weight does not have a class for each token and the blank");
}

ggml_tensor * CtcHead::build(Graph & g, ggml_tensor * encoded) const {
    return ggml_add(g.ctx(), mul_mat(g.ctx(), m_.tensor("ctc.weight"), encoded), m_.tensor("ctc.bias"));
}

std::vector<int32_t> CtcHead::greedy(const std::vector<float> & logits) const {
    const size_t classes = (size_t) blank_ + 1, frames = logits.size() / classes;
    std::vector<int32_t> ids;
    int32_t previous = blank_;
    for (size_t t = 0; t < frames; t++) {
        const float * row = &logits[t * classes];
        int32_t best = 0;
        // torch.max() keeps the first of equal values.
        for (size_t c = 1; c < classes; c++) {
            if (row[c] > row[best]) best = (int32_t) c;
        }
        if (best != blank_ && best != previous) ids.push_back(best);
        previous = best;
    }
    return ids;
}

std::vector<float> log_softmax(const std::vector<float> & logits, int classes) {
    std::vector<float> out(logits.size());
    for (size_t r = 0; r < logits.size() / (size_t) classes; r++) {
        const float * row = &logits[r * classes];
        double top = row[0];
        for (int c = 1; c < classes; c++) top = std::max(top, (double) row[c]);
        double sum = 0;
        for (int c = 0; c < classes; c++) sum += std::exp((double) row[c] - top);
        const double log_sum = top + std::log(sum);
        for (int c = 0; c < classes; c++) out[r * classes + c] = (float) ((double) row[c] - log_sum);
    }
    return out;
}

Detokenizer::Detokenizer(const ModelFile & m)
    : pieces_(m.str_array("tokenizer.tokens")),
      punctuation_(m.str_array("tokenizer.punctuation")),
      unknown_id_((int) m.u32("tokenizer.unknown_id")),
      unknown_surface_(m.str("tokenizer.unknown_surface")),
      strip_leading_space_(m.u32("tokenizer.strip_leading_space") != 0) {}

std::string Detokenizer::text(const std::vector<int32_t> & ids) const {
    std::string text;
    for (int32_t id : ids) {
        if (id < 0 || (size_t) id >= pieces_.size()) throw std::runtime_error("token id " + std::to_string(id) + " is not in the vocabulary");
        if (id == unknown_id_) {
            text += unknown_surface_;
            continue;
        }
        std::string piece = pieces_[(size_t) id];
        // SentencePiece drops the leading "▁" of each piece until the text is no longer empty.
        if (strip_leading_space_ && text.empty() && piece.compare(0, kSpaceSymbol.size(), kSpaceSymbol) == 0) {
            piece.erase(0, kSpaceSymbol.size());
        }
        text += replace_all(piece, kSpaceSymbol, " ");
    }
    // The converter checks that no piece holds whitespace other than "▁" and that the unknown surface holds
    // only spaces, so the space is the one whitespace character the official pattern's \s can meet here.
    std::string out;
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == ' ') {
            bool before_mark = false;
            for (const std::string & p : punctuation_) before_mark = before_mark || text.compare(i + 1, p.size(), p) == 0;
            if (before_mark) continue;
        }
        out += text[i];
    }
    return out;
}

}  // namespace fastconformer
