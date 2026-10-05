#include "detokenizer.h"

#include <stdexcept>

namespace fastconformer {

namespace {

const std::string kSpaceSymbol = "\xe2\x96\x81";

std::string replace_all(std::string s, const std::string & from, const std::string & to) {
    for (size_t at = s.find(from); at != std::string::npos; at = s.find(from, at + to.size())) s.replace(at, from.size(), to);
    return s;
}

}  // namespace

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
