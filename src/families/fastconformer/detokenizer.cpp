#include "detokenizer.h"

#include <algorithm>
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

std::string Detokenizer::piece_text(int32_t id, bool empty) const {
    if (id < 0 || (size_t) id >= pieces_.size()) throw std::runtime_error("token id " + std::to_string(id) + " is not in the vocabulary");
    if (id == unknown_id_) return unknown_surface_;
    std::string piece = pieces_[(size_t) id];
    // SentencePiece drops the leading "▁" of each piece until the text is no longer empty.
    if (strip_leading_space_ && empty && piece.compare(0, kSpaceSymbol.size(), kSpaceSymbol) == 0) piece.erase(0, kSpaceSymbol.size());
    return replace_all(piece, kSpaceSymbol, " ");
}

std::vector<std::string> Detokenizer::token_texts(const std::vector<int32_t> & ids) const {
    std::vector<std::string> pieces;
    std::string text;
    for (int32_t id : ids) {
        pieces.push_back(piece_text(id, text.empty()));
        text += pieces.back();
    }
    // The converter checks that no piece holds whitespace other than "▁" and that the unknown surface holds
    // only spaces, so the space is the one whitespace character the official pattern's \s can meet here.
    std::vector<std::string> out(pieces.size());
    size_t i = 0;
    for (size_t k = 0; k < pieces.size(); k++) {
        for (char c : pieces[k]) {
            bool before_mark = false;
            if (c == ' ') {
                for (const std::string & p : punctuation_) before_mark = before_mark || text.compare(i + 1, p.size(), p) == 0;
            }
            if (!before_mark) out[k] += c;
            i++;
        }
    }
    return out;
}

std::string Detokenizer::text(const std::vector<int32_t> & ids) const {
    std::string text;
    for (const std::string & t : token_texts(ids)) text += t;
    return text;
}

bool Detokenizer::punctuation(int32_t id) const {
    const std::string alone = piece_text(id, true);
    return std::find(punctuation_.begin(), punctuation_.end(), alone) != punctuation_.end();
}

std::vector<bool> Detokenizer::word_starts(const std::vector<int32_t> & ids) const {
    std::vector<bool> starts;
    for (int32_t id : ids) {
        const std::string alone = piece_text(id, true);
        starts.push_back(alone != pieces_[(size_t) id] && !punctuation(id));
    }
    return starts;
}

}  // namespace fastconformer
