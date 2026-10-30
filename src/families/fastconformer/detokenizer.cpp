#include "detokenizer.h"

#include <algorithm>
#include <stdexcept>

#include "utf8.h"

namespace fastconformer {

namespace {

const std::string kSpaceSymbol = "\xe2\x96\x81";
const std::string kReplacement = "\xef\xbf\xbd";

std::string replace_all(std::string s, const std::string & from, const std::string & to) {
    for (size_t at = s.find(from); at != std::string::npos; at = s.find(from, at + to.size())) s.replace(at, from.size(), to);
    return s;
}

/**
 * Whether Python's \s, which the official pattern of whitespace before a punctuation mark uses, matches the code point:
 * the characters str.isspace() takes.
 */
bool whitespace(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

/**
 * The length of the UTF-8 character `bytes` begins at `at`, or 0 where none begins there, as SentencePiece's
 * DecodeUTF8() takes it: the length its first byte announces, if the bytes that follow make a well-formed character.
 */
size_t character_at(const std::string & bytes, size_t at) {
    const unsigned char c = (unsigned char) bytes[at];
    const size_t len = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
    if (len == 0 || at + len > bytes.size() || !decode_utf8(bytes.substr(at, len))) return 0;
    return len;
}

}  // namespace

Detokenizer::Detokenizer(const ModelFile & m)
    : pieces_(m.str_array("fastconformer.tokenizer.tokens")),
      punctuation_(m.str_array("fastconformer.tokenizer.punctuation")),
      unknown_id_((int) m.u32("fastconformer.tokenizer.unknown_id")),
      unknown_surface_(m.str("fastconformer.tokenizer.unknown_surface")),
      strip_leading_space_(m.boolean("fastconformer.tokenizer.strip_leading_space")) {
    const std::vector<int32_t> byte_pieces = m.i32_array("fastconformer.tokenizer.byte_pieces");
    if (byte_pieces.empty()) return;
    bytes_.assign(pieces_.size(), -1);
    for (size_t b = 0; b < byte_pieces.size(); b++) bytes_[(size_t) byte_pieces[b]] = (int) b;
}

int Detokenizer::byte_of(int32_t id) const {
    if (id < 0 || (size_t) id >= pieces_.size()) throw std::runtime_error("token id " + std::to_string(id) + " is not in the vocabulary");
    return bytes_.empty() ? -1 : bytes_[(size_t) id];
}

std::string Detokenizer::piece_text(int32_t id, bool empty) const {
    if (id == unknown_id_) return unknown_surface_;
    std::string piece = pieces_[(size_t) id];
    // SentencePiece drops the leading "▁" of each piece until the text is no longer empty.
    if (strip_leading_space_ && empty && piece.compare(0, kSpaceSymbol.size(), kSpaceSymbol) == 0) piece.erase(0, kSpaceSymbol.size());
    return replace_all(piece, kSpaceSymbol, " ");
}

std::vector<std::string> Detokenizer::surfaces(const std::vector<int32_t> & ids) const {
    std::vector<std::string> out(ids.size());
    bool empty = true;
    for (size_t i = 0; i < ids.size();) {
        if (byte_of(ids[i]) < 0) {
            out[i] = piece_text(ids[i], empty);
            empty = empty && out[i].empty();
            i++;
            continue;
        }
        std::string bytes;
        size_t end = i;
        for (; end < ids.size() && byte_of(ids[end]) >= 0; end++) bytes += (char) byte_of(ids[end]);
        for (size_t at = 0; at < bytes.size();) {
            const size_t len = character_at(bytes, at);
            out[i + at + std::max<size_t>(len, 1) - 1] = len ? bytes.substr(at, len) : kReplacement;
            at += std::max<size_t>(len, 1);
        }
        empty = false;
        i = end;
    }
    return out;
}

std::vector<std::string> Detokenizer::token_texts(const std::vector<int32_t> & ids) const {
    const std::vector<std::string> pieces = surfaces(ids);
    std::string text;
    for (const std::string & p : pieces) text += p;
    // A piece's surface is whole characters, a byte piece's included, so each character is tested where it stands.
    std::vector<std::string> out(pieces.size());
    size_t i = 0;
    for (size_t k = 0; k < pieces.size(); k++) {
        for (size_t at = 0; at < pieces[k].size();) {
            const size_t len = std::max<size_t>(character_at(pieces[k], at), 1);
            const auto c = decode_utf8(pieces[k].substr(at, len));
            bool before_mark = false;
            if (c && whitespace(c->front())) {
                for (const std::string & p : punctuation_) before_mark = before_mark || text.compare(i + len, p.size(), p) == 0;
            }
            if (!before_mark) out[k] += pieces[k].substr(at, len);
            at += len;
            i += len;
        }
    }
    return out;
}

std::string Detokenizer::text(const std::vector<int32_t> & ids) const {
    std::string text;
    for (const std::string & t : token_texts(ids)) text += t;
    return text;
}

std::string Detokenizer::decoded_alone(int32_t id) const {
    return surfaces({id}).front();
}

bool Detokenizer::punctuation(int32_t id) const {
    const std::string alone = decoded_alone(id);
    return std::find(punctuation_.begin(), punctuation_.end(), alone) != punctuation_.end();
}

std::vector<bool> Detokenizer::word_starts(const std::vector<int32_t> & ids) const {
    std::vector<bool> starts;
    for (int32_t id : ids) starts.push_back(decoded_alone(id) != pieces_[(size_t) id] && !punctuation(id));
    return starts;
}

}  // namespace fastconformer
