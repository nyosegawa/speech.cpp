#include "ctc-tokenizer.h"

#include <algorithm>
#include <cstring>
#include <queue>
#include <string_view>

#include "error.h"

namespace nemotron_transcribe {

namespace {

const std::string kProto = "nemotron-transcribe.ctc.tokenizer.model_proto";
const std::string kSpace = "\xE2\x96\x81";

/** The types of SentencePiece's ModelProto.SentencePiece. */
enum PieceType { kNormal = 1, kUnknown = 2, kControl = 3, kUserDefined = 4, kUnused = 5, kByte = 6 };

/** A reader of the protocol buffers' wire format over [p, end); malformed data throws `fail`'s error. */
struct Wire {
    const uint8_t * p, * end;
    const ModelFile * m;

    [[noreturn]] void fail(const std::string & what) const {
        throw Error(Fault::File, m->path() + ": " + kProto + " is not a SentencePiece model this release reads (" + what + "); " + m->remedy());
    }
    bool done() const { return p >= end; }
    uint64_t varint() {
        uint64_t v = 0;
        for (int shift = 0;; shift += 7) {
            if (p >= end || shift > 63) fail("a varint runs past its message");
            const uint8_t b = *p++;
            v |= (uint64_t) (b & 0x7F) << shift;
            if (!(b & 0x80)) return v;
        }
    }
    Wire bytes() {
        const uint64_t n = varint();
        if (n > (uint64_t) (end - p)) fail("a field runs past its message");
        Wire w{p, p + n, m};
        p += n;
        return w;
    }
    uint32_t fixed32() {
        if (end - p < 4) fail("a field runs past its message");
        uint32_t v;
        std::memcpy(&v, p, 4);
        p += 4;
        return v;
    }
    void skip(int type) {
        if (type == 0) varint();
        else if (type == 2) bytes();
        else if (type == 5) fixed32();
        else if (type == 1 && end - p >= 8) p += 8;
        else fail("a field of wire type " + std::to_string(type));
    }
};

/** The length of the UTF-8 character whose first byte is `lead`, from that byte alone, as OneCharLen() takes it. */
size_t one_char_len(unsigned char lead) {
    static const size_t lengths[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    return lengths[lead >> 4];
}

/** The length of the valid UTF-8 character at the start of `s`, 0 where none starts, as IsValidDecodeUTF8() checks it. */
size_t valid_char_len(std::string_view s) {
    const unsigned char c = (unsigned char) s[0];
    if (c < 0x80) return 1;
    size_t n;
    uint32_t cp;
    if (c >= 0xC2 && c <= 0xDF) n = 2, cp = c & 0x1F;
    else if (c >= 0xE0 && c <= 0xEF) n = 3, cp = c & 0x0F;
    else if (c >= 0xF0 && c <= 0xF4) n = 4, cp = c & 0x07;
    else return 0;
    if (s.size() < n) return 0;
    for (size_t i = 1; i < n; i++) {
        const unsigned char d = (unsigned char) s[i];
        if ((d & 0xC0) != 0x80) return 0;
        cp = (cp << 6) | (d & 0x3F);
    }
    if ((n == 3 && (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF))) || (n == 4 && (cp < 0x10000 || cp > 0x10FFFF))) return 0;
    return n;
}

}  // namespace

CtcTokenizer::CtcTokenizer(const ModelFile & m) {
    const std::vector<uint8_t> proto = m.u8_array(kProto);
    Wire model{proto.data(), proto.data() + proto.size(), &m};
    bool bpe = false, byte_fallback = false, suffix = false, rules = false, denormalizer = false;
    std::string charsmap;
    while (!model.done()) {
        const uint64_t tag = model.varint();
        const uint64_t field = tag >> 3;
        const int type = (int) (tag & 7);
        if (field == 1 && type == 2) {
            Wire w = model.bytes();
            Piece piece{"", 0.0f, kNormal};
            while (!w.done()) {
                const uint64_t t = w.varint();
                if (t == (1 << 3 | 2)) {
                    const Wire s = w.bytes();
                    piece.text.assign((const char *) s.p, (size_t) (s.end - s.p));
                } else if (t == (2 << 3 | 5)) {
                    const uint32_t bits = w.fixed32();
                    std::memcpy(&piece.score, &bits, sizeof bits);
                } else if (t == (3 << 3 | 0)) {
                    piece.type = (int) w.varint();
                } else {
                    w.skip((int) (t & 7));
                }
            }
            pieces_.push_back(std::move(piece));
        } else if (field == 2 && type == 2) {
            Wire w = model.bytes();
            // A ModelProto's model_type defaults to UNIGRAM, 1; BPE is 2.
            while (!w.done()) {
                const uint64_t t = w.varint();
                if (t == (3 << 3 | 0)) bpe = w.varint() == 2;
                else if (t == (24 << 3 | 0)) suffix = w.varint() != 0;
                else if (t == (35 << 3 | 0)) byte_fallback = w.varint() != 0;
                else w.skip((int) (t & 7));
            }
        } else if (field == 3 && type == 2) {
            Wire w = model.bytes();
            while (!w.done()) {
                const uint64_t t = w.varint();
                if (t == (2 << 3 | 2)) {
                    const Wire s = w.bytes();
                    charsmap.assign((const char *) s.p, (size_t) (s.end - s.p));
                } else if (t == (3 << 3 | 0)) {
                    add_dummy_prefix_ = w.varint() != 0;
                } else if (t == (4 << 3 | 0)) {
                    remove_extra_whitespaces_ = w.varint() != 0;
                } else if (t == (5 << 3 | 0)) {
                    escape_whitespaces_ = w.varint() != 0;
                } else if (t == (6 << 3 | 2)) {
                    const Wire s = w.bytes();
                    rules = s.end != s.p;
                } else {
                    w.skip((int) (t & 7));
                }
            }
        } else if (field == 5 && type == 2) {
            denormalizer = true;
            model.bytes();
        } else {
            model.skip(type);
        }
    }
    if (!bpe) model.fail("its model is not BPE");
    if (byte_fallback || suffix || rules || denormalizer) model.fail("it falls back to bytes, takes whitespace as a suffix, or holds rules of normalization or denormalization");
    int unknowns = 0;
    for (size_t i = 0; i < pieces_.size(); i++) {
        const Piece & p = pieces_[i];
        if (p.type == kUnused || p.type == kByte) model.fail("it holds unused or byte pieces");
        const bool word = p.type == kNormal || p.type == kUserDefined;
        if (!(word ? vocabulary_ : reserved_).emplace(p.text, (int32_t) i).second) model.fail("the piece " + p.text + " is held twice");
        if (p.type == kUserDefined) user_defined_.push_back(p.text);
        if (p.type == kUnknown) unknown_ = (int32_t) i, unknowns++;
    }
    if (unknowns != 1) model.fail("it does not hold one unknown piece");

    // The character map: the size of the trie in bytes, the trie's units, and the normalized strings, each ending in NUL.
    if (!charsmap.empty()) {
        uint32_t trie_bytes = 0;
        if (charsmap.size() >= 4) std::memcpy(&trie_bytes, charsmap.data(), 4);
        if (charsmap.size() < 4 || trie_bytes % 4 != 0 || trie_bytes == 0 || trie_bytes > charsmap.size() - 4) model.fail("its character map is malformed");
        trie_.resize(trie_bytes / 4);
        std::memcpy(trie_.data(), charsmap.data() + 4, trie_bytes);
        normalized_ = charsmap.substr(4 + trie_bytes);
        if (normalized_.empty() || normalized_.back() != '\0') model.fail("its character map's strings do not end in NUL");
    }
}

size_t CtcTokenizer::user_defined_at(const std::string & text, size_t pos) const {
    size_t best = 0;
    for (const std::string & u : user_defined_) {
        if (u.size() > best && text.compare(pos, u.size(), u) == 0) best = u.size();
    }
    return best;
}

std::string CtcTokenizer::normalize(const std::string & text) const {
    // NormalizePrefix(): a user-defined piece as it is; else the longest prefix the trie holds, of the first 32 that its
    // commonPrefixSearch() gives, as its normalized string; else one character, or U+FFFD for a byte that starts none.
    auto prefix = [&](size_t pos) -> std::pair<std::string_view, size_t> {
        if (const size_t n = user_defined_at(text, pos)) return {std::string_view(text).substr(pos, n), n};
        size_t length = 0, found = 0;
        uint32_t value = 0;
        if (!trie_.empty()) {
            // darts-clone's DoubleArray: a unit's label is its low byte with bit 31, which a value unit sets; its leaf
            // flag bit 8; the offset to its children its bits 10 to 30, shifted by 8 more where bit 9 says so.
            auto offset = [](uint32_t unit) { return (unit >> 10) << ((unit & (1u << 9)) >> 6); };
            uint32_t id = offset(trie_[0]);
            for (size_t i = pos; i < text.size(); i++) {
                const uint8_t c = (uint8_t) text[i];
                id ^= c;
                if (id >= trie_.size() || (trie_[id] & ((1u << 31) | 0xFF)) != c) break;
                const uint32_t unit = trie_[id];
                id ^= offset(unit);
                if (id >= trie_.size()) break;
                if ((unit >> 8) & 1) {
                    if (found < 32) length = i - pos + 1, value = trie_[id] & ((1u << 31) - 1);
                    found++;
                }
            }
        }
        if (length > 0) {
            if (value >= normalized_.size()) throw Error(Fault::File, kProto + " maps a character past its strings");
            return {std::string_view(normalized_.c_str() + value), length};
        }
        const size_t n = valid_char_len(std::string_view(text).substr(pos));
        if (n == 0) return {"\xEF\xBF\xBD", 1};
        return {std::string_view(text).substr(pos, n), n};
    };
    const std::string space = escape_whitespaces_ ? kSpace : " ";
    std::string out;
    size_t pos = 0;
    if (remove_extra_whitespaces_) {
        while (pos < text.size()) {
            const auto p = prefix(pos);
            if (p.first != " ") break;
            pos += p.second;
        }
    }
    if (pos >= text.size()) return out;
    if (add_dummy_prefix_) out += space;
    bool previous_space = remove_extra_whitespaces_;
    while (pos < text.size()) {
        auto [sp, consumed] = prefix(pos);
        if (previous_space) {
            while (!sp.empty() && sp[0] == ' ') sp.remove_prefix(1);
        }
        if (!sp.empty()) {
            for (const char ch : sp) {
                if (escape_whitespaces_ && ch == ' ') out += kSpace;
                else out += ch;
            }
            previous_space = sp.back() == ' ';
        }
        pos += consumed;
        if (!remove_extra_whitespaces_) previous_space = false;
    }
    if (remove_extra_whitespaces_) {
        while (out.size() >= space.size() && out.compare(out.size() - space.size(), space.size(), space) == 0) out.resize(out.size() - space.size());
    }
    return out;
}

int32_t CtcTokenizer::piece_to_id(const std::string & piece) const {
    if (const auto r = reserved_.find(piece); r != reserved_.end()) return r->second;
    if (const auto v = vocabulary_.find(piece); v != vocabulary_.end()) return v->second;
    return unknown_;
}

std::vector<int32_t> CtcTokenizer::encode(const std::string & text) const {
    const std::string n = normalize(text);
    struct Symbol {
        size_t start, length;
        int prev, next;
        bool freeze;
    };
    std::vector<Symbol> symbols;
    for (size_t pos = 0; pos < n.size();) {
        size_t length = user_defined_at(n, pos);
        const bool freeze = length > 0;
        if (!freeze) length = std::min(n.size() - pos, one_char_len((unsigned char) n[pos]));
        const int index = (int) symbols.size();
        symbols.push_back({pos, length, index - 1, pos + length < n.size() ? index + 1 : -1, freeze});
        pos += length;
    }
    // The agenda of joins, the one of highest score first and, of equal scores, the leftmost (SymbolPairComparator).
    struct Pair {
        float score;
        int left, right;
        size_t size;
    };
    auto later = [](const Pair & a, const Pair & b) { return a.score < b.score || (a.score == b.score && a.left > b.left); };
    std::priority_queue<Pair, std::vector<Pair>, decltype(later)> agenda(later);
    auto consider = [&](int left, int right) {
        if (left < 0 || right < 0 || symbols[(size_t) left].freeze || symbols[(size_t) right].freeze) return;
        const size_t size = symbols[(size_t) left].length + symbols[(size_t) right].length;
        const auto it = vocabulary_.find(n.substr(symbols[(size_t) left].start, size));
        if (it == vocabulary_.end()) return;
        agenda.push({pieces_[(size_t) it->second].score, left, right, size});
    };
    for (size_t i = 1; i < symbols.size(); i++) consider((int) i - 1, (int) i);
    while (!agenda.empty()) {
        const Pair top = agenda.top();
        agenda.pop();
        Symbol & left = symbols[(size_t) top.left];
        Symbol & right = symbols[(size_t) top.right];
        if (left.length == 0 || right.length == 0 || left.length + right.length != top.size) continue;
        left.length += right.length;
        left.next = right.next;
        if (right.next >= 0) symbols[(size_t) right.next].prev = top.left;
        right.length = 0;
        consider(left.prev, top.left);
        consider(top.left, left.next);
    }
    std::vector<int32_t> ids;
    bool unknown = false;
    for (int i = symbols.empty() ? -1 : 0; i >= 0; i = symbols[(size_t) i].next) {
        const int32_t id = piece_to_id(n.substr(symbols[(size_t) i].start, symbols[(size_t) i].length));
        if (id == unknown_ && unknown) continue;
        ids.push_back(id);
        unknown = id == unknown_;
    }
    return ids;
}

std::string CtcTokenizer::decode(const std::vector<int32_t> & ids) const {
    std::string out;
    for (size_t i = 0; i < ids.size(); i++) {
        std::string_view p = piece(ids[i]);
        if (i == 0 && (add_dummy_prefix_ || remove_extra_whitespaces_) && p.substr(0, kSpace.size()) == kSpace) p.remove_prefix(kSpace.size());
        for (size_t k = 0; k < p.size();) {
            if (p.compare(k, kSpace.size(), kSpace) == 0) {
                out += ' ';
                k += kSpace.size();
            } else {
                out += p[k++];
            }
        }
    }
    return out;
}

}  // namespace nemotron_transcribe
