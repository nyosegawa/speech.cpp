#include "sentencepiece.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <queue>

#include "error.h"

namespace fastconformer {

namespace {

constexpr std::string_view kSpaceSymbol = "\xe2\x96\x81";
constexpr std::string_view kReplacementCharacter = "\xef\xbf\xbd";
// Normalizer::kMaxTrieResultsSize: of the keys of the charsmap a text begins with, the normalizer takes the longest of
// the 32 shortest.
constexpr size_t kMaxTrieResults = 32;
// ModelInterface::InitializePieces() refuses a piece of this length or longer.
constexpr size_t kMaxPieceSize = 8000;

/** The length of the character a byte begins, as OneCharLen() of SentencePiece's util.h tells it from the byte alone. */
size_t one_char_len(char c) {
    return (size_t) "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(unsigned char) c >> 4];
}

bool trail_byte(char c) {
    return ((unsigned char) c & 0xC0) == 0x80;
}

bool valid_code_point(uint32_t c) {
    return c < 0xD800 || (c >= 0xE000 && c <= 0x10FFFF);
}

/**
 * The length of the well-formed UTF-8 character `s` begins with, or 0 where none begins (DecodeUTF8() of util.cc, as
 * IsValidDecodeUTF8() tells it: a U+FFFD written out in the text is a character like any other).
 */
size_t utf8_character(std::string_view s) {
    const unsigned char b0 = (unsigned char) s[0];
    if (b0 < 0x80) return 1;
    if (s.size() >= 2 && (b0 & 0xE0) == 0xC0) {
        const uint32_t c = ((b0 & 0x1Fu) << 6) | (s[1] & 0x3Fu);
        return trail_byte(s[1]) && c >= 0x80 && valid_code_point(c) ? 2 : 0;
    }
    if (s.size() >= 3 && (b0 & 0xF0) == 0xE0) {
        const uint32_t c = ((b0 & 0x0Fu) << 12) | ((s[1] & 0x3Fu) << 6) | (s[2] & 0x3Fu);
        return trail_byte(s[1]) && trail_byte(s[2]) && c >= 0x800 && valid_code_point(c) ? 3 : 0;
    }
    if (s.size() >= 4 && (b0 & 0xF8) == 0xF0) {
        const uint32_t c = ((b0 & 0x07u) << 18) | ((s[1] & 0x3Fu) << 12) | ((s[2] & 0x3Fu) << 6) | (s[3] & 0x3Fu);
        return trail_byte(s[1]) && trail_byte(s[2]) && trail_byte(s[3]) && c >= 0x10000 && valid_code_point(c) ? 4 : 0;
    }
    return 0;
}

uint32_t unit_label(uint32_t unit) {
    return unit & ((1u << 31) | 0xFFu);
}

bool unit_has_leaf(uint32_t unit) {
    return ((unit >> 8) & 1u) == 1u;
}

uint32_t unit_value(uint32_t unit) {
    return unit & ((1u << 31) - 1);
}

uint32_t unit_offset(uint32_t unit) {
    return (unit >> 10) << ((unit & (1u << 9)) >> 6);
}

/**
 * Whether every node of the double array has its children inside it and every leaf a value below `values`, which
 * DoubleArray::validate() checks of a charsmap before the normalizer searches it without bounds.
 */
bool valid_trie(const std::vector<uint32_t> & units, size_t values) {
    const uint32_t root = units[0];
    if (unit_label(root) != 0 || unit_has_leaf(root) || unit_offset(root) == 0 || (unit_offset(root) | 0xFFu) >= units.size()) return false;
    for (size_t i = 1; i < units.size(); i++) {
        if (unit_label(units[i]) <= 0xFF) {
            if (((i ^ unit_offset(units[i])) | 0xFFu) >= units.size()) return false;
        } else if (unit_value(units[i]) >= values) {
            return false;
        }
    }
    return true;
}

std::string byte_piece_name(int byte) {
    char name[8];
    std::snprintf(name, sizeof name, "<0x%02X>", byte);
    return name;
}

std::string type_name(PieceType type) {
    switch (type) {
        case PieceType::Control: return "a control piece";
        case PieceType::UserDefined: return "a user-defined piece";
        case PieceType::Unused: return "an unused piece";
        default: return "of the type " + std::to_string((int32_t) type);
    }
}

/** A symbol of the text: its bytes [begin, begin + size) and its neighbours, -1 at either end; merged away when empty. */
struct Symbol {
    int prev;
    int next;
    size_t begin;
    size_t size;
};

/** Two neighbouring symbols whose bytes together, `size` of them, are a piece of the score whose bits `score` holds. */
struct SymbolPair {
    int32_t score;
    int left;
    int right;
    size_t size;
};

/**
 * Whether `a` merges after `b` (SymbolPairComparator of bpe_model.cc): a lower score, or the same score further right.
 * The scores compare by their bits, so that -0.0, the score of the first merge a trainer writes, ranks above every
 * other negative score and below +0.0, which float comparison would take for equals.
 */
struct MergesLater {
    bool operator()(const SymbolPair & a, const SymbolPair & b) const {
        const int32_t i1 = a.score, i2 = b.score;
        if ((i1 & i2) < 0) return i1 > i2 || (i1 == i2 && a.left > b.left);
        const bool score_less = (i1 ^ i2) < 0 ? i1 < 0 : i1 < i2;
        return score_less || (i1 == i2 && a.left > b.left);
    }
};

}  // namespace

PieceEncoder::PieceEncoder(PieceModel model) : model_(std::move(model)) {
    const size_t n = model_.pieces.size();
    if (model_.scores.size() != n || model_.types.size() != n) {
        throw Error(Fault::File, "the tokenizer has " + std::to_string(n) + " pieces but " + std::to_string(model_.scores.size()) + " scores and " +
                                     std::to_string(model_.types.size()) + " types; convert the model again");
    }
    std::unordered_map<std::string, int32_t> ids;
    for (size_t i = 0; i < n; i++) {
        const std::string & piece = model_.pieces[i];
        const int32_t id = (int32_t) i;
        if (piece.empty() || piece.size() >= kMaxPieceSize || piece.find('\0') != std::string::npos) {
            throw Error(Fault::File, "the tokenizer's piece " + std::to_string(id) + " is empty, holds a NUL or is " + std::to_string(kMaxPieceSize) +
                                         " bytes long or longer; convert the model again");
        }
        if (!ids.emplace(piece, id).second) throw Error(Fault::File, "the tokenizer has the piece \"" + piece + "\" twice; convert the model again");
        switch (model_.types[i]) {
            case PieceType::Normal:
                mergeable_.emplace(piece, id);
                break;
            case PieceType::Unknown:
                if (unknown_id_ >= 0) throw Error(Fault::File, "the tokenizer has two unknown pieces; convert the model again");
                unknown_id_ = id;
                break;
            case PieceType::Byte:
                byte_pieces_.push_back(id);
                break;
            default:
                throw Error(Fault::File, "the tokenizer's piece \"" + piece + "\" is " + type_name(model_.types[i]) +
                                             ", which speech.cpp cannot encode text with; use a model whose tokenizer has only normal, unknown and byte pieces");
        }
    }
    if (unknown_id_ < 0) throw Error(Fault::File, "the tokenizer has no unknown piece; convert the model again");
    // A model has byte pieces only with byte fallback, and then all 256 of them (InitializePieces()), in any order of
    // ids; the bytes take them by name.
    if (!byte_pieces_.empty()) {
        if (byte_pieces_.size() != 256) {
            throw Error(Fault::File, "the tokenizer has " + std::to_string(byte_pieces_.size()) + " byte pieces, not 256; convert the model again");
        }
        for (int b = 0; b < 256; b++) {
            const auto it = ids.find(byte_piece_name(b));
            if (it == ids.end() || model_.types[(size_t) it->second] != PieceType::Byte) {
                throw Error(Fault::File, "the tokenizer has no byte piece " + byte_piece_name(b) + "; convert the model again");
            }
            byte_pieces_[(size_t) b] = it->second;
        }
    }

    // The charsmap is the size of the double array in 4 little-endian bytes, the array, and the replacements
    // (DecodePrecompiledCharsMap() of normalizer.cc). An empty charsmap rewrites nothing.
    const std::string & charsmap = model_.charsmap;
    if (charsmap.empty()) return;
    const std::string broken = "the tokenizer's precompiled charsmap is broken; convert the model again";
    if (charsmap.size() <= 4) throw Error(Fault::File, broken);
    uint32_t trie_size = 0;
    for (int k = 3; k >= 0; k--) trie_size = (trie_size << 8) | (unsigned char) charsmap[(size_t) k];
    if (trie_size >= charsmap.size() - 4 || trie_size < 1024 || (trie_size & 0x3FF) != 0) throw Error(Fault::File, broken);
    trie_.resize(trie_size / 4);
    for (size_t u = 0; u < trie_.size(); u++) {
        const unsigned char * b = (const unsigned char *) charsmap.data() + 4 + 4 * u;
        trie_[u] = (uint32_t) b[0] | ((uint32_t) b[1] << 8) | ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24);
    }
    replacements_ = charsmap.substr(4 + trie_size);
    if (replacements_.back() != '\0' || !valid_trie(trie_, replacements_.size())) throw Error(Fault::File, broken);
}

std::pair<std::string_view, size_t> PieceEncoder::normalize_prefix(std::string_view input) const {
    size_t longest_length = 0;
    size_t longest_value = 0;
    if (!trie_.empty()) {
        // DoubleArray::commonPrefixSearch() finds the keys the input begins with, the shortest first, and the
        // normalizer takes the longest of the first kMaxTrieResults.
        size_t matches = 0;
        size_t node = unit_offset(trie_[0]);
        for (size_t i = 0; i < input.size(); i++) {
            const unsigned char b = (unsigned char) input[i];
            node ^= b;
            const uint32_t unit = trie_[node];
            if (unit_label(unit) != b) break;
            node ^= unit_offset(unit);
            if (!unit_has_leaf(unit)) continue;
            if (matches < kMaxTrieResults) {
                longest_length = i + 1;
                longest_value = unit_value(trie_[node]);
            }
            matches++;
        }
    }
    if (longest_length == 0 || longest_length > input.size() || longest_value >= replacements_.size()) {
        const size_t length = utf8_character(input);
        // A byte that begins no character becomes U+FFFD, and the next byte is read as the start of one.
        if (length == 0) return {kReplacementCharacter, 1};
        return {input.substr(0, length), length};
    }
    // The replacement runs to its NUL.
    return {std::string_view(replacements_.data() + longest_value), longest_length};
}

std::string PieceEncoder::normalize(const std::string & text) const {
    std::string normalized;
    std::string_view input = text;
    if (model_.remove_extra_whitespaces) {
        while (!input.empty()) {
            const auto p = normalize_prefix(input);
            if (p.first != " ") break;
            input.remove_prefix(p.second);
        }
    }
    if (input.empty()) return normalized;

    const std::string_view space = model_.escape_whitespaces ? kSpaceSymbol : std::string_view(" ");
    if (model_.add_dummy_prefix) normalized += space;
    bool prev_space = model_.remove_extra_whitespaces;
    while (!input.empty()) {
        const auto p = normalize_prefix(input);
        std::string_view sp = p.first;
        while (prev_space && !sp.empty() && sp.front() == ' ') sp.remove_prefix(1);
        if (!sp.empty()) {
            for (char c : sp) {
                if (c == ' ') {
                    normalized += space;
                } else {
                    normalized += c;
                }
            }
            prev_space = sp.back() == ' ';
        }
        input.remove_prefix(p.second);
        if (!model_.remove_extra_whitespaces) prev_space = false;
    }
    if (model_.remove_extra_whitespaces) {
        while (normalized.size() >= space.size() && normalized.compare(normalized.size() - space.size(), space.size(), space) == 0) {
            normalized.resize(normalized.size() - space.size());
        }
    }
    return normalized;
}

std::vector<int32_t> PieceEncoder::encode_normalized(const std::string & normalized) const {
    if (normalized.empty()) return {};
    const std::string_view text = normalized;

    // A symbol per character, by the length its first byte announces.
    std::vector<Symbol> symbols;
    for (size_t at = 0; at < text.size();) {
        const size_t length = std::min(text.size() - at, one_char_len(text[at]));
        const int index = (int) symbols.size();
        symbols.push_back({index - 1, at + length < text.size() ? index + 1 : -1, at, length});
        at += length;
    }

    std::priority_queue<SymbolPair, std::vector<SymbolPair>, MergesLater> agenda;
    auto add_pair = [&](int left, int right) {
        if (left == -1 || right == -1) return;
        const size_t size = symbols[(size_t) left].size + symbols[(size_t) right].size;
        const auto it = mergeable_.find(std::string(text.substr(symbols[(size_t) left].begin, size)));
        if (it == mergeable_.end()) return;
        SymbolPair pair{0, left, right, size};
        std::memcpy(&pair.score, &model_.scores[(size_t) it->second], sizeof pair.score);
        agenda.push(pair);
    };
    for (size_t i = 1; i < symbols.size(); i++) add_pair((int) i - 1, (int) i);

    while (!agenda.empty()) {
        const SymbolPair top = agenda.top();
        agenda.pop();
        Symbol & left = symbols[(size_t) top.left];
        Symbol & right = symbols[(size_t) top.right];
        // A pair whose symbols have merged since is stale.
        if (left.size == 0 || right.size == 0 || left.size + right.size != top.size) continue;
        left.size += right.size;
        left.next = right.next;
        if (right.next >= 0) symbols[(size_t) right.next].prev = top.left;
        right.size = 0;
        add_pair(left.prev, top.left);
        add_pair(top.left, left.next);
    }

    std::vector<int32_t> ids;
    bool prev_unknown = false;
    for (int index = 0; index != -1; index = symbols[(size_t) index].next) {
        const std::string_view piece = text.substr(symbols[(size_t) index].begin, symbols[(size_t) index].size);
        // SentencePiece looks a symbol up among all the pieces, but a symbol is one character or a piece a merge formed:
        // a byte piece's name is six characters, and the unknown piece is unknown either way.
        const auto it = mergeable_.find(std::string(piece));
        const bool unknown = it == mergeable_.end();
        if (unknown && !byte_pieces_.empty()) {
            for (char c : piece) ids.push_back(byte_pieces_[(unsigned char) c]);
        } else if (!unknown) {
            ids.push_back(it->second);
        } else if (!prev_unknown) {
            // Without byte fallback, a run of pieces the model lacks is one unknown piece.
            ids.push_back(unknown_id_);
        }
        prev_unknown = unknown;
    }
    return ids;
}

std::vector<int32_t> PieceEncoder::encode(const std::string & text) const {
    return encode_normalized(normalize(text));
}

}  // namespace fastconformer
