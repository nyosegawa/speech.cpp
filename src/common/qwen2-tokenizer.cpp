#include "qwen2-tokenizer.h"

#include <algorithm>
#include <climits>
#include <stdexcept>

#include "error.h"

namespace {

struct CodepointRange {
    uint32_t first, last;
};

#include "unicode-ranges.inc"

template <size_t N>
bool in_ranges(const CodepointRange (&ranges)[N], uint32_t cp) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < ranges[mid].first) hi = mid;
        else if (cp > ranges[mid].last) lo = mid + 1;
        else return true;
    }
    return false;
}

bool is_letter(uint32_t cp) { return in_ranges(kLetters, cp); }
bool is_number(uint32_t cp) { return in_ranges(kNumbers, cp); }
bool is_space(uint32_t cp) { return in_ranges(kSpaces, cp); }
bool is_crlf(uint32_t cp) { return cp == '\r' || cp == '\n'; }
/** Neither whitespace, a letter nor a number: [^\s\p{L}\p{N}]. */
bool is_other(uint32_t cp) { return !is_space(cp) && !is_letter(cp) && !is_number(cp); }

/**
 * The length of the UTF-8 sequence at `s[i]`, by the well-formed sequences of the Unicode Standard's table 3-7: a whole
 * character's when `whole` comes back true, and otherwise that of the longest start of one, at least one byte, which
 * the replacement of maximal subparts that Rust and Python follow turns into one U+FFFD.
 */
size_t utf8_sequence(const std::string & s, size_t i, bool & whole) {
    const unsigned char c = (unsigned char) s[i];
    whole = false;
    if (c < 0x80) {
        whole = true;
        return 1;
    }
    size_t len = 0;
    unsigned char lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) {
        len = 2;
    } else if (c >= 0xE0 && c <= 0xEF) {
        len = 3;
        if (c == 0xE0) lo = 0xA0;
        if (c == 0xED) hi = 0x9F;
    } else if (c >= 0xF0 && c <= 0xF4) {
        len = 4;
        if (c == 0xF0) lo = 0x90;
        if (c == 0xF4) hi = 0x8F;
    } else {
        return 1;
    }
    for (size_t k = 1; k < len; k++) {
        if (i + k >= s.size()) return k;
        const unsigned char cc = (unsigned char) s[i + k];
        if (cc < (k == 1 ? lo : 0x80) || cc > (k == 1 ? hi : 0xBF)) return k;
    }
    whole = true;
    return len;
}

/** The code point of the whole UTF-8 sequence of `len` bytes at `s[i]`. */
uint32_t code_point(const std::string & s, size_t i, size_t len) {
    const unsigned char c = (unsigned char) s[i];
    uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
    for (size_t k = 1; k < len; k++) cp = (cp << 6) | ((unsigned char) s[i + k] & 0x3F);
    return cp;
}

std::vector<uint32_t> decode_utf8(const std::string & s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        bool whole;
        const size_t len = utf8_sequence(s, i, whole);
        if (!whole) throw Error(Fault::InvalidArgument, "the text to tokenize is not valid UTF-8; give UTF-8");
        out.push_back(code_point(s, i, len));
        i += len;
    }
    return out;
}

std::string encode_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) s += (char) cp;
    else if (cp < 0x800) { s += (char) (0xC0 | (cp >> 6)); s += (char) (0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { s += (char) (0xE0 | (cp >> 12)); s += (char) (0x80 | ((cp >> 6) & 0x3F)); s += (char) (0x80 | (cp & 0x3F)); }
    else { s += (char) (0xF0 | (cp >> 18)); s += (char) (0x80 | ((cp >> 12) & 0x3F)); s += (char) (0x80 | ((cp >> 6) & 0x3F)); s += (char) (0x80 | (cp & 0x3F)); }
    return s;
}

/** Length of the pre-token at `i`, following the alternatives of the Qwen2 pattern in order. */
size_t match_at(const std::vector<uint32_t> & cps, size_t i) {
    const size_t n = cps.size();
    auto lower = [](uint32_t c) { return c < 128 ? (uint32_t) std::tolower((int) c) : c; };
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (cps[i] == '\'' && i + 1 < n) {
        const uint32_t a = lower(cps[i + 1]);
        const uint32_t b = i + 2 < n ? lower(cps[i + 2]) : 0;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
    }
    // [^\r\n\p{L}\p{N}]?\p{L}+
    {
        size_t j = i;
        if (!is_letter(cps[j]) && !is_crlf(cps[j]) && !is_number(cps[j]) && j + 1 < n && is_letter(cps[j + 1])) j++;
        if (is_letter(cps[j])) {
            while (j < n && is_letter(cps[j])) j++;
            return j - i;
        }
    }
    // \p{N}
    if (is_number(cps[i])) return 1;
    // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
    {
        size_t j = i;
        if (cps[j] == ' ' && j + 1 < n && is_other(cps[j + 1])) j++;
        if (is_other(cps[j])) {
            while (j < n && is_other(cps[j])) j++;
            while (j < n && is_crlf(cps[j])) j++;
            return j - i;
        }
    }
    // The remaining alternatives all start with whitespace; take the whole run first.
    size_t run = i;
    while (run < n && is_space(cps[run])) run++;
    // \s*[\r\n]+ : ends after the last \r or \n of the run.
    for (size_t j = run; j > i; j--) {
        if (is_crlf(cps[j - 1])) return j - i;
    }
    // \s+(?!\S) : the run, less its last character when something other than whitespace follows.
    if (run == n) return run - i;
    if (run - i >= 2) return run - i - 1;
    // \s+
    return run - i;
}

}  // namespace

Qwen2Tokenizer::Qwen2Tokenizer(const ModelFile & m, const std::string & prefix) : tokens_(m.str_array(prefix + ".tokens")) {
    for (size_t i = 0; i < tokens_.size(); i++) {
        if (!tokens_[i].empty()) vocab_[tokens_[i]] = (int32_t) i;
    }
    const std::vector<std::string> merges = m.str_array(prefix + ".merges");
    for (size_t r = 0; r < merges.size(); r++) {
        const size_t sp = merges[r].find(' ');
        ranks_[{merges[r].substr(0, sp), merges[r].substr(sp + 1)}] = (int) r;
    }
    // GPT-2's byte-to-unicode table: printable bytes map to themselves, the rest to 256 and up.
    int extra = 0;
    for (int b = 0; b < 256; b++) {
        const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        const uint32_t cp = printable ? (uint32_t) b : (uint32_t) (256 + extra++);
        byte_to_unicode_[b] = encode_utf8(cp);
        unicode_to_byte_[cp] = (unsigned char) b;
    }
}

std::vector<std::string> Qwen2Tokenizer::pre_tokenize(const std::string & text) const {
    const std::vector<uint32_t> cps = decode_utf8(text);
    std::vector<std::string> pieces;
    for (size_t i = 0; i < cps.size();) {
        const size_t len = match_at(cps, i);
        std::string piece;
        for (size_t k = i; k < i + len; k++) piece += encode_utf8(cps[k]);
        pieces.push_back(piece);
        i += len;
    }
    return pieces;
}

std::vector<int32_t> Qwen2Tokenizer::bpe(const std::string & piece) const {
    std::vector<std::string> symbols;
    for (unsigned char c : piece) symbols.push_back(byte_to_unicode_[c]);
    while (symbols.size() > 1) {
        int best = INT_MAX;
        size_t at = 0;
        for (size_t k = 0; k + 1 < symbols.size(); k++) {
            const auto it = ranks_.find({symbols[k], symbols[k + 1]});
            if (it != ranks_.end() && it->second < best) {
                best = it->second;
                at = k;
            }
        }
        if (best == INT_MAX) break;
        symbols[at] += symbols[at + 1];
        symbols.erase(symbols.begin() + at + 1);
    }
    std::vector<int32_t> ids;
    for (const std::string & s : symbols) {
        const auto it = vocab_.find(s);
        if (it == vocab_.end()) throw std::runtime_error("the tokenizer has no token for a byte sequence");
        ids.push_back(it->second);
    }
    return ids;
}

std::vector<int32_t> Qwen2Tokenizer::encode(const std::string & text) const {
    std::vector<int32_t> ids;
    for (const std::string & piece : pre_tokenize(text)) {
        const std::vector<int32_t> p = bpe(piece);
        ids.insert(ids.end(), p.begin(), p.end());
    }
    return ids;
}

std::string Qwen2Tokenizer::decode(const std::vector<int32_t> & ids) const {
    std::string bytes;
    for (int32_t id : ids) {
        if (id < 0 || (size_t) id >= tokens_.size()) continue;
        const std::string & token = tokens_[id];
        std::string mapped;
        bool byte_level = true;
        for (size_t i = 0; i < token.size() && byte_level;) {
            bool whole;
            const size_t len = utf8_sequence(token, i, whole);
            const auto it = whole ? unicode_to_byte_.find(code_point(token, i, len)) : unicode_to_byte_.end();
            if (it == unicode_to_byte_.end()) byte_level = false;
            else mapped += (char) it->second;
            i += len;
        }
        bytes += byte_level ? mapped : token;
    }
    std::string text;
    for (size_t i = 0; i < bytes.size();) {
        bool whole;
        const size_t len = utf8_sequence(bytes, i, whole);
        if (whole) text.append(bytes, i, len);
        else text += "\xEF\xBF\xBD";
        i += len;
    }
    return text;
}
