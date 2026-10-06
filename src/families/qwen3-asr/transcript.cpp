#include "transcript.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "utf8.h"

namespace qwen3_asr {

namespace {

using Text = std::vector<uint32_t>;

/** The code points of UTF-8 that the tokenizer's decoding made, which replaces whatever is not UTF-8. */
Text code_points(const std::string & s) {
    std::optional<Text> text = decode_utf8(s);
    if (!text) throw std::logic_error("the decoded text is not UTF-8");
    return std::move(*text);
}

/** What Python's str.isspace() takes for whitespace, which str.strip() removes (Unicode 15.0, Python 3.12). */
bool python_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

Text stripped(const Text & text, size_t from = 0) {
    size_t first = from, last = text.size();
    while (first < last && python_space(text[first])) first++;
    while (last > first && python_space(text[last - 1])) last--;
    return Text(text.begin() + (std::ptrdiff_t) first, text.begin() + (std::ptrdiff_t) last);
}

/** fix_char_repeats(): a run of more than `threshold` of one character kept once. */
Text fix_character_runs(const Text & s, size_t threshold) {
    Text out;
    for (size_t i = 0; i < s.size();) {
        size_t count = 1;
        while (i + count < s.size() && s[i + count] == s[i]) count++;
        out.insert(out.end(), s.begin() + (std::ptrdiff_t) i, s.begin() + (std::ptrdiff_t) (count > threshold ? i + 1 : i + count));
        i += count;
    }
    return out;
}

bool same(const Text & s, size_t a, size_t b, size_t k) {
    return std::equal(s.begin() + (std::ptrdiff_t) a, s.begin() + (std::ptrdiff_t) (a + k), s.begin() + (std::ptrdiff_t) b);
}

/**
 * fix_pattern_repeats(), whose recursion on what follows a repeated pattern is a loop here: the official calls itself on
 * that rest and appends what it returns.
 */
Text fix_pattern_repeats(const Text & s, size_t threshold, size_t max_period) {
    const size_t min_chars = threshold * 2;
    Text out;
    for (size_t at = 0;;) {
        const size_t n = s.size() - at;
        if (n < min_chars) {
            out.insert(out.end(), s.begin() + (std::ptrdiff_t) at, s.end());
            return out;
        }
        size_t i = 0, rest = SIZE_MAX;
        for (; i <= n - min_chars && rest == SIZE_MAX; i++) {
            for (size_t k = 1; k <= max_period && i + k * threshold <= n; k++) {
                const size_t start = at + i;
                bool repeated = true;
                for (size_t r = 1; r < threshold && repeated; r++) repeated = same(s, start, start + r * k, k);
                if (!repeated) continue;
                size_t end = start + threshold * k;
                while (end + k <= s.size() && same(s, start, end, k)) end += k;
                out.insert(out.end(), s.begin() + (std::ptrdiff_t) start, s.begin() + (std::ptrdiff_t) (start + k));
                rest = end;
                break;
            }
            if (rest == SIZE_MAX) out.push_back(s[at + i]);
        }
        if (rest == SIZE_MAX) {
            out.insert(out.end(), s.begin() + (std::ptrdiff_t) (at + i), s.end());
            return out;
        }
        at = rest;
    }
}

}  // namespace

Transcript::Transcript(const ModelFile & m)
    : asr_text_(m.str("qwen3-asr.prompt.asr_text")),
      threshold_(m.u32("qwen3-asr.output.repetition_threshold")),
      max_period_(m.u32("qwen3-asr.output.repetition_max_period")) {}

std::string Transcript::fix_repetitions(const std::string & text) const {
    return encode_utf8(fix_pattern_repeats(fix_character_runs(code_points(text), threshold_), threshold_, max_period_));
}

std::string Transcript::text(const std::string & raw, bool forced) const {
    const Text s = stripped(code_points(raw));
    if (s.empty()) return "";
    const std::string fixed = fix_repetitions(encode_utf8(s));
    if (forced) return fixed;
    const size_t tag = fixed.find(asr_text_);
    const Text text = code_points(tag == std::string::npos ? fixed : fixed.substr(tag + asr_text_.size()));
    return encode_utf8(stripped(text));
}

}  // namespace qwen3_asr
