#include "transcript.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "unicode.h"
#include "utf8.h"

namespace qwen3_asr {

namespace {

using Text = std::vector<uint32_t>;

/**
 * The code points of UTF-8 that the tokenizer's decoding made, which replaces whatever is not UTF-8, or of a key the
 * layout checked to be ASCII.
 */
Text code_points(const std::string & s) {
    std::optional<Text> text = decode_utf8(s);
    if (!text) throw std::logic_error("the decoded text is not UTF-8");
    return std::move(*text);
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

/**
 * parse_asr_output()'s mark of audio without speech, which it looks for anywhere in the output before <asr_text>,
 * lowercased. It is the parse's own text, which the converter finds in qwen-asr's code, and no key holds it: a key would
 * need a layout that the files converted for layout 1 do not have.
 */
const Text kNoSpeech = {'l', 'a', 'n', 'g', 'u', 'a', 'g', 'e', ' ', 'n', 'o', 'n', 'e'};

/** Whether str.splitlines() ends a line at `cp`. It takes \r\n as one break, which splits as two here around an empty line. */
bool line_break(uint32_t cp) {
    return (cp >= 0x0A && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x1E) || cp == 0x85 || cp == 0x2028 || cp == 0x2029;
}

/**
 * Python's str.lower() of `cp` where it is ASCII: a capital of ASCII lowered, and the Kelvin sign as k; any other code
 * point is kept. No other code point outside ASCII lowers into ASCII alone: İ gives i and a combining dot, which a name
 * compared whole never equals and which the prefix and the mark of no speech, holding no i, never match.
 */
uint32_t lower(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + ('a' - 'A');
    return cp == 0x212A ? 'k' : cp;
}

/** Python's str.upper() of `cp` where it gives one capital of ASCII: a small letter of ASCII, ı or ſ, which give I and S. */
uint32_t upper(uint32_t cp) {
    if (cp >= 'a' && cp <= 'z') return cp - ('a' - 'A');
    return cp == 0x131 ? 'I' : cp == 0x17F ? 'S' : cp;
}

/**
 * Whether normalize_language_name() turns `value`, which is stripped, into `name`, which the layout checks to be ASCII
 * with no capital but its first letter: whether value[:1].upper() + value[1:].lower() is the name. Of the code points
 * outside ASCII, str.upper() turns ß and the ligatures into two or three capitals and str.lower() İ into i and a
 * combining dot, none of which such a name holds.
 */
bool normalizes_to(const Text & value, const std::vector<uint32_t> & name) {
    if (value.empty() || value.size() != name.size() || upper(value[0]) != name[0]) return false;
    return std::equal(value.begin() + 1, value.end(), name.begin() + 1, [](uint32_t v, uint32_t n) { return lower(v) == n; });
}

}  // namespace

Transcript::Transcript(const ModelFile & m)
    : asr_text_(m.str("qwen3-asr.prompt.asr_text")),
      prefix_(code_points(m.str("qwen3-asr.prompt.language_prefix"))),
      threshold_(m.u32("qwen3-asr.output.repetition_threshold")),
      max_period_(m.u32("qwen3-asr.output.repetition_max_period")) {
    for (const std::string & name : m.str_array("qwen3-asr.language_names")) names_.push_back(code_points(name));
}

std::string Transcript::fix_repetitions(const std::string & text) const {
    return encode_utf8(fix_pattern_repeats(fix_character_runs(code_points(text), threshold_), threshold_, max_period_));
}

Parsed Transcript::parse(const std::string & raw, std::optional<size_t> forced) const {
    Parsed out;
    const Text s = stripped(code_points(raw));
    // parse_asr_output() returns no language for an empty output before it looks at a forced one.
    if (s.empty()) return out;
    const std::string fixed = fix_repetitions(encode_utf8(s));
    if (forced) {
        out.text = fixed;
        out.language = forced;
        return out;
    }
    const size_t tag = fixed.find(asr_text_);
    if (tag == std::string::npos) {
        out.text = encode_utf8(stripped(code_points(fixed)));
        return out;
    }
    read_language(code_points(fixed.substr(0, tag)), out);
    out.text = encode_utf8(stripped(code_points(fixed.substr(tag + asr_text_.size()))));
    return out;
}

void Transcript::read_language(const Text & meta, Parsed & out) const {
    Text lowered(meta.size());
    std::transform(meta.begin(), meta.end(), lowered.begin(), lower);
    if (std::search(lowered.begin(), lowered.end(), kNoSpeech.begin(), kNoSpeech.end()) != lowered.end()) return;
    for (size_t start = 0; start < meta.size();) {
        size_t end = start;
        while (end < meta.size() && !line_break(meta[end])) end++;
        const Text line = stripped(Text(meta.begin() + (std::ptrdiff_t) start, meta.begin() + (std::ptrdiff_t) end));
        start = end + 1;
        const bool prefixed = line.size() >= prefix_.size() &&
                              std::equal(prefix_.begin(), prefix_.end(), line.begin(), [](uint32_t p, uint32_t c) { return lower(c) == p; });
        if (!prefixed) continue;
        const Text value = stripped(line, prefix_.size());
        if (value.empty()) return;
        for (size_t i = 0; i < names_.size(); i++) {
            if (normalizes_to(value, names_[i])) {
                out.language = i;
                return;
            }
        }
        out.other_language = encode_utf8(value);
        return;
    }
}

}  // namespace qwen3_asr
