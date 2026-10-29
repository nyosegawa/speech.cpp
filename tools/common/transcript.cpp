#include "transcript.h"

#include <cstdint>
#include <optional>

#include "failure.h"
#include "json.h"
#include "unicode.h"
#include "utf8.h"

namespace {

/** The code point that begins `s` (or ends it, with `last`), or nothing for an empty or ill-formed one. */
std::optional<uint32_t> code_point(const std::string & s, bool last) {
    if (s.empty()) return std::nullopt;
    size_t at = 0, n = s.size();
    if (last) {
        at = n - 1;
        while (at > 0 && ((unsigned char) s[at] & 0xC0) == 0x80) at--;
        n -= at;
    } else {
        const unsigned char lead = (unsigned char) s[0];
        n = lead < 0x80 ? 1 : lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : 2;
    }
    const auto decoded = decode_utf8(s.substr(at, n));
    if (!decoded || decoded->size() != 1) return std::nullopt;
    return (*decoded)[0];
}

/**
 * Whether `cp` belongs to a script written without spaces between words, between whose texts a join adds none: Han,
 * Hiragana and Katakana with their blocks of radicals, symbols, punctuation and full-width forms. Thai and the other
 * scripts that leave out spaces between words put one between phrases, where a pause cuts a region.
 */
bool unspaced(uint32_t cp) {
    return (cp >= 0x2E80 && cp <= 0x2FDF) || (cp >= 0x3000 && cp <= 0x30FF) || (cp >= 0x31F0 && cp <= 0x33FF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0xFF00 && cp <= 0xFFEF) || (cp >= 0x1AFF0 && cp <= 0x1B16F) || (cp >= 0x20000 && cp <= 0x323AF);
}

/** Whether `cp` is a mark that ends a sentence or a phrase: . , ; : ? ! … and their Japanese, Chinese and full-width forms. */
bool closing_mark(uint32_t cp) {
    switch (cp) {
        case '.': case ',': case ';': case ':': case '?': case '!': case 0x2026: case 0x3001: case 0x3002:
        case 0xFF01: case 0xFF0C: case 0xFF0E: case 0xFF1A: case 0xFF1B: case 0xFF1F:
            return true;
        default:
            return false;
    }
}

/** The text a join puts between `before` and `after`: a space, unless either is written without spaces or has one there. */
std::string separator(const std::string & before, const std::string & after) {
    const std::optional<uint32_t> a = code_point(before, true), b = code_point(after, false);
    if (!a || !b || python_space(*a) || python_space(*b) || unspaced(*a) || unspaced(*b)) return "";
    return " ";
}

std::vector<Timed> timed(const speech_result * result, bool segments, bool speakers) {
    const size_t n = segments ? speech_result_segment_count(result) : speech_result_token_count(result);
    std::vector<Timed> out(n);
    for (size_t i = 0; i < n; i++) {
        const char * text = nullptr;
        check(segments ? speech_result_segment(result, i, &out[i].start, &out[i].end, &text)
                       : speech_result_token(result, i, &out[i].start, &out[i].end, &text));
        out[i].text = text;
        if (segments && speakers) out[i].speaker = speech_result_segment_speaker(result, i);
    }
    return out;
}

std::string timed_json(const std::vector<Timed> & items, bool speakers) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); i++) {
        out += std::string(i ? "," : "") + "{\"start\":" + json_number(items[i].start) + ",\"end\":" + json_number(items[i].end) +
               ",\"text\":" + json_string(items[i].text);
        if (speakers) out += ",\"speaker\":" + (items[i].speaker < 0 ? std::string("null") : std::to_string(items[i].speaker));
        out += "}";
    }
    return out + "]";
}

}  // namespace

Transcript transcript_of(const speech_result * result, bool timestamps, bool speakers) {
    Transcript t;
    t.text = speech_result_text(result);
    t.stop = speech_result_stop(result);
    for (size_t i = 0; i < speech_result_language_count(result); i++) t.languages.push_back(speech_result_language(result, i));
    if (timestamps || speakers) t.segments = timed(result, true, speakers);
    if (timestamps) t.tokens = timed(result, false, false);
    if (const char * untimed = speech_result_untimed(result)) t.untimed = untimed;
    return t;
}

std::vector<Timed> words_of(const std::vector<Timed> & tokens) {
    std::vector<Timed> out;
    for (size_t i = 0; i < tokens.size(); i++) {
        const Timed & t = tokens[i];
        const std::optional<uint32_t> first = code_point(t.text, false);
        const std::optional<uint32_t> before = i ? code_point(tokens[i - 1].text, true) : std::nullopt;
        const bool starts = out.empty() || (first && (python_space(*first) || unspaced(*first))) || (before && unspaced(*before));
        if (starts) out.push_back({t.start, t.end, "", -1});
        out.back().end = t.end;
        out.back().text += t.text;
    }
    for (Timed & w : out) {
        const std::optional<std::vector<uint32_t>> cps = decode_utf8(w.text);
        if (!cps) continue;
        size_t k = 0;
        while (k < cps->size() && python_space((*cps)[k])) k++;
        w.text = encode_utf8(std::vector<uint32_t>(cps->begin() + (std::ptrdiff_t) k, cps->end()));
    }
    return out;
}

void append_part(Transcript & whole, const Transcript & part, double offset) {
    if (part.stop == SPEECH_STOP_MODEL_LIMIT) whole.stop = SPEECH_STOP_MODEL_LIMIT;
    for (const std::string & language : part.languages) {
        if (whole.languages.empty() || whole.languages.back() != language) whole.languages.push_back(language);
    }
    if (part.untimed && !whole.untimed) {
        whole.untimed = part.untimed;
        whole.segments.clear();
        whole.tokens.clear();
    }
    if (part.text.empty()) return;
    const std::string space = separator(whole.text, part.text);
    whole.text += space + part.text;
    if (whole.untimed) return;
    for (const auto & [items, into] : {std::make_pair(&part.segments, &whole.segments), std::make_pair(&part.tokens, &whole.tokens)}) {
        for (size_t i = 0; i < items->size(); i++) {
            const Timed & t = (*items)[i];
            into->push_back({t.start + offset, t.end + offset, (i == 0 ? space : "") + t.text, t.speaker});
        }
    }
}

std::string agreed_beginning(const std::string & earlier, const std::string & later) {
    size_t n = 0;
    while (n < earlier.size()) {
        const unsigned char lead = (unsigned char) earlier[n];
        const size_t length = lead < 0x80 ? 1 : lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : 2;
        if (n + length > earlier.size() || n + length > later.size() || earlier.compare(n, length, later, n, length) != 0) break;
        n += length;
    }
    // Steps back over the code point before n while `drop` takes it.
    const auto back = [&](auto drop) {
        while (n > 0 && drop(code_point(earlier.substr(0, n), true))) {
            n--;
            while (n > 0 && ((unsigned char) earlier[n] & 0xC0) == 0x80) n--;
        }
    };
    const auto in_word = [](std::optional<uint32_t> cp) { return cp && !python_space(*cp) && !unspaced(*cp); };
    const bool ends_here = n == later.size();
    if (in_word(code_point(earlier.substr(0, n), true)) &&
        (in_word(code_point(earlier.substr(n), false)) || in_word(code_point(later.substr(n), false)))) {
        back(in_word);
    }
    // The end of the later reading is the end of its audio, which closes whatever was said last, and two readings that
    // end alike have most often heard a pause or a noise; what they agree on counts up to the mark or space before it.
    if (ends_here) {
        const auto closing = [](std::optional<uint32_t> cp) { return cp && (python_space(*cp) || closing_mark(*cp)); };
        back(closing);
        back([&](std::optional<uint32_t> cp) { return cp && !closing(cp); });
    }
    return earlier.substr(0, n);
}

std::string recognition_members(const Transcript & t, bool timestamps, bool speakers) {
    std::string out = ",\"text\":" + json_string(t.text) + ",\"stop\":\"" + speech_stop_name(t.stop) + "\"";
    for (size_t i = 0; i < t.languages.size(); i++) out += (i ? "," : ",\"languages\":[") + json_string(t.languages[i]);
    if (!t.languages.empty()) out += "]";
    if (timestamps || speakers) out += ",\"segments\":" + timed_json(t.segments, speakers);
    if (timestamps) out += ",\"tokens\":" + timed_json(t.tokens, false);
    if (t.untimed) out += ",\"untimed\":" + json_string(*t.untimed);
    return out;
}
