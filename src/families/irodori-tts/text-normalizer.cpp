#include "text-normalizer.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#include "error.h"
#include "unicode.h"
#include "utf8.h"

namespace irodori {

namespace {

void replace_all(std::vector<uint32_t> & text, const std::vector<uint32_t> & from, const std::vector<uint32_t> & to) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < text.size();) {
        if (i + from.size() <= text.size() && std::equal(from.begin(), from.end(), text.begin() + i)) {
            out.insert(out.end(), to.begin(), to.end());
            i += from.size();
        } else {
            out.push_back(text[i++]);
        }
    }
    text = std::move(out);
}

std::vector<uint32_t> u32(const char * utf8) { return decode_utf8(utf8).value(); }

/** The UTF-8 of `text` without what Python's str.strip() removes at either end. */
std::string strip(const std::vector<uint32_t> & text) {
    size_t begin = 0, end = text.size();
    while (begin < end && python_space(text[begin])) begin++;
    while (end > begin && python_space(text[end - 1])) end--;
    return encode_utf8(std::vector<uint32_t>(text.begin() + begin, text.begin() + end));
}

/** strip_outer_brackets(): removes a pair of brackets as long as it encloses the whole text. */
void strip_outer_brackets(std::vector<uint32_t> & text) {
    static const std::vector<std::pair<uint32_t, uint32_t>> pairs = {
        {0x300C, 0x300D}, {0x300E, 0x300F}, {0xFF08, 0xFF09}, {0x3010, 0x3011}, {'(', ')'}};
    while (text.size() >= 2) {
        const uint32_t open = text.front(), close = text.back();
        const auto pair = std::find_if(pairs.begin(), pairs.end(), [&](const auto & p) { return p.first == open; });
        if (pair == pairs.end() || pair->second != close) return;
        int depth = 0;
        bool encloses_all = true;
        for (size_t i = 0; i < text.size(); i++) {
            if (text[i] == open) depth++;
            else if (text[i] == close) depth--;
            if (depth == 0 && i + 1 < text.size()) {
                encloses_all = false;
                break;
            }
        }
        if (!encloses_all || depth != 0) return;
        text = std::vector<uint32_t>(text.begin() + 1, text.end() - 1);
    }
}

}  // namespace

std::string normalize_text(const std::string & raw) {
    std::optional<std::vector<uint32_t>> decoded = decode_utf8(raw);
    if (!decoded) throw Error(Fault::InvalidArgument, "the text is not valid UTF-8", "text");
    std::vector<uint32_t> text = std::move(*decoded);
    static const std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> simple = {
        {u32("\t"), {}}, {u32("[n]"), {}}, {u32("\\[n\\]"), {}}, {u32("　"), {}}, {u32("？"), u32("?")},
        {u32("！"), u32("!")}, {u32("♥"), u32("♡")}, {u32("●"), u32("○")}, {u32("◯"), u32("○")},
        {u32("〇"), u32("○")}};
    for (const auto & [from, to] : simple) replace_all(text, from, to);

    static const std::vector<uint32_t> removed = u32(";▼♀♂《》≪≫①②③④⑤⑥");
    auto is_dash = [](uint32_t c) {
        return c == 0x02D7 || (c >= 0x2010 && c <= 0x2015) || c == 0x2043 || c == 0x2212 || c == 0x23AF ||
               c == 0x23E4 || c == 0x2500 || c == 0x2501 || c == 0x2E3A || c == 0x2E3B;
    };
    text.erase(std::remove_if(text.begin(), text.end(), [&](uint32_t c) {
                   return std::find(removed.begin(), removed.end(), c) != removed.end() || is_dash(c);
               }),
               text.end());
    for (uint32_t & c : text) {
        if (c == 0xFF5E || c == 0x301C) c = 0x30FC;
    }
    // A run of three or more ellipses becomes two.
    std::vector<uint32_t> runs;
    for (size_t i = 0; i < text.size();) {
        size_t j = i;
        while (j < text.size() && text[j] == 0x2026) j++;
        if (j - i >= 3) {
            runs.push_back(0x2026);
            runs.push_back(0x2026);
            i = j;
        } else if (j > i) {
            runs.insert(runs.end(), text.begin() + i, text.begin() + j);
            i = j;
        } else {
            runs.push_back(text[i++]);
        }
    }
    text = std::move(runs);

    strip_outer_brackets(text);
    text = nfkc(text, UnicodeVersion::V13);
    replace_all(text, u32("..."), {0x2026});
    replace_all(text, u32(".."), {0x2026});

    return strip(text);
}

std::string strip_caption(const std::string & caption) {
    std::optional<std::vector<uint32_t>> decoded = decode_utf8(caption);
    if (!decoded) throw Error(Fault::InvalidArgument, "the instructions are not valid UTF-8", "instructions");
    return strip(*decoded);
}

}  // namespace irodori
