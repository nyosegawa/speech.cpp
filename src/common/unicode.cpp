#include "unicode.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>

namespace {

/** A decomposition mapping of one level, and the first of the versions that has it. */
struct Decomposition {
    uint32_t code_point;
    uint32_t offset;
    uint8_t length;
    bool compatibility;
    uint8_t since;
};

struct CombiningClassRange {
    uint32_t first, last;
    uint8_t combining_class;
    uint8_t since;
};

struct Composition {
    uint32_t first, second, composite;
    uint8_t since;
};

struct CodepointRange {
    uint32_t first, last;
};

/** What str.lower() makes of a code point alone: `length` code points of kLowercasePool from `offset`. */
struct CaseMapping {
    uint32_t code_point;
    uint32_t offset;
    uint8_t length;
};

#include "unicode-data.inc"
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

constexpr uint32_t kHangulFirst = 0xAC00, kLeadFirst = 0x1100, kVowelFirst = 0x1161, kTrailFirst = 0x11A7;
constexpr uint32_t kLeads = 19, kVowels = 21, kTrails = 28, kSyllables = kLeads * kVowels * kTrails;

bool has(uint8_t since, UnicodeVersion version) { return since <= (uint8_t) version; }

int combining_class(uint32_t cp, UnicodeVersion version) {
    const auto * end = std::end(kCombiningClasses);
    const auto * it = std::upper_bound(std::begin(kCombiningClasses), end, cp,
                                       [](uint32_t c, const CombiningClassRange & r) { return c < r.first; });
    if (it == std::begin(kCombiningClasses)) return 0;
    --it;
    return cp <= it->last && has(it->since, version) ? it->combining_class : 0;
}

/** The full decomposition of `cp`, canonical alone or with the compatibility mappings, before canonical ordering. */
void decompose(uint32_t cp, bool compatibility, UnicodeVersion version, std::vector<uint32_t> & out) {
    if (cp >= kHangulFirst && cp < kHangulFirst + kSyllables) {
        const uint32_t s = cp - kHangulFirst;
        out.push_back(kLeadFirst + s / (kVowels * kTrails));
        out.push_back(kVowelFirst + s % (kVowels * kTrails) / kTrails);
        if (s % kTrails) out.push_back(kTrailFirst + s % kTrails);
        return;
    }
    const auto * end = std::end(kDecompositions);
    const auto * it = std::lower_bound(std::begin(kDecompositions), end, cp,
                                       [](const Decomposition & d, uint32_t c) { return d.code_point < c; });
    if (it == end || it->code_point != cp || !has(it->since, version) || (it->compatibility && !compatibility)) {
        out.push_back(cp);
        return;
    }
    for (uint32_t i = it->offset; i < it->offset + it->length; i++) {
        decompose(kDecompositionPool[i], compatibility, version, out);
    }
}

/** The canonical composite of a starter and the character after it, or 0 when they do not compose. */
uint32_t compose(uint32_t a, uint32_t b, UnicodeVersion version) {
    if (a >= kLeadFirst && a < kLeadFirst + kLeads && b >= kVowelFirst && b < kVowelFirst + kVowels) {
        return kHangulFirst + ((a - kLeadFirst) * kVowels + (b - kVowelFirst)) * kTrails;
    }
    if (a >= kHangulFirst && a < kHangulFirst + kSyllables && (a - kHangulFirst) % kTrails == 0 && b > kTrailFirst &&
        b < kTrailFirst + kTrails) {
        return a + (b - kTrailFirst);
    }
    const auto * end = std::end(kCompositions);
    const auto * it = std::lower_bound(std::begin(kCompositions), end, std::make_pair(a, b),
                                       [](const Composition & c, const std::pair<uint32_t, uint32_t> & k) {
                                           return c.first < k.first || (c.first == k.first && c.second < k.second);
                                       });
    return it != end && it->first == a && it->second == b && has(it->since, version) ? it->composite : 0;
}

std::vector<uint32_t> normalize(const std::vector<uint32_t> & text, bool compatibility, UnicodeVersion version) {
    std::vector<uint32_t> d;
    for (uint32_t cp : text) decompose(cp, compatibility, version, d);
    const auto lower_class = [version](uint32_t a, uint32_t b) { return combining_class(a, version) < combining_class(b, version); };
    // Canonical ordering: each run of non-starters sorted by combining class, keeping the order of equals.
    for (size_t i = 0; i < d.size();) {
        if (combining_class(d[i], version) == 0) {
            i++;
            continue;
        }
        size_t j = i;
        while (j < d.size() && combining_class(d[j], version) != 0) j++;
        std::stable_sort(d.begin() + (std::ptrdiff_t) i, d.begin() + (std::ptrdiff_t) j, lower_class);
        i = j;
    }
    std::vector<uint32_t> out;
    size_t starter = SIZE_MAX;
    // The combining class of the last character kept after the starter, -1 when the starter is the last.
    int last_class = -1;
    for (uint32_t cp : d) {
        const int cc = combining_class(cp, version);
        if (starter != SIZE_MAX && (last_class == -1 || last_class < cc)) {
            if (const uint32_t composite = compose(out[starter], cp, version)) {
                out[starter] = composite;
                continue;
            }
        }
        if (cc == 0) {
            starter = out.size();
            last_class = -1;
        } else {
            last_class = cc;
        }
        out.push_back(cp);
    }
    return out;
}

constexpr uint32_t kCapitalSigma = 0x3A3, kSmallSigma = 0x3C3, kFinalSigma = 0x3C2;

/**
 * Whether the capital sigma at `i` of `text` is a final sigma, as handle_capital_sigma() of CPython's unicodeobject.c
 * decides: a cased character before it and none after it, each the first that is not case-ignorable.
 */
bool final_sigma(const std::vector<uint32_t> & text, size_t i) {
    size_t before = i;
    while (before > 0 && in_ranges(kCaseIgnorable, text[before - 1])) before--;
    if (before == 0 || !in_ranges(kCased, text[before - 1])) return false;
    size_t after = i + 1;
    while (after < text.size() && in_ranges(kCaseIgnorable, text[after])) after++;
    return after == text.size() || !in_ranges(kCased, text[after]);
}

}  // namespace

std::vector<uint32_t> nfc(const std::vector<uint32_t> & text, UnicodeVersion version) {
    return normalize(text, false, version);
}

std::vector<uint32_t> nfkc(const std::vector<uint32_t> & text, UnicodeVersion version) {
    return normalize(text, true, version);
}

bool python_space(uint32_t cp) {
    for (const CodepointRange & r : kWhitespace) {
        if (cp >= r.first && cp <= r.last) return true;
    }
    return false;
}

std::vector<uint32_t> python_lower(const std::vector<uint32_t> & text) {
    std::vector<uint32_t> out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size(); i++) {
        const uint32_t cp = text[i];
        if (cp == kCapitalSigma) {
            out.push_back(final_sigma(text, i) ? kFinalSigma : kSmallSigma);
            continue;
        }
        const auto * end = std::end(kLowercase);
        const auto * it = std::lower_bound(std::begin(kLowercase), end, cp,
                                           [](const CaseMapping & m, uint32_t c) { return m.code_point < c; });
        if (it == end || it->code_point != cp) {
            out.push_back(cp);
            continue;
        }
        out.insert(out.end(), kLowercasePool + it->offset, kLowercasePool + it->offset + it->length);
    }
    return out;
}

bool python_digit(uint32_t cp) { return in_ranges(kDigits, cp); }

bool unicode_letter(uint32_t cp) { return in_ranges(kLetters, cp); }
bool unicode_number(uint32_t cp) { return in_ranges(kNumbers, cp); }
bool unicode_mark(uint32_t cp) { return in_ranges(kMarks, cp); }
bool unicode_space(uint32_t cp) { return in_ranges(kSpaces, cp); }
bool unicode_upper(uint32_t cp) { return in_ranges(kUppers, cp); }
bool unicode_lower(uint32_t cp) { return in_ranges(kLowers, cp); }
bool unicode_other_letter(uint32_t cp) { return in_ranges(kOtherLetters, cp); }
