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

#include "unicode-data.inc"

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
