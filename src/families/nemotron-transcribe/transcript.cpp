#include "transcript.h"

#include <algorithm>
#include <optional>
#include <vector>

#include "utf8.h"

namespace nemotron_transcribe {

Guardrail::Guardrail(const ModelFile & m)
    : char_run_(m.u32("nemotron-transcribe.generation.char_run")),
      block_run_(m.u32("nemotron-transcribe.generation.block_run")),
      block_max_(m.u32("nemotron-transcribe.generation.block_max_chars")) {}

std::string Guardrail::apply(const std::string & text) const {
    // The decoded text is valid UTF-8: the tokenizer's decode writes U+FFFD for what is not.
    const std::optional<std::vector<uint32_t>> decoded = decode_utf8(text);
    if (!decoded) return text;
    const std::vector<uint32_t> & in = *decoded;

    // CHAR_RUN: a greedy run of one character, kept once where it is long enough.
    std::vector<uint32_t> once;
    once.reserve(in.size());
    for (size_t i = 0; i < in.size();) {
        size_t j = i + 1;
        while (j < in.size() && in[j] == in[i]) j++;
        once.insert(once.end(), j - i >= char_run_ ? 1 : j - i, in[i]);
        i = j;
    }

    // BLOCK_RUN: at each place, the shortest block that the next block_run - 1 copies of it follow, with every copy
    // after them taken too; where none does, the character is kept and the scan moves on.
    const std::vector<uint32_t> & s = once;
    std::vector<uint32_t> out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        bool matched = false;
        const size_t longest = std::min(block_max_, (s.size() - i) / block_run_);
        for (size_t len = 1; len <= longest && !matched; len++) {
            size_t copies = 1;
            while (i + (copies + 1) * len <= s.size() && std::equal(s.begin() + (std::ptrdiff_t) i, s.begin() + (std::ptrdiff_t) (i + len),
                                                                    s.begin() + (std::ptrdiff_t) (i + copies * len))) {
                copies++;
            }
            if (copies >= block_run_) {
                out.insert(out.end(), s.begin() + (std::ptrdiff_t) i, s.begin() + (std::ptrdiff_t) (i + len));
                i += copies * len;
                matched = true;
            }
        }
        if (!matched) out.push_back(s[i++]);
    }
    return encode_utf8(out);
}

bool Guardrail::loops(const std::string & text) const {
    const std::optional<std::vector<uint32_t>> decoded = decode_utf8(text);
    if (!decoded) return false;
    const std::vector<uint32_t> & s = *decoded;
    if (s.size() >= char_run_ && std::all_of(s.end() - (std::ptrdiff_t) char_run_, s.end(), [&](uint32_t c) { return c == s.back(); })) return true;
    // The text's last block_run * p characters repeat a block of p where the reversed end of the text matches itself
    // shifted by p for all but p of them: its Z-function at p, in time linear in the longest run a block can make.
    const size_t n = std::min(s.size(), block_run_ * block_max_);
    const std::vector<uint32_t> r(s.rbegin(), s.rbegin() + (std::ptrdiff_t) n);
    std::vector<size_t> z(n, 0);
    for (size_t i = 1, left = 0, right = 0; i < n; i++) {
        if (i < right) z[i] = std::min(right - i, z[i - left]);
        while (i + z[i] < n && r[z[i]] == r[i + z[i]]) z[i]++;
        if (i + z[i] > right) {
            left = i;
            right = i + z[i];
        }
    }
    for (size_t p = 1; p * block_run_ <= n; p++) {
        if (z[p] >= (block_run_ - 1) * p) return true;
    }
    return false;
}

}  // namespace nemotron_transcribe
