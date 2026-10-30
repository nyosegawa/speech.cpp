#include "guard.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "utf8.h"

namespace fastconformer {

namespace {

using Text = std::vector<uint32_t>;

/** A run a[a:a+size] equal to b[b:b+size]. */
struct Match {
    size_t a, b, size;
};

/**
 * difflib.SequenceMatcher (Lib/difflib.py) without junk and with autojunk=False, so that every element of `b` is in
 * b2j and no element is popular.
 */
class SequenceMatcher {
public:
    SequenceMatcher(const Text & a, const Text & b) : a_(a), b_(b) {
        for (size_t j = 0; j < b.size(); j++) b2j_[b[j]].push_back(j);
    }

    /** find_longest_match(alo, ahi, blo, bhi). */
    Match longest_match(size_t alo, size_t ahi, size_t blo, size_t bhi) const {
        size_t besti = alo, bestj = blo, bestsize = 0;
        // The dict j2len of the row before, with j2len[j + 1] the length of the run that ends with a[i - 1] and b[j], and
        // that of this row; an index absent from the dict is 0, and each row clears the indices it wrote when it is done.
        std::vector<size_t> j2len(b_.size() + 1, 0), newj2len(b_.size() + 1, 0);
        std::vector<size_t> written, newwritten;
        for (size_t i = alo; i < ahi; i++) {
            const auto indices = b2j_.find(a_[i]);
            if (indices != b2j_.end()) {
                for (size_t j : indices->second) {
                    if (j < blo) continue;
                    if (j >= bhi) break;
                    const size_t k = newj2len[j + 1] = j2len[j] + 1;
                    newwritten.push_back(j + 1);
                    if (k > bestsize) {
                        besti = i - k + 1;
                        bestj = j - k + 1;
                        bestsize = k;
                    }
                }
            }
            for (size_t j : written) j2len[j] = 0;
            std::swap(j2len, newj2len);
            std::swap(written, newwritten);
            newwritten.clear();
        }
        // The extension by equal elements on each side; those by junk, which follow in difflib, never extend without it.
        while (besti > alo && bestj > blo && a_[besti - 1] == b_[bestj - 1]) {
            besti--;
            bestj--;
            bestsize++;
        }
        while (besti + bestsize < ahi && bestj + bestsize < bhi && a_[besti + bestsize] == b_[bestj + bestsize]) bestsize++;
        return {besti, bestj, bestsize};
    }

    /** get_matching_blocks(), ended by (len(a), len(b), 0). */
    std::vector<Match> matching_blocks() const {
        const size_t la = a_.size(), lb = b_.size();
        std::vector<std::array<size_t, 4>> queue{{0, la, 0, lb}};
        std::vector<Match> blocks;
        while (!queue.empty()) {
            const auto [alo, ahi, blo, bhi] = queue.back();
            queue.pop_back();
            const Match x = longest_match(alo, ahi, blo, bhi);
            if (x.size == 0) continue;
            blocks.push_back(x);
            if (alo < x.a && blo < x.b) queue.push_back({alo, x.a, blo, x.b});
            if (x.a + x.size < ahi && x.b + x.size < bhi) queue.push_back({x.a + x.size, ahi, x.b + x.size, bhi});
        }
        std::sort(blocks.begin(), blocks.end(), [](const Match & x, const Match & y) {
            return std::tie(x.a, x.b, x.size) < std::tie(y.a, y.b, y.size);
        });
        size_t i1 = 0, j1 = 0, k1 = 0;
        std::vector<Match> non_adjacent;
        for (const Match & m : blocks) {
            if (i1 + k1 == m.a && j1 + k1 == m.b) {
                k1 += m.size;
                continue;
            }
            if (k1) non_adjacent.push_back({i1, j1, k1});
            i1 = m.a;
            j1 = m.b;
            k1 = m.size;
        }
        if (k1) non_adjacent.push_back({i1, j1, k1});
        non_adjacent.push_back({la, lb, 0});
        return non_adjacent;
    }

private:
    const Text & a_;
    const Text & b_;
    std::unordered_map<uint32_t, std::vector<size_t>> b2j_;
};

Text code_points(const std::string & s) {
    std::optional<Text> text = decode_utf8(s);
    if (!text) throw std::invalid_argument("a transcript is not valid UTF-8");
    return *text;
}

/** The sum of `n` floats from `a` as pairwise_sum() of NumPy's loops_utils.h.src adds them. */
float pairwise_sum(const float * a, size_t n) {
    if (n < 8) {
        float res = 0.0f;
        for (size_t i = 0; i < n; i++) res += a[i];
        return res;
    }
    if (n <= 128) {
        // Eight accumulators, each over every eighth value, then joined in pairs and the rest added.
        float r[8];
        for (size_t k = 0; k < 8; k++) r[k] = a[k];
        size_t i = 8;
        for (; i < n - n % 8; i += 8) {
            for (size_t k = 0; k < 8; k++) r[k] += a[i + k];
        }
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

}  // namespace

std::vector<Opcode> opcodes(const Text & a, const Text & b) {
    const SequenceMatcher matcher(a, b);
    std::vector<Opcode> answer;
    size_t i = 0, j = 0;
    for (const Match & m : matcher.matching_blocks()) {
        if (i < m.a && j < m.b) {
            answer.push_back({Opcode::Tag::Replace, i, m.a, j, m.b});
        } else if (i < m.a) {
            answer.push_back({Opcode::Tag::Delete, i, m.a, j, m.b});
        } else if (j < m.b) {
            answer.push_back({Opcode::Tag::Insert, i, m.a, j, m.b});
        }
        i = m.a + m.size;
        j = m.b + m.size;
        if (m.size) answer.push_back({Opcode::Tag::Equal, m.a, i, m.b, j});
    }
    return answer;
}

std::string guarded(const std::string & plain, const std::string & boosted, const Vocabulary & vocabulary) {
    const Text p = code_points(plain), b = code_points(boosted);
    const std::vector<FoundWord> found = vocabulary.find(b);
    Text out;
    for (const Opcode & op : opcodes(p, b)) {
        const bool changed = op.tag == Opcode::Tag::Replace || op.tag == Opcode::Tag::Insert;
        const bool listed =
            std::any_of(found.begin(), found.end(), [&op](const FoundWord & w) { return w.start < op.j2 && w.end > op.j1; });
        if (op.tag == Opcode::Tag::Equal || (changed && listed)) {
            out.insert(out.end(), b.begin() + (std::ptrdiff_t) op.j1, b.begin() + (std::ptrdiff_t) op.j2);
            continue;
        }
        const bool absorbed = std::any_of(found.begin(), found.end(),
                                          [&op](const FoundWord & w) { return op.j1 == w.start || op.j1 == w.end; });
        if (op.tag == Opcode::Tag::Delete && op.i2 - op.i1 <= 1 && absorbed) continue;
        out.insert(out.end(), p.begin() + (std::ptrdiff_t) op.i1, p.begin() + (std::ptrdiff_t) op.i2);
    }
    return encode_utf8(out);
}

bool has_speech(const std::vector<float> & samples, float rms, size_t frame) {
    if (frame == 0) throw std::invalid_argument("has_speech() takes frames of at least one sample");
    if (samples.empty()) return false;
    const size_t frames = (samples.size() + frame - 1) / frame;
    std::vector<float> squares(frames * frame, 0.0f);
    for (size_t i = 0; i < samples.size(); i++) squares[i] = samples[i] * samples[i];
    for (size_t f = 0; f < frames; f++) {
        // NumPy divides the float32 sum by the count in float64 and rounds the quotient to float32, which is the float32
        // quotient: rounding twice is harmless for a quotient of two float32 values in a format of 53 >= 2 * 24 + 2 bits.
        const float mean = pairwise_sum(squares.data() + f * frame, frame) / (float) frame;
        if (std::sqrt(mean) >= rms) return true;
    }
    return false;
}

}  // namespace fastconformer
