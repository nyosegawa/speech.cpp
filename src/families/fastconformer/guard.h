#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "vocabulary.h"

namespace fastconformer {

/** An operation of difflib.SequenceMatcher.get_opcodes(): what a[i1:i2] is to b[j1:j2]. */
struct Opcode {
    enum class Tag { Equal, Replace, Delete, Insert };
    Tag tag;
    size_t i1, i2, j1, j2;
};

/**
 * The operations that turn `a` into `b` as difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() of CPython
 * 3.11 gives them: the longest run the two have in common, of equally long ones the one that ends first in `a` and then
 * in `b`, then the same on each side of it, the runs joined where they meet.
 */
std::vector<Opcode> opcodes(const std::vector<uint32_t> & a, const std::vector<uint32_t> & b);

/**
 * `boosted`, decoded with the words of `vocabulary` boosted, where it agrees with `plain`, decoded without them, or where
 * it changes a run that overlaps a listed word it holds; elsewhere `plain`, but for one character of `plain` that
 * `boosted` drops where a listed word starts or ends, which stays dropped, since a word's own spelling may absorb one
 * (guarded() of Haas-ASR's boosting.py). So a word the boost wrote without the audio's support goes back to what was read
 * without it, and speech the boost skipped after a word comes back. Both texts are UTF-8.
 */
std::string guarded(const std::string & plain, const std::string & boosted, const Vocabulary & vocabulary);

/**
 * Whether any frame of `frame` samples of `samples`, the last padded with zeros, has a root mean square of at least
 * `rms`, as has_speech() of boosting.py computes it with NumPy in float32: the squares of a frame summed in the order
 * NumPy sums a row, divided by the frame's length, and the square root compared with `rms`.
 */
bool has_speech(const std::vector<float> & samples, float rms, size_t frame);

}  // namespace fastconformer
