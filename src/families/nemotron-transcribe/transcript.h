#pragma once

#include <cstddef>
#include <string>

#include "model-file.h"

namespace nemotron_transcribe {

/**
 * The model card's generation-loop guardrail, which its reported transcriptions apply to the decoded text once: every
 * run of generation.char_run or more of one character kept once (re.sub(r"(.)\1{20,}", r"\1", text, flags=re.DOTALL)
 * with the card's 21), then, scanning from the start, every block of 1 to generation.block_max_chars characters repeated
 * generation.block_run times or more in a row kept once, the shortest block that repeats so at a place taken first, as
 * the card's lazy BLOCK_RUN (.{1,4096}?)\1{19,} takes it. It counts characters as Python does, in code points.
 */
class Guardrail {
public:
    explicit Guardrail(const ModelFile & m);

    std::string apply(const std::string & text) const;

    /**
     * Whether `text` ends in a run that apply() keeps once: one character written generation.char_run times or more, or
     * a block of 1 to generation.block_max_chars characters written generation.block_run times or more in a row, up to
     * its end, as either of the card's expressions anchored at the end matches it.
     */
    bool loops(const std::string & text) const;

private:
    size_t char_run_, block_run_, block_max_;
};

}  // namespace nemotron_transcribe
