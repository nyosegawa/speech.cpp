#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "ctc-text.h"
#include "model-file.h"

namespace nemotron_transcribe {

/** What the alignment searches: the CTC head's and the speaker branch's outputs over one recording. */
struct AlignmentInputs {
    /** The log probabilities of AlignmentText::columns at each CTC frame, [frames, columns] row-major. */
    std::vector<float> log_probs;
    int64_t frames = 0;
    /** The speaker branch's probabilities on the encoder's frames, [speaker_frames, speakers] row-major. */
    std::vector<float> speaker_probs;
    int64_t speaker_frames = 0;
    int speakers = 0;
    /** The speaker branch's 10 ms probabilities over activity_threshold, [label_speakers, label_frames] row-major. */
    std::vector<uint8_t> labels;
    int64_t label_frames = 0;
    int label_speakers = 0;
    /** The audio's length in seconds. */
    double duration = 0;
};

/** One speaker stream's search, for the checks: the frames searched, the first state of the band and the path's state at each, and its score. */
struct StreamSearch {
    std::vector<int64_t> positions, starts, path;
    float score = 0;
};

/** A start and an end in seconds. */
struct Span {
    double start, end;
};

/** What the alignment found, for the result and the checks. */
struct Alignment {
    /** The speaker probabilities on the CTC frames, [frames, speakers] row-major. */
    std::vector<float> speaker;
    /** The speaker branch's column of each stream, -1 for none. */
    std::vector<int> columns;
    std::vector<StreamSearch> searches;
    /** Each unit's span from its path, and after the refinement, in seconds. */
    std::vector<Span> rows, refined;
    /** Each unit's refined span rounded to the millisecond, as the vLLM plugin's public result gives it. */
    std::vector<Span> times;
};

/**
 * NeMo Speech's alignment of a transcript's speaker streams with the CTC head's output (_align_banded_ctc() of
 * ctc_timestamp_utils.py, which MultiSpeakerSOTWordTimestampAligner and NVIDIA's vLLM plugin run with the settings of
 * nemotron-transcribe.align.*). The speaker probabilities, clamped to [0, 1], are linearly resampled to the CTC frames as
 * torch.linspace() places them. _one_pass_speaker_mapping() matches each stream to a speaker column: the greedy CTC
 * tokens' onsets counted under each column where its probability reaches speaker_activity_threshold, compared with each
 * stream's tokens by an IDF-weighted cosine of square-root counts less a tenth of the log ratio of their totals, and
 * assigned by scipy's linear_sum_assignment(). Each stream is searched over the frames where its column's probability
 * reaches 0.01, with speaker_logprob_weight times the log of that probability, floored at epsilon, added to its tokens'
 * log probabilities: _banded_ctc_search()'s max-sum dynamic program in float32, over every state below 1000 frames and
 * otherwise in a band of about a tenth of the states plus 100 on either side of a line through unique token trigrams,
 * ties preferring to stay, then to step, then to skip. A word spans the frames its path holds its tokens, at most
 * maximum_token_len seconds; _refine_word_timestamps() then moves the first word of each 10 ms speaker segment of its
 * speaker to the segment's onset and ends each word at the next word's start or its segment's end, within max_word_frames
 * CTC frames of the shorter of a CTC frame and an encoder frame. NeMo drops the words of a stream without a path, and
 * this alignment fails instead, as it does where a stream has more tokens than the frames or a path skips a word.
 */
class Aligner {
public:
    explicit Aligner(const ModelFile & m);

    /**
     * The alignment of `text` with `inputs`; one that cannot be found throws AlignmentFailure. `proceed`, asked before
     * each stream's search, stops the work where it says false, and the alignment is then none.
     */
    std::optional<Alignment> align(const AlignmentText & text, const AlignmentInputs & inputs, const std::function<bool()> & proceed = {}) const;

private:
    float speaker_threshold_, speaker_weight_, epsilon_;
    double maximum_token_len_, label_seconds_, encoder_seconds_;
    int max_word_frames_;
};

/** `x` rounded to the millisecond as Python's round(x, 3) rounds it: the nearest double to the nearest thousandth, ties to even. */
double round_ms(double x);

}  // namespace nemotron_transcribe
