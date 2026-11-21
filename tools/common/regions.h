#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cancellation.h"
#include "library.h"
#include "request-options.h"
#include "transcript.h"

// Transcription by regions: a detection model finds where someone speaks in the audio, a recognition model recognizes
// each region alone, and the texts are joined. A FastConformer model given 15 to 20 s that hold several sentences drops
// whole sentences, as NeMo's own implementation does, and writes words for audio in which no one speaks; given one
// region at a time it hears one or a few sentences and nothing where no one speaks.

/** A member of OpenAI's server_vad, which chunking_strategy and the Realtime API's turn_detection share, and the option it sets. */
struct VadMember {
    const char * name;
    speech_option option;
};

/**
 * The members of server_vad that speech.cpp takes: threshold as threshold, prefix_padding_ms, the audio kept before
 * speech, as speech_pad_ms, and silence_duration_ms, the silence that ends speech, as min_silence_duration_ms. RegionCut
 * keeps that padding around the speech itself.
 */
constexpr VadMember kServerVad[] = {
    {"threshold", SPEECH_OPT_THRESHOLD},
    {"prefix_padding_ms", SPEECH_OPT_SPEECH_PAD_MS},
    {"silence_duration_ms", SPEECH_OPT_MIN_SILENCE_DURATION_MS},
};

/** The member of server_vad that sets the detection option named `option`, or "". */
std::string server_vad_member(const std::string & option);

/** A region where someone speaks, in seconds from the start of the audio. */
struct Region {
    double start = 0;
    double end = 0;
};

/** The samples [first, last) of a region of audio that has `n` samples at `rate`. */
std::pair<size_t, size_t> region_samples(const Region & region, int rate, size_t n);

/**
 * The detection options of transcription by regions: `given`, then the detection model's `defaults` from the command
 * line that started the program, then OpenAI's server_vad defaults (threshold 0.5, speech_pad_ms 300 for its
 * prefix_padding_ms, min_silence_duration_ms 500 for its silence_duration_ms) and a longest region,
 * max_speech_duration_s, which its form has no member for.
 */
std::vector<RequestOption> region_options(const std::vector<RequestOption> & given, const std::vector<RequestOption> & defaults = {});

/** A detection request on `detector` with the audio and the options set, which the library checks as they are set. */
Request detection_request(speech_model * detector, const std::vector<float> & samples, int rate, const std::vector<RequestOption> & options);

/** The regions that a detection request set up by detection_request() finds, or nothing once it is cancelled. */
std::optional<std::vector<Region>> detect_regions(speech_request * detection, Cancellation & cancellation);

/**
 * How transcription by regions cuts audio, as OpenAI's server_vad describes its turns: a turn ends once
 * silence_duration_ms of silence has followed its speech, and keeps prefix_padding_ms before the speech. The detection
 * finds the speech alone, with speech_pad_ms 0, and each region keeps speech_pad_ms before its speech and as much after
 * it, but no more than min_silence_duration_ms, so that the audio of a region is all there once the silence that ends it
 * has passed. Silero VAD's own padding gives a region half the silence before the next one where that starts within
 * twice speech_pad_ms, which a stream knows only once that much silence has passed: 600 ms with OpenAI's defaults, where
 * silence_duration_ms says 500.
 */
class RegionCut {
public:
    /**
     * The cut that `options`, as region_options() completes them, ask of `detector`, which checks them as a request
     * does; it throws a Failure where the padding leaves no room for speech within max_speech_duration_s.
     */
    RegionCut(speech_model * detector, const std::vector<RequestOption> & options);

    /** The detection's options: `options` with speech_pad_ms 0, and max_speech_duration_s less the padding. */
    const std::vector<RequestOption> & detection() const { return detection_; }
    /** The seconds kept before a region's speech and after it, and the silence that ends a region. */
    double before() const { return before_; }
    double after() const { return after_; }
    double silence() const { return silence_; }

    /**
     * The region of `speech`, in audio of `length` seconds: from before() earlier, but not before `from`, the end of the
     * region before it, to after() later, but not past `length`. Where the speech after it starts at `next` within the
     * silence that ends a region, as where max_speech_duration_s cut the speech, the region ends after() later or halfway
     * to `next`, whichever comes first.
     */
    Region pad(const Region & speech, double from, std::optional<double> next, double length) const;
    /** The regions of `speech`, the speech that a detection with detection() found in `length` seconds of audio. */
    std::vector<Region> regions(const std::vector<Region> & speech, double length) const;

private:
    std::vector<RequestOption> detection_;
    double before_ = 0, after_ = 0, silence_ = 0;
};

/**
 * Checks that `recognizer` is a model of speech recognition and takes `options`, before any region, so that audio without
 * regions refuses what audio with them would: the library's refusal throws a Failure, and another task unsupported.
 */
void check_recognition(speech_model * recognizer, const std::vector<RequestOption> & options);

/**
 * Recognizes each region of `samples` alone on `recognizer` with `options`, and joins the results with append_part(),
 * the times of the segments and tokens (with `timestamps`) moved to those of the whole audio, after check_recognition(),
 * so audio without regions gives "". `parts`, when given, receives each region's recognition as it is, in order. It
 * returns nothing once cancelled.
 */
std::optional<Transcript> transcribe_regions(speech_model * recognizer, const std::vector<float> & samples, int rate,
                                             const std::vector<Region> & regions, const std::vector<RequestOption> & options,
                                             bool timestamps, Cancellation & cancellation, std::vector<Transcript> * parts = nullptr);
