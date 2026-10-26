#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "cancellation.h"
#include "library.h"
#include "request-options.h"

// Diarization in the tools: who speaks where in a whole recording, as speech_diarize() finds it, or in audio as it
// arrives, and the speakers within an utterance, a region of transcription by regions or a span of a stream.

/** A segment of one speaker, in seconds from the start of the audio, and the model's number for the speaker. */
struct SpeakerSegment {
    double start = 0;
    double end = 0;
    int speaker = 0;
};

/** A diarization request on `diarizer` with the audio and the options set, which the library checks as they are set. */
Request diarization_request(speech_model * diarizer, const std::vector<float> & samples, int rate, const std::vector<RequestOption> & options);

/** The segments that a diarization request set up by diarization_request() finds, or nothing once it is cancelled. */
std::optional<std::vector<SpeakerSegment>> diarize_segments(speech_request * diarization, Cancellation & cancellation);

/**
 * The parts of `segments` within [start, end): each segment that overlaps it, cut to it, in order of their starts and,
 * at the same start, of their speakers.
 */
std::vector<SpeakerSegment> speakers_within(const std::vector<SpeakerSegment> & segments, double start, double end);

/** The speaker who speaks the longest in `within`, the lower number of two who speak as long, or -1 for none. */
int main_speaker(const std::vector<SpeakerSegment> & within);

/** A speaker as the command line's text writes it: its number, or "-" for none (-1). */
inline std::string speaker_column(int speaker) {
    return speaker < 0 ? "-" : std::to_string(speaker);
}

/**
 * A speaker as the tools write it in OpenAI's shapes, the Realtime events and diarized_json, whose speakers are capital
 * letters: A for 0, B for 1, and after Z, AA, AB and on.
 */
std::string speaker_label(int speaker);

/** Segments as speech.cpp's own JSON writes them, [{"speaker", "start", "end"}], the times in seconds to the millisecond. */
std::string speakers_json(const std::vector<SpeakerSegment> & segments);

/**
 * A diarization of audio given a piece at a time (speech_diarization) on a diarization model that `keep` keeps until
 * the diarization is freed, with the request options it was started with. One thread pushes the audio and ends it while
 * another reads the speakers of the audio settled so far.
 */
class Diarization {
public:
    /** Starts a diarization of audio at `rate` with `options`, which the library checks; its refusal throws a Failure. */
    Diarization(speech_model * model, std::shared_ptr<const void> keep, std::vector<RequestOption> options, int rate);

    const std::vector<RequestOption> & options() const { return options_; }
    /** What keeps the model, which tells the model a host holds now from the one the diarization runs on. */
    const void * model() const { return keep_.get(); }

    /** Takes the next `n` samples; a failure throws, after which the diarization takes none and settles everything. */
    void push(const float * samples, size_t n);
    void end();
    /**
     * Whether the speakers of the audio up to `seconds` from the start of the diarization's audio are final: settled,
     * or the audio has ended, or the diarization failed, which leaves the speakers it gave.
     */
    bool covers(double seconds) const;
    /** The speakers of [start, end) of the diarization's audio, the segments given and those under way, as speakers_within(). */
    std::vector<SpeakerSegment> within(double start, double end) const;
    /** One more than the highest number of a speaker in the segments given and under way, or 0 for none. */
    int numbers_used() const;

private:
    void collect();
    /** The segments given and those under way up to the settled time. */
    std::vector<SpeakerSegment> segments() const;

    std::shared_ptr<const void> keep_;
    std::vector<RequestOption> options_;
    int speakers_ = 0;
    mutable std::mutex mutex_;
    std::unique_ptr<speech_diarization, decltype(&speech_diarization_free)> diarization_{nullptr, speech_diarization_free};
    std::vector<SpeakerSegment> given_;
    bool failed_ = false;
};
