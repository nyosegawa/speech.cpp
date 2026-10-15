#pragma once

#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "audio-decoder.h"
#include "codec.h"
#include "model-file.h"
#include "prompt.h"
#include "sampler.h"
#include "text-model.h"

namespace drama3 {

/** What a synthesis is asked to speak, and how. */
struct Request {
    std::string text;
    Sampling sampling;
    uint64_t seed = 0;
    /** The frames each chunk of the text may take at most, within the file's generation.max_frames. */
    int64_t max_frames = 0;
};

/** The references of a voice's speakers, which the prompt lists; none for the built-in voice none. */
struct Voice {
    std::vector<Utterance> references;
};

/** How a synthesis ended: every chunk spoke to its end, a chunk reached max_frames, or the sink stopped it. */
enum class Ending { Complete, MaxFrames, Stopped };

/** Called with each piece of audio at sample_rate() as it is decoded; returning false stops the synthesis. */
using AudioSink = std::function<bool(const float * samples, size_t n)>;

/**
 * Text in, audio out: the prompt, the text model, the audio decoder, the sampler and the codec of one model file. A text
 * is spoken in the chunks split_text() cuts it into, one after another, each prompted with the voice's references and
 * with the first and the latest chunks before it, which the prompt leaves out, the latest first, while it would take
 * more rows than the model does. Each chunk's frames are drawn until the end of speech or max_frames, and its audio
 * is decoded and passed on in windows of 4, 8, 16, 32, 64 and then 128 frames as the frames are drawn.
 */
class Synthesizer {
public:
    /** The synthesizer of the loaded model file `m`, which outlives it. */
    Synthesizer(const ModelFile & m, ggml_backend_t backend);

    int sample_rate() const { return codec_.sample_rate(); }
    int samples_per_frame() const { return codec_.hop(); }
    /** The tokens of `text` as the prompt counts them. */
    size_t count_tokens(const std::string & text) const { return prompt_.tokenize(text).size(); }

    /**
     * Speaks `r.text` in `voice`, passing the audio to `sink`. Every chunk's prompt is written before any is spoken, so
     * that a text or a voice the prompt refuses throws its Error before any audio.
     */
    Ending synthesize(const Request & r, const Voice & voice, const AudioSink & sink);

private:
    struct Chunk;

    /**
     * Draws the frames that follow the prompt `rows` until the end of speech or `r.max_frames`, passing their audio to
     * `sink` as it goes, and returns their codes and how the chunk ended.
     */
    Chunk speak(const std::vector<int32_t> & rows, const Request & r, const FrameSampler & sampler, std::mt19937_64 & rng,
                const AudioSink & sink);

    const ModelFile & m_;
    Prompt prompt_;
    TextModel text_;
    AudioDecoder decoder_;
    Codec codec_;
    size_t chunk_bytes_ = 0, chunk_speakers_ = 0, recent_turns_ = 0;
    int32_t eos_ = 0;
};

}  // namespace drama3
