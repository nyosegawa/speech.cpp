#include "synthesizer.h"

#include <algorithm>
#include <chrono>
#include <random>
#include <stdexcept>

#include "layout.h"

Synthesizer::Synthesizer(const std::string & path, ggml_backend_t backend)
    : model_(path, backend, qwen3_tts_layout),
      talker_(model_, backend),
      codec_(model_, backend),
      tokenizer_(model_),
      ids_(model_),
      generation_(model_) {}

namespace {

/** Adds the seconds from its construction to its destruction to `total`, when `total` is set. */
struct Timer {
    double * total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~Timer() {
        if (total) *total += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
};

}  // namespace

int Synthesizer::synthesize(const SynthesisRequest & r, const AudioSink & sink, int frames_per_piece,
                            SynthesisStats * stats) {
    SynthesisStats local;
    SynthesisStats & st = stats ? *stats : local;
    Prompt prompt;
    {
        Timer t{&st.prompt};
        std::vector<int32_t> text_ids = {ids_.im_start, ids_.assistant, ids_.newline};
        const std::vector<int32_t> body = tokenizer_.encode(r.text);
        if (body.empty()) throw std::runtime_error("the text is empty");
        if ((int) body.size() > max_text_tokens()) {
            throw std::runtime_error("the text is " + std::to_string(body.size()) + " tokens long and Qwen3-TTS takes at most " +
                                     std::to_string(max_text_tokens()) + "; split it into shorter texts");
        }
        text_ids.insert(text_ids.end(), body.begin(), body.end());
        text_ids.insert(text_ids.end(), {ids_.im_end, ids_.newline, ids_.im_start, ids_.assistant, ids_.newline});
        prompt = build_prompt(talker_, ids_, text_ids, r.speaker, r.language);
    }

    const int n_groups = talker_.num_code_groups();
    const int vocab = talker_.vocab();
    const std::vector<bool> none;
    std::mt19937_64 rng(r.seed);
    const int max_frames = std::min(r.max_frames, generation_.max_frames);

    {
        Timer t{&st.talker};
        // Every frame takes one position of the talker's cache after the prompt's.
        talker_.prefill(prompt.embeds, prompt.n, (int64_t) prompt.n + max_frames);
    }
    codec_.reset();
    std::vector<int32_t> history, frame(n_groups), pending;
    std::vector<float> audio;
    int frames = 0, pending_frames = 0;
    bool stopped = false;

    auto flush = [&]() {
        if (pending_frames == 0) return;
        audio.clear();
        {
            Timer t{&st.codec};
            codec_.decode(pending.data(), pending_frames, audio);
        }
        pending.clear();
        pending_frames = 0;
        if (!sink(audio.data(), audio.size())) stopped = true;
    };

    while (frames < max_frames && !stopped) {
        frame[0] = sample(talker_.logits(), generation_.talker, history, generation_.banned(vocab, ids_.codec_eos, frames), rng);
        if (frame[0] == ids_.codec_eos) break;
        history.push_back(frame[0]);
        {
            Timer t{&st.code_predictor};
            frame[1] = sample(talker_.cp_begin(frame[0]), generation_.code_predictor, {}, none, rng);
            for (int g = 1; g < n_groups - 1; g++) {
                frame[g + 1] = sample(talker_.cp_next(g, frame[g]), generation_.code_predictor, {}, none, rng);
            }
        }
        pending.insert(pending.end(), frame.begin(), frame.end());
        pending_frames++;
        frames++;
        if (frames == 1 || pending_frames >= frames_per_piece) flush();
        if (stopped) break;
        Timer t{&st.talker};
        talker_.step(frame.data(), prompt.frame_extra);
    }
    if (!stopped) flush();
    return frames;
}
