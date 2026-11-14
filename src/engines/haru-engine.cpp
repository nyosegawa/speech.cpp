#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "engine.h"
#include "error.h"
#include "haru/dialogue.h"
#include "haru/layout.h"
#include "haru/voice-file.h"
#include "json.h"
#include "language.h"

namespace {

/** The options that fill a field of the character card, in the order of the file's haru.card.fields. */
const std::pair<speech_option, const char *> kCardFields[] = {
    {SPEECH_OPT_PERSONA, "persona"},         {SPEECH_OPT_NAME, "name"},
    {SPEECH_OPT_GENDER, "gender"},           {SPEECH_OPT_BACKCHANNEL, "backchannel"},
    {SPEECH_OPT_OTHER_NAME, "other_name"},   {SPEECH_OPT_RELATIONSHIP, "relationship"},
    {SPEECH_OPT_SITUATION, "situation"},
};

/**
 * The shortest decimal that reads back as `v` in float32, so that a default the model file holds in float32 shows as
 * the official code writes it (0.7, not 0.699999988079071); the request rounds it back to `v`.
 */
double shortest(float v) {
    char text[32];
    for (int digits = 1;; digits++) {
        std::snprintf(text, sizeof text, "%.*g", digits, (double) v);
        if ((float) std::strtod(text, nullptr) == v) return std::strtod(text, nullptr);
    }
}

/** The seconds of `frames` frames of the model. */
double seconds_of(const ModelFile & m, int64_t frames) {
    return (double) frames * m.size("haru.frame_samples") / m.size("speech.sample_rate");
}

/** Passes each frame of a conversation to the sink: the first when it starts, then one for each whole frame it hears. */
class Stream : public DialogueStream {
public:
    Stream(std::unique_ptr<haru::Dialogue> dialogue, uint64_t seed, DialogueSink sink)
        : dialogue_(std::move(dialogue)), random_(seed), sink_(std::move(sink)) {}

    /** Runs the prompt and passes the first frame; the C API learns from its sink whether the caller stopped there. */
    void start(const std::vector<int32_t> & card, const haru::Voice * voice, const haru::DialogueSettings & s) {
        const haru::DialogueFrame f = dialogue_->start(card, voice ? voice->means : std::vector<float>(), voice ? voice->quietest : -1, s, random_);
        sink_(f.audio.data(), f.audio.size(), f.text);
    }

    bool push(const float * samples, size_t n) override {
        heard_.insert(heard_.end(), samples, samples + n);
        const size_t frame = (size_t) dialogue_->frame_samples();
        size_t used = 0;
        bool more = true;
        while (more && heard_.size() - used >= frame) {
            if (dialogue_->done()) {
                throw Error(Fault::OutOfRange, "the conversation has given the frames of its max_seconds; start a new one with a longer max_seconds to go on",
                            "max_seconds");
            }
            const haru::DialogueFrame f = dialogue_->step(std::vector<float>(heard_.begin() + (std::ptrdiff_t) used, heard_.begin() + (std::ptrdiff_t) (used + frame)), random_);
            used += frame;
            more = sink_(f.audio.data(), f.audio.size(), f.text);
        }
        heard_.erase(heard_.begin(), heard_.begin() + (std::ptrdiff_t) used);
        return more;
    }

private:
    std::unique_ptr<haru::Dialogue> dialogue_;
    haru::SeededRandomness random_;
    DialogueSink sink_;
    /** The audio heard short of a whole frame. */
    std::vector<float> heard_;
};

class HaruEngine : public Engine {
public:
    HaruEngine(const std::string & path, ggml_backend_t backend)
        : model_(std::make_unique<ModelFile>(path, backend, haru::model_layout)), backend_(backend), codec_(*model_, backend) {}

    std::unique_ptr<DialogueStream> start_dialogue(const RequestValues & values, DialogueSink sink) override {
        const ModelFile & m = *model_;
        haru::DialogueSettings s = haru::file_dialogue_settings(m);
        s.sampling.temperature = (float) values.number(SPEECH_OPT_TEMPERATURE);
        s.sampling.top_k = (int) values.integer(SPEECH_OPT_TOP_K);
        s.sampling.top_p = (float) values.number(SPEECH_OPT_TOP_P);
        s.guidance = (float) values.number(SPEECH_OPT_GUIDANCE_SCALE);
        s.eop_bias = (float) values.number(SPEECH_OPT_EOP_BIAS);
        s.speak_first = values.boolean(SPEECH_OPT_SPEAK_FIRST);
        const double seconds = values.number(SPEECH_OPT_MAX_SECONDS);
        s.max_frames = (int64_t) std::floor(seconds * m.size("speech.sample_rate") / m.size("haru.frame_samples") + 1e-9);
        if (s.max_frames < 1) {
            throw Error(Fault::OutOfRange, "max_seconds is " + json_number(seconds) + ", shorter than one frame of " + json_number(seconds_of(m, 1)) + " s",
                        "max_seconds");
        }
        std::map<std::string, std::string> fields;
        for (const auto & [option, field] : kCardFields) {
            if (values.has(option) && !values.string(option).empty()) fields[field] = values.string(option);
        }
        auto dialogue = std::make_unique<haru::Dialogue>(m, backend_);
        std::vector<int32_t> card;
        if (!fields.empty()) {
            const std::string & tag = values.string(SPEECH_OPT_LANGUAGE);
            if (language_is_auto(tag)) {
                throw Error(Fault::InvalidArgument, "the character card is written in a language, and the request's is auto; give the language of the conversation",
                            "language");
            }
            std::string code;
            for (const std::string & l : m.str_array("haru.card.languages")) {
                if (bcp47_matches(tag, l)) code = l;
            }
            card = haru::card_ids(m, dialogue->tokenizer(), dialogue->added(), haru::render_card(m, code, fields));
        }
        const haru::Voice * voice = values.has(SPEECH_OPT_VOICE) ? &voices_.at(values.string(SPEECH_OPT_VOICE)) : nullptr;
        auto stream = std::make_unique<Stream>(std::move(dialogue), (uint64_t) values.integer(SPEECH_OPT_SEED), std::move(sink));
        try {
            stream->start(card, voice, s);
        } catch (const Error & e) {
            if (e.fault() != Fault::OutOfMemory) throw;
            // The keys and values of every frame of max_seconds are allocated at the start, 64 KB a frame for Haru-4B:
            // an hour does not fit beside Q8_0's weights on a GPU of 8 GB.
            throw Error(Fault::OutOfMemory,
                        std::string(e.what()) + "; a conversation holds the keys and values of its max_seconds, " + json_number(seconds) +
                            " s, from its start: give a shorter max_seconds, or load a smaller type of the model",
                        "max_seconds");
        }
        return stream;
    }

    std::string add_voice(const std::string & name, const std::string & path) override {
        voices_.emplace(name, haru::load_voice(*model_, codec_, backend_, path));
        return "";
    }

    void warm_up() override {
        // The first conversation compiles the GPU kernels: its prompt and a frame of silence run through every graph.
        haru::Dialogue dialogue(*model_, backend_);
        haru::DialogueSettings s = haru::file_dialogue_settings(*model_);
        s.max_frames = 2;
        haru::SeededRandomness random(0);
        dialogue.start({}, {}, -1, s, random);
        dialogue.step(std::vector<float>((size_t) dialogue.frame_samples(), 0.0f), random);
    }

private:
    std::unique_ptr<ModelFile> model_;
    ggml_backend_t backend_;
    /** The codec the voices are encoded with, apart from any conversation's. */
    haru::Codec codec_;
    std::map<std::string, haru::Voice> voices_;
};

}  // namespace

/**
 * The table of the options Haru takes: the voice, which the model speaks with and none of which the file has of its own;
 * the language, in whose wording the character card is written; the seed of the draws; the conversation's length, by
 * default and at most the official session's; the official session's sampling of the text, a temperature of 0 drawing
 * the likeliest token and a top_k of 0 every token, its guidance, its bias of the token that starts the model's speech
 * and whether the model speaks first, by default the official ones, which the file holds; and the character card's
 * fields, which the official server writes into the card, its gender and backchannels chosen from the file's values.
 */
FamilyInfo describe_haru(const std::shared_ptr<const ModelFile> & file) {
    const ModelFile & m = *file;
    FamilyInfo info;
    info.voice_codec = m.str("haru.codec.sha256");
    info.voice_recordings = true;
    info.frame_samples = m.size("haru.frame_samples");
    const haru::DialogueSettings d = haru::file_dialogue_settings(m);
    const double longest = seconds_of(m, m.size("haru.dialogue.max_frames"));
    info.options = {
        {SPEECH_OPT_VOICE, false, true},
        {SPEECH_OPT_LANGUAGE, false, true, std::string("auto")},
        {SPEECH_OPT_SEED, false, true, std::nullopt, 0, (double) kMaxSeed},
        {SPEECH_OPT_MAX_SECONDS, false, true, longest, 0, longest, true},
        {SPEECH_OPT_TOP_K, false, true, (int64_t) d.sampling.top_k, 0, (double) m.size("haru.text.vocab_size")},
        {SPEECH_OPT_TOP_P, false, true, shortest(d.sampling.top_p), 0, 1, true},
        {SPEECH_OPT_TEMPERATURE, false, true, shortest(d.sampling.temperature), 0, INFINITY},
    };
    for (const auto & [option, field] : kCardFields) {
        OptionSpec spec{option, false, true};
        const std::string values = std::string("haru.card.") + field + "_values";
        if (m.has(values)) spec.choices = m.str_array(values);
        info.options.push_back(spec);
    }
    info.options.push_back({SPEECH_OPT_GUIDANCE_SCALE, false, true, shortest(d.guidance), (double) m.f32("haru.head.guidance_min"),
                            (double) m.f32("haru.head.guidance_max")});
    // A bias of 10,000 forces or forbids the start of speech at every frame, and keeps the logit within a float.
    info.options.push_back({SPEECH_OPT_EOP_BIAS, false, true, shortest(d.eop_bias), -1e4, 1e4});
    info.options.push_back({SPEECH_OPT_SPEAK_FIRST, false, true, false});
    return info;
}

std::unique_ptr<Engine> load_haru(const std::string & path, ggml_backend_t backend, int) {
    return std::make_unique<HaruEngine>(path, backend);
}

void make_haru_voice(const std::string & model_path, const VoiceRecipe & recipe, const std::string & voice_path, ggml_backend_t backend) {
    if (!recipe.embedding.empty() || !recipe.normalize || recipe.lufs) {
        throw Error(Fault::InvalidArgument, "Haru takes its recording as it is, without an embedding or a loudness; leave them out",
                    !recipe.embedding.empty() ? "embedding" : "loudness");
    }
    if (recipe.parts.size() != 1 || recipe.parts[0].voice_file || recipe.parts[0].transcript || recipe.parts[0].speaker) {
        throw Error(Fault::InvalidArgument, "Haru makes a voice of one WAVE file of one speaker, without transcripts or speakers; give it one recording",
                    "references");
    }
    haru::make_voice_file(model_path, recipe.parts[0].path, voice_path, backend);
}
