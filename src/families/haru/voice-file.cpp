#include "voice-file.h"

#include <cstring>
#include <memory>

#include "error.h"
#include "gguf.h"
#include "json.h"
#include "layout.h"
#include "prompt.h"
#include "resample.h"
#include "wav.h"

namespace haru {

namespace {

constexpr uint32_t kVoiceLayout = 1;
constexpr const char * kVoiceLayoutRequires = "0.11.0";
const Storage kF32 = float32_storage();

void require(bool condition, const ModelFile & m, const std::string & what) {
    if (!condition) throw Error(Fault::File, m.path() + ": " + what + "; " + m.remedy());
}

}  // namespace

Layout voice_layout(const ModelFile & model) {
    return {kVoiceArchitecture, kVoiceLayout, "make it again from its WAVE file with speech voice MODEL REFERENCE.wav VOICE.gguf",
            [&model](const ModelFile & m) {
                const std::string p = "haru-voice.";
                const std::string codec = m.str(p + "codec_sha256"), own = model.str("haru.codec.sha256");
                require(codec.size() == 64, m, "haru-voice.codec_sha256 is not a SHA-256 in hexadecimal");
                if (codec != own) {
                    throw Error(Fault::InvalidArgument, m.path() + " was made with the codec of SHA-256 " + codec + ", and " + model.str("general.name") +
                                                            " has the codec " + own + "; make the voice again from its WAVE file with this model");
                }
                const int64_t frames = m.width("latent", 1);
                const int32_t quietest = m.i32(p + "quietest_frame");
                require(quietest >= -1 && quietest < frames, m, "haru-voice.quietest_frame is not one of the voice's frames");
                require(m.f32(p + "reference.seconds") > 0 && m.u32(p + "reference.sample_rate") > 0, m, "the voice file gives its recording no length or no rate");
                return std::vector<TensorSpec>{{"latent", {model.size("haru.latent_dim"), frames}, kF32}};
            }};
}

Voice voice_of_recording(const ModelFile & model, Codec & codec, const std::vector<float> & mono, int sample_rate, const std::string & input) {
    const int rate = model.size("speech.sample_rate");
    const std::vector<float> audio = sample_rate == rate ? mono : Resampler(sample_rate, rate)(mono);
    const std::vector<float> reference = prepare_reference(model, audio);
    const double seconds = (double) reference.size() / rate, least = model.f32("haru.dialogue.reference_min_seconds");
    if (seconds < least) {
        throw Error(Fault::OutOfRange,
                    "the recording has " + json_number(seconds) + " s of sound, and " + model.str("general.name") + " takes a voice of " +
                        json_number(least) + " s at least; give a longer recording of one speaker",
                    input);
    }
    codec.reset_encoder();
    Voice v;
    v.means = codec.encode(reference);
    codec.reset_encoder();
    v.quietest = quietest_frame(reference, codec.frame_samples());
    return v;
}

Voice load_voice(const ModelFile & model, Codec & codec, ggml_backend_t backend, const std::string & path) {
    if (starts_with_riff(path)) {
        const Wav wav = read_wav(path);
        return voice_of_recording(model, codec, wav.mono(), wav.sample_rate, "voice_path");
    }
    const ModelFile file(path, backend, voice_layout(model));
    ggml_tensor * t = file.tensor("latent");
    Voice v;
    v.means.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.means.data(), 0, ggml_nbytes(t));
    v.quietest = file.i32("haru-voice.quietest_frame");
    return v;
}

void make_voice_file(const std::string & model_path, const std::string & wav_path, const std::string & voice_path, ggml_backend_t backend) {
    // The codec's encoder reads these tensors and no others.
    const ModelFile model(model_path, backend, model_layout, [](const std::string & name) { return name.rfind("codec.enc.", 0) == 0; });
    Codec codec(model, backend);
    const Wav wav = naming("references", [&] { return read_wav(wav_path); });
    const Voice v = voice_of_recording(model, codec, wav.mono(), wav.sample_rate, "references");
    const int64_t latent = model.size("haru.latent_dim");
    const size_t bytes = v.means.size() * sizeof(float);
    ggml_init_params params = {ggml_tensor_overhead() + bytes + 64, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    if (!ctx) throw Error(Fault::OutOfMemory, "cannot hold the latent of a voice file in memory");
    ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, latent, (int64_t) v.means.size() / latent);
    ggml_set_name(t, "latent");
    std::memcpy(t->data, v.means.data(), bytes);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> g(gguf_init_empty(), gguf_free);
    gguf_set_val_str(g.get(), "general.architecture", kVoiceArchitecture);
    gguf_set_val_u32(g.get(), "speech.layout", kVoiceLayout);
    gguf_set_val_str(g.get(), "speech.requires", kVoiceLayoutRequires);
    gguf_set_val_str(g.get(), "haru-voice.codec_sha256", model.str("haru.codec.sha256").c_str());
    gguf_set_val_i32(g.get(), "haru-voice.quietest_frame", (int32_t) v.quietest);
    gguf_set_val_f32(g.get(), "haru-voice.reference.seconds", (float) ((double) wav.mono().size() / wav.sample_rate));
    gguf_set_val_u32(g.get(), "haru-voice.reference.sample_rate", (uint32_t) wav.sample_rate);
    gguf_add_tensor(g.get(), t);
    naming("voice_path", [&] {
        if (!gguf_write_to_file(g.get(), voice_path.c_str(), false)) {
            throw Error(Fault::Io, "cannot write " + voice_path + "; check that its folder exists and is writable");
        }
    });
}

}  // namespace haru
