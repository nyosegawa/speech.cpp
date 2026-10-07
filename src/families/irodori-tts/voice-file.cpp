#include "voice-file.h"

#include <cstring>
#include <memory>

#include "codec.h"
#include "error.h"
#include "gguf.h"
#include "layout.h"
#include "reference.h"

namespace irodori {

std::string device_kind(ggml_backend_t backend) {
    switch (ggml_backend_dev_type(ggml_backend_get_device(backend))) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU: return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return "igpu";
        default: throw std::logic_error(std::string("the device ") + ggml_backend_name(backend) + " is of a kind a voice file cannot name");
    }
}

void write_voice_file(const std::string & path, const std::vector<float> & latent, int latent_dim, const std::string & codec_sha256,
                      const VoiceOrigin & origin) {
    const size_t bytes = latent.size() * sizeof(float);
    ggml_init_params params = {ggml_tensor_overhead() + bytes + 64, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    if (!ctx) throw Error(Fault::OutOfMemory, "cannot hold the latent of a voice file in memory");
    ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, latent_dim, (int64_t) (latent.size() / latent_dim));
    ggml_set_name(t, "latent");
    std::memcpy(t->data, latent.data(), bytes);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> g(gguf_init_empty(), gguf_free);
    gguf_set_val_str(g.get(), "general.architecture", kVoiceArchitecture);
    gguf_set_val_u32(g.get(), "speech.layout", kVoiceLayout);
    gguf_set_val_str(g.get(), "speech.requires", kVoiceLayoutRequires);
    gguf_set_val_str(g.get(), "irodori-tts-voice.codec_sha256", codec_sha256.c_str());
    std::vector<float> seconds;
    std::vector<int32_t> rates;
    for (const Recording & r : origin.recordings) {
        seconds.push_back((float) r.seconds);
        rates.push_back(r.sample_rate);
    }
    gguf_set_arr_data(g.get(), "irodori-tts-voice.references.seconds", GGUF_TYPE_FLOAT32, seconds.data(), seconds.size());
    gguf_set_arr_data(g.get(), "irodori-tts-voice.references.sample_rates", GGUF_TYPE_INT32, rates.data(), rates.size());
    gguf_set_val_bool(g.get(), "irodori-tts-voice.normalized", origin.lufs.has_value());
    if (origin.lufs) gguf_set_val_f32(g.get(), "irodori-tts-voice.lufs", (float) *origin.lufs);
    gguf_set_val_str(g.get(), "irodori-tts-voice.device_kind", origin.device_kind.c_str());
    gguf_add_tensor(g.get(), t);
    if (!gguf_write_to_file(g.get(), path.c_str(), false)) {
        throw Error(Fault::Io, "cannot write " + path + "; check that its folder exists and is writable");
    }
}

void make_voice_file(const std::string & model_path, const std::vector<std::string> & reference_paths, const Loudness & loudness,
                     const std::string & voice_path, ggml_backend_t backend) {
    // Codec::build_encoder() reads these tensors and no others.
    const auto model = naming("model_path", [&] {
        return std::make_unique<const ModelFile>(model_path, backend, model_layout, [](const std::string & name) {
            return name.compare(0, 10, "codec.enc.") == 0 || name.compare(0, 17, "codec.bottleneck.") == 0;
        });
    });
    Codec codec(*model, backend);
    ReferenceRules rules(*model);
    if (!loudness.normalize) rules.lufs.reset();
    else if (loudness.lufs) rules.lufs = loudness.lufs;
    // One reference is the reference_path of speech_voice_make(), several the references of a voice's parameters.
    const char * input = reference_paths.size() == 1 ? "reference_path" : "references";
    std::vector<EncodedReference> references;
    VoiceOrigin origin{{}, rules.lufs, device_kind(backend)};
    for (const std::string & path : reference_paths) {
        references.push_back(naming(input, [&] { return encode_reference(codec, path, rules); }));
        origin.recordings.push_back({references.back().seconds, references.back().sample_rate});
    }
    const std::vector<float> latent = join_references(references, codec, rules, input);
    naming("voice_path", [&] { write_voice_file(voice_path, latent, codec.latent_dim(), codec.sha256(), origin); });
}

}  // namespace irodori
