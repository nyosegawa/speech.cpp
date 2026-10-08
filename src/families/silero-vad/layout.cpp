#include "layout.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "error.h"

namespace silero_vad {

namespace {

/** Every tensor in F32 in a file of every weight type: the model's 309K parameters take 1.2 MB, and its files are F32 alone. */
const Storage kF32 = float32_storage();

void require(bool condition, const ModelFile & m, const std::string & what) {
    if (!condition) throw Error(Fault::File, m.path() + ": " + what + "; " + m.remedy());
}

/** Reads every key of the layout, checks the ones that must agree, and names the tensors they call for. */
std::vector<TensorSpec> tensors(const ModelFile & m) {
    check_model_keys(m, "detection", nullptr);
    const std::string p = "silero-vad.";
    const int chunk = m.size(p + "chunk_size"), context = m.size(p + "context_size");
    const int n_fft = m.size(p + "stft.n_fft"), hop = m.size(p + "stft.hop_length"), reflect = m.size(p + "stft.reflect");
    // torch.nn.functional.pad() reflects by less than the input's length alone.
    require(reflect < context + chunk, m, "silero-vad.stft.reflect is not shorter than a chunk with its context");
    require(n_fft % 2 == 0 && n_fft <= context + chunk + reflect, m, "silero-vad.stft.n_fft is odd or longer than a chunk with its context and reflection");
    const std::vector<int32_t> strides = m.i32_array(p + "encoder.strides"), padding = m.i32_array(p + "encoder.padding");
    require(!strides.empty() && strides.size() == padding.size() && (int64_t) strides.size() <= m.tensor_count(), m,
            "silero-vad.encoder.strides and silero-vad.encoder.padding do not give each block of the encoder one value");
    require(std::all_of(strides.begin(), strides.end(), [](int32_t s) { return s >= 1; }) &&
                std::all_of(padding.begin(), padding.end(), [](int32_t s) { return s >= 0; }),
            m, "silero-vad.encoder.strides holds a stride below 1 or silero-vad.encoder.padding a negative padding");
    const double threshold = m.f64(p + "threshold"), offset = m.f64(p + "neg_threshold_offset"), floor = m.f64(p + "neg_threshold_floor");
    require(threshold >= 0 && threshold <= 1 && std::isfinite(offset) && floor >= 0 && floor <= 1, m,
            "silero-vad.threshold or silero-vad.neg_threshold_floor is not a probability, or silero-vad.neg_threshold_offset is not finite");
    for (const char * key : {"min_speech_duration_ms", "min_silence_duration_ms", "speech_pad_ms", "min_silence_at_max_speech_ms"}) m.u32(p + key);

    // The widths of the encoder's channels and of the LSTM cell have no key: each is the width of one tensor, which the
    // others are checked against.
    std::vector<TensorSpec> t = {{"stft.basis", {n_fft, n_fft + 2}, kF32}};
    int64_t channels = n_fft / 2 + 1, length = (context + chunk + reflect - n_fft) / hop + 1;
    for (size_t i = 0; i < strides.size(); i++) {
        const std::string name = "encoder." + std::to_string(i);
        const int64_t width = m.width(name + ".weight", 0), out = m.width(name + ".weight", 2);
        t.push_back({name + ".weight", {width, channels, out}, kF32});
        t.push_back({name + ".bias", {out}, kF32});
        require(length + 2 * padding[i] >= width, m, "the convolution of encoder block " + std::to_string(i) + " is wider than its padded input");
        length = (length + 2 * padding[i] - width) / strides[i] + 1;
        channels = out;
    }
    // The LSTM cell takes one vector of a chunk, which the official decoder squeezes out of the encoder's last axis.
    require(length == 1, m, "the encoder leaves " + std::to_string(length) + " frames of a chunk, where the LSTM cell takes one");
    const int64_t hidden = m.width("lstm.hh.weight", 0);
    t.push_back({"lstm.ih.weight", {channels, 4 * hidden}, kF32});
    t.push_back({"lstm.hh.weight", {hidden, 4 * hidden}, kF32});
    t.push_back({"lstm.bias", {4 * hidden}, kF32});
    t.push_back({"decoder.weight", {hidden, 1}, kF32});
    t.push_back({"decoder.bias", {1}, kF32});
    return t;
}

}  // namespace

const Layout layout = {"silero-vad", 1,
                       "convert it again with reference/silero-vad/convert.py, or download it again from its Hugging Face repository", tensors};

}  // namespace silero_vad
