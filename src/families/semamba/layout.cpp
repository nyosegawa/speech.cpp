#include "layout.h"

#include <cmath>
#include <string>
#include <vector>

#include "error.h"

namespace semamba {

namespace {

/** Every tensor in F32: the model's 9.6M parameters take 38 MB, and its files are F32 alone. */
const Storage kF32 = float32_storage();

void require(bool condition, const ModelFile & m, const std::string & what) {
    if (!condition) throw Error(Fault::File, m.path() + ": " + what + "; " + m.remedy());
}

/** A Conv2d of `in` channels to `out` with a kernel of `kw` bins by `kh` frames, its InstanceNorm2d and its PReLU. */
void add_conv(std::vector<TensorSpec> & t, const std::string & name, int64_t kw, int64_t kh, int64_t in, int64_t out, int64_t channels) {
    add_block(t, name, {{".conv.weight", {kw, kh, in, out}, kF32},
                        {".conv.bias", {out}, kF32},
                        {".norm.weight", {channels}, kF32},
                        {".norm.bias", {channels}, kF32},
                        {".prelu", {channels}, kF32}});
}

void add_dense(std::vector<TensorSpec> & t, const std::string & name, int depth, int64_t hidden) {
    for (int i = 0; i < depth; i++) add_conv(t, name + "." + std::to_string(i), 3, 3, hidden * (i + 1), hidden, hidden);
}

/** Reads every key of the layout, checks the ones that must agree, and names the tensors they call for. */
std::vector<TensorSpec> tensors(const ModelFile & m) {
    check_model_keys(m, "enhancement", nullptr);
    const std::string p = "semamba.";
    const int reference = m.size(p + "stft.sampling_rate"), n_fft = m.size(p + "stft.n_fft"), hop = m.size(p + "stft.hop_size");
    require(hop <= n_fft, m, "semamba.stft.hop_size is longer than semamba.stft.n_fft");
    const int lowest = m.size(p + "sample_rate_min"), highest = m.size(p + "sample_rate_max");
    require(lowest <= highest, m, "semamba.sample_rate_min is above semamba.sample_rate_max");
    // The FFT at the lowest rate must still span a few samples, as the scaling of the STFT rounds to even sizes.
    require((int64_t) n_fft * lowest / reference >= 4 && (int64_t) hop * lowest / reference >= 2, m,
            "the STFT has fewer than 4 samples, or its hop fewer than 2, at semamba.sample_rate_min");
    const float chunk = m.f32(p + "chunk_size_in_seconds"), portion = m.f32(p + "hop_length_portion");
    require(chunk > 0 && std::isfinite(chunk) && portion > 0 && portion <= 1, m,
            "semamba.chunk_size_in_seconds is not positive, or semamba.hop_length_portion is not above 0 and at most 1");
    const float floor = m.f32(p + "window_floor"), zero = m.f32(p + "zero_portion"), pad = m.f32(p + "pad_value");
    require(floor > 0 && floor < 1 && zero >= 0 && zero <= 1 && std::isfinite(pad), m,
            "semamba.window_floor is not between 0 and 1, semamba.zero_portion is not a share, or semamba.pad_value is not finite");
    // The padding of the input's bins and frames: more than an FFT would pad a chunk with more zeros than it holds.
    require(m.u32(p + "input_padding") <= (uint32_t) n_fft, m, "semamba.input_padding is longer than semamba.stft.n_fft");
    const int depth = m.count(p + "dense_depth"), frame_stride = m.size(p + "frame_stride"), bin_stride = m.size(p + "bin_stride");
    const int blocks = m.count(p + "block_count");
    for (const char * key : {"instance_norm_eps", "layer_norm_eps"}) {
        const float eps = m.f32(p + key);
        require(eps > 0 && std::isfinite(eps), m, p + key + " is not positive");
    }

    // The widths have no key: the hidden channels are the first convolution's outputs, and Mamba's inner width, state,
    // convolution and dt_rank are those of the first block's tensors, which every other is checked against.
    const int64_t hidden = m.width("enc.in.conv.weight", 3);
    const int64_t inner = m.width("blk.0.time.fw.in_proj.weight", 1) / 2, state = m.width("blk.0.time.fw.A_log", 0);
    const int64_t width = m.width("blk.0.time.fw.conv1d.weight", 0), rank = m.width("blk.0.time.fw.dt_proj.weight", 0);
    std::vector<TensorSpec> t;
    // The magnitude and the phase of each bin of each frame are the network's two input channels.
    add_conv(t, "enc.in", 1, 1, 2, hidden, hidden);
    add_dense(t, "enc.dense", depth, hidden);
    add_conv(t, "enc.out", 3, 1, hidden, hidden, hidden);
    for (int b = 0; b < blocks; b++) {
        for (const char * part : {"time", "freq"}) {
            const std::string name = "blk." + std::to_string(b) + "." + part;
            for (const char * direction : {".fw", ".bw"}) {
                add_block(t, name + direction, {{".in_proj.weight", {hidden, 2 * inner}, kF32},
                                                {".conv1d.weight", {width, inner}, kF32},
                                                {".conv1d.bias", {inner}, kF32},
                                                {".x_proj.weight", {inner, rank + 2 * state}, kF32},
                                                {".dt_proj.weight", {rank, inner}, kF32},
                                                {".dt_proj.bias", {inner}, kF32},
                                                {".A_log", {state, inner}, kF32},
                                                {".D", {inner}, kF32},
                                                {".out_proj.weight", {inner, hidden}, kF32}});
            }
            add_block(t, name, {{".proj.weight", {2 * hidden, hidden}, kF32},
                                {".proj.bias", {hidden}, kF32},
                                {".norm.weight", {hidden}, kF32},
                                {".norm.bias", {hidden}, kF32}});
        }
    }
    for (const char * decoder : {"mag", "pha"}) {
        const std::string name = decoder;
        add_dense(t, name + ".dense", depth, hidden);
        // The first up-sampling gives the bins back what the encoder's stride took, the second the frames.
        add_conv(t, name + ".up1", 3, 1, hidden, hidden * bin_stride, hidden);
        add_conv(t, name + ".up2", 3, 1, hidden, hidden * frame_stride, hidden);
    }
    add_block(t, "mag.out", {{".weight", {1, 1, hidden, 1}, kF32}, {".bias", {1}, kF32}});
    for (const char * part : {"pha.r", "pha.i"}) add_block(t, part, {{".weight", {1, 1, hidden, 1}, kF32}, {".bias", {1}, kF32}});
    return t;
}

}  // namespace

const Layout layout = {"semamba", 1, "convert it again with reference/semamba/convert.py, or download it again from its Hugging Face repository",
                       tensors};

}  // namespace semamba
