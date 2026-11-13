#include "codec.h"

#include <algorithm>
#include <string>

#include "error.h"

namespace haru {

/*
 * Activations are channel-first ([channels, time], ne0 = channels), so a convolution of width K runs as K matrix
 * products, one per tap, over views of its input shifted in time. Every convolution is causal and takes its left
 * context from its state, the last inputs of its previous call, which the graph overwrites with this call's last
 * inputs once it has read it. A convolution of width 2s and stride s cuts its input, its state before it, into frames
 * of s samples, which a reshape of the channel-first layout gives for free; output frame n is the first half of the
 * kernel on input frame n - 1 and the second half on frame n. A transposed convolution of width 2s and stride s gives
 * every tap of every input frame in one matrix product; output frame n is the first s taps of input frame n plus the
 * last s taps of frame n - 1, its state holding those of the previous call's last frame.
 */

namespace {

/** The frames of audio one graph of encode() takes at most: its widest activation is 128 channels of 3200 samples each. */
constexpr int64_t kEncodeFrames = 32;
/** The latents one graph of decode() takes at most. */
constexpr int64_t kDecodeFrames = 32;

}  // namespace

/** The state of each convolution of a stack, in the order the stack runs them, in a buffer of the codec's backend. */
struct Codec::States {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor *> tensors;

    States(ggml_backend_t backend, const std::vector<std::pair<int64_t, int64_t>> & shapes) {
        ggml_init_params params = {ggml_tensor_overhead() * (shapes.size() + 1), nullptr, true};
        ctx = ggml_init(params);
        if (!ctx) throw Error(Fault::OutOfMemory, "cannot create a ggml context");
        for (const auto & [channels, time] : shapes) tensors.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, time));
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) {
            ggml_free(ctx);
            throw Error(Fault::OutOfMemory, "cannot allocate the codec's state");
        }
        ggml_backend_buffer_clear(buffer, 0);
    }
    ~States() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    States(const States &) = delete;
    States & operator=(const States &) = delete;
};

/** The layers of one graph of a stack, which take the stack's states in the order they run. */
struct Codec::Net {
    Graph & g;
    const ModelFile & m;
    States & states;
    float eps;
    size_t next = 0;

    ggml_context * ctx() const { return g.ctx(); }

    /** The next layer's state, and the inputs that follow it in time. */
    ggml_tensor * after_state(ggml_tensor * x) {
        ggml_tensor * state = states.tensors.at(next++);
        return ggml_concat(ctx(), state, x, 1);
    }

    /** Keeps the last `n` inputs of `xp`, a layer's state and its inputs, as the layer's state for the next call. */
    void keep(ggml_tensor * xp, int64_t n) {
        ggml_tensor * state = states.tensors.at(next - 1);
        g.copy(ggml_view_2d(ctx(), xp, xp->ne[0], n, xp->nb[1], (xp->ne[1] - n) * xp->nb[1]), state);
    }

    ggml_tensor * linear(ggml_tensor * x, const std::string & name) {
        return ggml_add(ctx(), mul_mat(ctx(), m.tensor(name + ".weight"), x), m.tensor(name + ".bias"));
    }

    ggml_tensor * rms_norm(ggml_tensor * x, const std::string & name) {
        return ggml_mul(ctx(), ggml_rms_norm(ctx(), x, eps), m.tensor(name));
    }

    /** A causal convolution of stride 1; weight ne = [in, out, k]. */
    ggml_tensor * conv(ggml_tensor * x, const std::string & name) {
        ggml_tensor * w = m.tensor(name + ".weight");
        const int64_t in = w->ne[0], out = w->ne[1], width = w->ne[2], t = x->ne[1];
        ggml_tensor * xp = after_state(x);
        ggml_tensor * y = nullptr;
        for (int64_t k = 0; k < width; k++) {
            ggml_tensor * wk = ggml_view_2d(ctx(), w, in, out, w->nb[1], k * w->nb[2]);
            ggml_tensor * yk = mul_mat(ctx(), wk, ggml_view_2d(ctx(), xp, in, t, xp->nb[1], k * xp->nb[1]));
            y = y ? ggml_add(ctx(), y, yk) : yk;
        }
        keep(xp, width - 1);
        return ggml_add(ctx(), y, m.tensor(name + ".bias"));
    }

    /**
     * A causal convolution of width 2s and stride s, its kernel stored in halves [s * in, out]. An input that is not
     * whole frames of s is padded with zeros on the right, as the official module pads the last part of a clip.
     */
    ggml_tensor * down(ggml_tensor * x, const std::string & name, int64_t s) {
        const int64_t c = x->ne[0], tail = x->ne[1] % s;
        if (tail) x = ggml_concat(ctx(), x, g.zeros(c, s - tail), 1);
        const int64_t n = x->ne[1] / s;
        ggml_tensor * xp = after_state(x);
        ggml_tensor * frames = ggml_reshape_2d(ctx(), xp, c * s, n + 1);
        ggml_tensor * before = ggml_view_2d(ctx(), frames, c * s, n, frames->nb[1], 0);
        ggml_tensor * own = ggml_view_2d(ctx(), frames, c * s, n, frames->nb[1], frames->nb[1]);
        keep(xp, s);
        ggml_tensor * y = ggml_add(ctx(), mul_mat(ctx(), m.tensor(name + ".first"), before), mul_mat(ctx(), m.tensor(name + ".second"), own));
        return ggml_add(ctx(), y, m.tensor(name + ".bias"));
    }

    /** A causal transposed convolution of width 2s and stride s; weight ne = [in, out, 2s]. */
    ggml_tensor * up(ggml_tensor * x, const std::string & name, int64_t s) {
        ggml_tensor * w = m.tensor(name + ".weight");
        const int64_t in = w->ne[0], out = w->ne[1], frames = x->ne[1];
        ggml_tensor * z = ggml_reshape_3d(ctx(), mul_mat(ctx(), ggml_reshape_2d(ctx(), w, in, out * 2 * s), x), out, 2 * s, frames);
        ggml_tensor * own = ggml_view_3d(ctx(), z, out, s, frames, z->nb[1], z->nb[2], 0);
        ggml_tensor * tails = ggml_view_3d(ctx(), z, out, s, frames, z->nb[1], z->nb[2], s * z->nb[1]);
        ggml_tensor * state = states.tensors.at(next++);
        ggml_tensor * held = ggml_reshape_3d(ctx(), state, out, s, 1);
        ggml_tensor * before = frames > 1
            ? ggml_concat(ctx(), held, ggml_cont(ctx(), ggml_view_3d(ctx(), tails, out, s, frames - 1, tails->nb[1], tails->nb[2], 0)), 2)
            : held;
        ggml_tensor * y = ggml_reshape_2d(ctx(), ggml_add(ctx(), ggml_cont(ctx(), own), before), out, s * frames);
        // The copy of this call's last taps does not depend on the sum that reads the state, so the sum goes into the
        // graph first; otherwise the graph would run the copy first and the sum would read the new state.
        g.expand(y);
        g.copy(ggml_cont(ctx(), ggml_view_3d(ctx(), tails, out, s, 1, tails->nb[1], tails->nb[2], (frames - 1) * tails->nb[2])), state);
        return ggml_add(ctx(), y, m.tensor(name + ".bias"));
    }

    /**
     * A block: a causal depthwise convolution of the normed input, scaled and added, then an MLP about an exact GELU of
     * the normed sum, scaled and added. The convolution's state holds its normed inputs.
     */
    ggml_tensor * block(ggml_tensor * x, const std::string & name) {
        ggml_tensor * w = m.tensor(name + ".conv.weight");
        const int64_t c = w->ne[0], width = w->ne[1], t = x->ne[1];
        ggml_tensor * hp = after_state(rms_norm(x, name + ".norm"));
        ggml_tensor * y = nullptr;
        for (int64_t k = 0; k < width; k++) {
            ggml_tensor * yk = ggml_mul(ctx(), ggml_view_2d(ctx(), hp, c, t, hp->nb[1], k * hp->nb[1]), ggml_view_1d(ctx(), w, c, k * w->nb[1]));
            y = y ? ggml_add(ctx(), y, yk) : yk;
        }
        keep(hp, width - 1);
        x = ggml_add(ctx(), x, ggml_mul(ctx(), ggml_add(ctx(), y, m.tensor(name + ".conv.bias")), m.tensor(name + ".gamma")));
        ggml_tensor * f = linear(ggml_gelu_erf(ctx(), linear(rms_norm(x, name + ".ffn_norm"), name + ".fc1")), name + ".fc2");
        return ggml_add(ctx(), x, ggml_mul(ctx(), f, m.tensor(name + ".ffn_gamma")));
    }

    ggml_tensor * blocks(ggml_tensor * x, const std::string & prefix, int count) {
        for (int j = 0; j < count; j++) x = block(x, prefix + std::to_string(j));
        return x;
    }
};

Codec::Codec(const ModelFile & m, ggml_backend_t backend) : m_(m), backend_(backend) {
    const std::string c = "haru.codec.";
    frame_samples_ = m.size("haru.frame_samples");
    latent_ = m.size(c + "latent_dim");
    base_ = m.size(c + "base_channels");
    width_ = m.size(c + "kernel");
    eps_ = m.f32(c + "norm_eps");
    ratios_ = m.i32_array(c + "ratios");
    encoder_depths_ = m.i32_array(c + "encoder_depths");
    decoder_depths_ = m.i32_array(c + "decoder_depths");
    const int n = (int) ratios_.size();

    std::vector<std::pair<int64_t, int64_t>> enc = {{1, width_ - 1}}, dec = {{latent_, width_ - 1}};
    for (int i = 0; i <= n; i++) {
        if (i > 0) enc.push_back({base_ << (i - 1), ratios_[(size_t) (n - i)]});
        for (int j = 0; j < encoder_depths_[(size_t) i]; j++) enc.push_back({base_ << i, width_ - 1});
    }
    enc.push_back({base_ << n, width_ - 1});
    for (int i = 0; i <= n; i++) {
        if (i > 0) dec.push_back({base_ << (n - i), ratios_[(size_t) (i - 1)]});
        for (int j = 0; j < decoder_depths_[(size_t) i]; j++) dec.push_back({base_ << (n - i), width_ - 1});
    }
    dec.push_back({base_, width_ - 1});
    encoder_state_ = new States(backend, enc);
    decoder_state_ = new States(backend, dec);
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!allocr_) throw Error(Fault::OutOfMemory, "cannot create a graph allocator for the codec");
}

Codec::~Codec() {
    ggml_gallocr_free(allocr_);
    delete encoder_state_;
    delete decoder_state_;
}

void Codec::reset_encoder() { ggml_backend_buffer_clear(encoder_state_->buffer, 0); }

void Codec::reset_decoder() { ggml_backend_buffer_clear(decoder_state_->buffer, 0); }

std::vector<float> Codec::encode(const std::vector<float> & audio, CodecStages * stages) {
    const int n = (int) ratios_.size();
    std::vector<float> means;
    for (size_t at = 0; at < audio.size(); at += (size_t) (kEncodeFrames * frame_samples_)) {
        const size_t count = std::min(audio.size() - at, (size_t) (kEncodeFrames * frame_samples_));
        Graph g;
        Net net{g, m_, *encoder_state_, eps_};
        ggml_tensor * x = g.input(std::vector<float>(audio.begin() + (std::ptrdiff_t) at, audio.begin() + (std::ptrdiff_t) (at + count)), 1,
                                  (int64_t) count);
        std::vector<std::pair<std::string, ggml_tensor *>> outs;
        for (int i = 0; i <= n; i++) {
            x = i == 0 ? net.conv(x, "codec.enc.stem")
                       : net.down(x, "codec.enc.down." + std::to_string(i - 1), ratios_[(size_t) (n - i)]);
            x = net.blocks(x, "codec.enc.blk." + std::to_string(i) + ".", encoder_depths_[(size_t) i]);
            if (stages) {
                g.output(x);
                outs.push_back({"encoder_" + std::to_string(i), x});
            }
        }
        x = net.conv(x, "codec.enc.head");
        g.output(x);
        outs.push_back({"encoder_head", x});
        g.compute(backend_, allocr_);
        const std::vector<float> chunk = Graph::read(x);
        means.insert(means.end(), chunk.begin(), chunk.end());
        if (stages) {
            for (const auto & [name, t] : outs) {
                const std::vector<float> v = Graph::read(t);
                std::vector<float> & dst = (*stages)[name];
                dst.insert(dst.end(), v.begin(), v.end());
            }
        }
    }
    return means;
}

std::vector<float> Codec::decode(const std::vector<float> & latents, int64_t frames, CodecStages * stages) {
    if ((int64_t) latents.size() != frames * latent_) {
        throw Error(Fault::InvalidArgument, "the codec decodes " + std::to_string(latents.size()) + " values as " + std::to_string(frames) +
                                                " latents of " + std::to_string(latent_));
    }
    const int n = (int) ratios_.size();
    std::vector<float> audio;
    for (int64_t at = 0; at < frames; at += kDecodeFrames) {
        const int64_t count = std::min(frames - at, kDecodeFrames);
        Graph g;
        Net net{g, m_, *decoder_state_, eps_};
        ggml_tensor * x = g.input(std::vector<float>(latents.begin() + (std::ptrdiff_t) (at * latent_),
                                                     latents.begin() + (std::ptrdiff_t) ((at + count) * latent_)),
                                  latent_, count);
        std::vector<std::pair<std::string, ggml_tensor *>> outs;
        for (int i = 0; i <= n; i++) {
            x = i == 0 ? net.conv(x, "codec.dec.stem") : net.up(x, "codec.dec.up." + std::to_string(i - 1), ratios_[(size_t) (i - 1)]);
            x = net.blocks(x, "codec.dec.blk." + std::to_string(i) + ".", decoder_depths_[(size_t) i]);
            if (stages) {
                g.output(x);
                outs.push_back({"decoder_" + std::to_string(i), x});
            }
        }
        x = net.conv(x, "codec.dec.head");
        g.output(x);
        outs.push_back({"decoder_head", x});
        g.compute(backend_, allocr_);
        const std::vector<float> chunk = Graph::read(x);
        audio.insert(audio.end(), chunk.begin(), chunk.end());
        if (stages) {
            for (const auto & [name, t] : outs) {
                const std::vector<float> v = Graph::read(t);
                std::vector<float> & dst = (*stages)[name];
                dst.insert(dst.end(), v.begin(), v.end());
            }
        }
    }
    return audio;
}

}  // namespace haru
