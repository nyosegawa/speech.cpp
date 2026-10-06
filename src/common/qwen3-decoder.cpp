#include "qwen3-decoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "error.h"

/*
 * Activations are channel-first ([hidden, rows]). Every row has one position, which in Qwen's multimodal RoPE stands
 * for the same position in each of its three sections, and there it is the ordinary rotate-half RoPE
 * (GGML_ROPE_TYPE_NEOX).
 */

namespace {

constexpr int kGraphSize = 8192;

/**
 * The cache grows in steps of this many positions, starting with room for the first run and as many more (20 s of
 * Qwen3-TTS's speech), and doubling as the sequence grows.
 */
constexpr int64_t kCacheStep = 256;

/**
 * A cache holds a multiple of this many positions, so that every row of its transposed values starts on 32 bytes in
 * half precision, as ggml aligns a tensor. Metal's matrix-vector product reads a row four values at a time whenever
 * its length is a multiple of four, whatever the stride between rows, and a row that does not start on four bytes
 * reads the wrong values: a cache of 85 positions puts the talker's logits 42% off on Metal.
 */
constexpr int64_t kCacheAlign = 16;

int64_t round_up(int64_t n, int64_t step) {
    return (n + step - 1) / step * step;
}

}  // namespace

Qwen3Shape read_qwen3_shape(const ModelFile & m, const std::string & prefix) {
    Qwen3Shape s;
    s.hidden = (int) m.u32(prefix + ".hidden_size");
    s.ffn = (int) m.u32(prefix + ".intermediate_size");
    s.n_layer = (int) m.u32(prefix + ".num_hidden_layers");
    s.n_head = (int) m.u32(prefix + ".num_attention_heads");
    s.n_kv_head = (int) m.u32(prefix + ".num_key_value_heads");
    s.head_dim = (int) m.u32(prefix + ".head_dim");
    s.rms_eps = m.f32(prefix + ".rms_norm_eps");
    s.rope_theta = m.f32(prefix + ".rope_theta");
    return s;
}

/**
 * The keys and values of every layer for `capacity` positions: the keys a row per position,
 * [kv heads * head dim, capacity], and the values transposed, a row per channel, [capacity, kv heads * head dim].
 * Attention reads both through views, the keys as [head dim, positions, kv heads] and the values as
 * [positions, head dim, kv heads], the operands of its two matrix products, so it reads the cache once and copies
 * none of it. Values kept a row per position need a copy into that layout at every step, with which a step of the 1.7B
 * talker 8192 frames into its speech took 205 ms instead of 28 on Metal and 259 instead of 40 on the CPU of an Apple M5
 * (2026-10-06). ggml_flash_attn_ext reads values a row per position, but it gains at most 3 ms a step there on Metal
 * and takes two to four times as long as the two products on the CPU, where it also sums the values in half precision.
 */
struct Qwen3Decoder::Cache {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor *> k, v;
    int64_t capacity = 0;

    /** A cache with room for `positions` positions, room(positions) in all. */
    Cache(ggml_backend_t backend, ggml_type type, const Qwen3Shape & s, int64_t positions) : capacity(room(positions)) {
        ggml_init_params params = {ggml_tensor_overhead() * (2 * s.n_layer + 1), nullptr, true};
        ctx = ggml_init(params);
        for (int l = 0; l < s.n_layer; l++) {
            k.push_back(ggml_new_tensor_2d(ctx, type, (int64_t) s.n_kv_head * s.head_dim, capacity));
            v.push_back(ggml_new_tensor_2d(ctx, type, capacity, (int64_t) s.n_kv_head * s.head_dim));
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) {
            ggml_free(ctx);
            throw Error(Fault::OutOfMemory, "cannot allocate a key/value cache of " + std::to_string(capacity) + " positions");
        }
    }
    ~Cache() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    Cache(const Cache &) = delete;
    Cache & operator=(const Cache &) = delete;

    /** The positions a cache made for `positions` holds: `positions` rounded up to a multiple of kCacheAlign. */
    static int64_t room(int64_t positions) { return round_up(positions, kCacheAlign); }

    /** The keys of layer `l` at positions [from, from + n), [kv dim, n]. */
    ggml_tensor * keys(ggml_context * c, int l, int64_t from, int64_t n) const {
        return ggml_view_2d(c, k[l], k[l]->ne[0], n, k[l]->nb[1], from * k[l]->nb[1]);
    }
    /** The values of layer `l` at positions [from, from + n), [n, kv dim]. */
    ggml_tensor * values(ggml_context * c, int l, int64_t from, int64_t n) const {
        return ggml_view_2d(c, v[l], n, v[l]->ne[1], v[l]->nb[1], from * v[l]->nb[0]);
    }
};

Qwen3Decoder::Qwen3Decoder(const ModelFile & m, ggml_backend_t backend, std::string tensors, const Qwen3Shape & shape,
                           int64_t max_positions, ggml_type cache_type, int64_t block_rows)
    : backend_(backend),
      m_(m),
      tensors_(std::move(tensors)),
      shape_(shape),
      max_positions_(max_positions),
      cache_type_(cache_type),
      block_rows_(block_rows) {
    if (block_rows_ < 1) throw std::logic_error("a decoder runs at least one row in a graph");
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
}

Qwen3Decoder::~Qwen3Decoder() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

int64_t Qwen3Decoder::cache_capacity() const {
    return cache_ ? cache_->capacity : 0;
}

size_t Qwen3Decoder::graph_bytes() const {
    return ggml_gallocr_get_buffer_size(allocr_, 0);
}

void Qwen3Decoder::read_cache(int layer, std::vector<float> & keys, std::vector<float> & values) const {
    const int64_t kv_dim = (int64_t) shape_.n_kv_head * shape_.head_dim, capacity = cache_->capacity;
    const auto as_floats = [&](const ggml_tensor * t, int64_t count) {
        std::vector<uint8_t> raw(ggml_row_size(t->type, count));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        std::vector<float> out((size_t) count);
        if (t->type == GGML_TYPE_F32) std::memcpy(out.data(), raw.data(), raw.size());
        else ggml_get_type_traits(t->type)->to_float(raw.data(), out.data(), count);
        return out;
    };
    keys = as_floats(cache_->k[layer], n_past_ * kv_dim);
    const std::vector<float> transposed = as_floats(cache_->v[layer], capacity * kv_dim);
    values.resize((size_t) (n_past_ * kv_dim));
    for (int64_t p = 0; p < n_past_; p++) {
        for (int64_t c = 0; c < kv_dim; c++) values[p * kv_dim + c] = transposed[c * capacity + p];
    }
}

void Qwen3Decoder::resize_cache(int64_t positions) {
    auto next = std::make_unique<Cache>(backend_, cache_type_, shape_, positions);
    if (n_past_ > 0) {
        // The positions so far are copied on the device, layer by layer, into the new buffer.
        Graph g(kGraphSize);
        for (int l = 0; l < shape_.n_layer; l++) {
            g.copy(cache_->keys(g.ctx(), l, 0, n_past_), next->keys(g.ctx(), l, 0, n_past_));
            g.copy(cache_->values(g.ctx(), l, 0, n_past_), next->values(g.ctx(), l, 0, n_past_));
        }
        g.compute(backend_, allocr_);
    }
    cache_ = std::move(next);
}

void Qwen3Decoder::start(int64_t positions, int64_t first) {
    if (positions > max_positions_ || first < 1 || first > positions) {
        throw std::logic_error("a sequence of " + std::to_string(positions) + " positions whose first run has " + std::to_string(first) +
                               " rows does not fit a decoder of " + std::to_string(max_positions_) + " positions");
    }
    positions_ = positions;
    n_past_ = 0;
    const int64_t room = std::min(positions, round_up(first + kCacheStep, kCacheStep));
    if (cache_capacity() != Cache::room(room)) resize_cache(room);
}

ggml_tensor * Qwen3Decoder::layers(Graph & g, ggml_tensor * x, int64_t rows) {
    const ModelFile & m = m_;
    const Qwen3Shape & s = shape_;
    ggml_context * ctx = g.ctx();
    const int64_t kv_dim = (int64_t) s.n_kv_head * s.head_dim;
    const int64_t n_kv = n_past_ + rows;

    std::vector<int32_t> positions(rows);
    for (int64_t i = 0; i < rows; i++) positions[i] = (int32_t) (n_past_ + i);
    ggml_tensor * pos = g.input(positions, rows);
    // A row sees the positions before its block and the rows of its block up to itself.
    ggml_tensor * mask = nullptr;
    if (rows > 1) {
        std::vector<float> causal((size_t) (n_kv * rows));
        for (int64_t i = 0; i < rows; i++) {
            for (int64_t j = 0; j < n_kv; j++) causal[i * n_kv + j] = j <= n_past_ + i ? 0.0f : -INFINITY;
        }
        mask = g.input(causal, n_kv, rows);
    }

    for (int l = 0; l < s.n_layer; l++) {
        const std::string b = tensors_ + ".blk." + std::to_string(l) + ".";
        ggml_tensor * h = ggml_mul(ctx, ggml_rms_norm(ctx, x, s.rms_eps), m.tensor(b + "attn_norm"));
        ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, m.tensor(b + "attn_q"), h), s.head_dim, s.n_head, rows);
        ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, m.tensor(b + "attn_k"), h), s.head_dim, s.n_kv_head, rows);
        ggml_tensor * v = ggml_mul_mat(ctx, m.tensor(b + "attn_v"), h);
        q = ggml_mul(ctx, ggml_rms_norm(ctx, q, s.rms_eps), m.tensor(b + "attn_q_norm"));
        k = ggml_mul(ctx, ggml_rms_norm(ctx, k, s.rms_eps), m.tensor(b + "attn_k_norm"));
        q = ggml_rope_ext(ctx, q, pos, nullptr, s.head_dim, GGML_ROPE_TYPE_NEOX, 0, s.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, s.head_dim, GGML_ROPE_TYPE_NEOX, 0, s.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        g.copy(ggml_reshape_2d(ctx, k, kv_dim, rows), cache_->keys(ctx, l, n_past_, rows));
        g.copy(ggml_transpose(ctx, v), cache_->values(ctx, l, n_past_, rows));

        ggml_tensor * kc = cache_->k[l];
        ggml_tensor * vc = cache_->v[l];
        ggml_tensor * keys = ggml_view_3d(ctx, kc, s.head_dim, s.n_kv_head, n_kv, ggml_row_size(kc->type, s.head_dim), kc->nb[1], 0);
        ggml_tensor * kh = ggml_permute(ctx, keys, 0, 2, 1, 3);
        ggml_tensor * vt = ggml_view_3d(ctx, vc, n_kv, s.head_dim, s.n_kv_head, vc->nb[1], s.head_dim * vc->nb[1], 0);
        // The query is a permuted view like the keys: Vulkan multiplies the keys by a single row in place only when
        // both are permuted alike, and otherwise copies the keys into a contiguous tensor first.
        ggml_tensor * qh = ggml_permute(ctx, q, 0, 2, 1, 3);
        ggml_tensor * kq = ggml_mul_mat(ctx, kh, qh);
        kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f / std::sqrt((float) s.head_dim), 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, vt, kq);
        ggml_tensor * o = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), (int64_t) s.n_head * s.head_dim, rows);
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, m.tensor(b + "attn_o"), o));

        h = ggml_mul(ctx, ggml_rms_norm(ctx, x, s.rms_eps), m.tensor(b + "ffn_norm"));
        ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, m.tensor(b + "ffn_gate"), h));
        h = ggml_mul_mat(ctx, m.tensor(b + "ffn_down"), ggml_mul(ctx, gate, ggml_mul_mat(ctx, m.tensor(b + "ffn_up"), h)));
        x = ggml_add(ctx, x, h);
    }
    return x;
}

void Qwen3Decoder::run(int64_t n, const Qwen3Rows & rows, ggml_tensor * head) {
    if (!cache_ || n < 1 || n_past_ + n > positions_) throw std::logic_error("a decoder was fed past the positions of its sequence");
    if (n_past_ + n > cache_->capacity) {
        resize_cache(std::min(positions_, std::max(n_past_ + n, round_up(2 * cache_->capacity, kCacheStep))));
    }
    for (int64_t from = 0; from < n; from += block_rows_) {
        const int64_t count = std::min(block_rows_, n - from);
        Graph g(kGraphSize);
        ggml_tensor * x = layers(g, rows(g, from, count), count);
        ggml_tensor * hidden = nullptr, * logits = nullptr;
        // A block whose output nobody reads computes only what the cache keeps: its last layer's keys and values.
        if (head && from + count == n) {
            ggml_context * ctx = g.ctx();
            ggml_tensor * last = ggml_view_2d(ctx, x, shape_.hidden, 1, x->nb[1], (count - 1) * x->nb[1]);
            hidden = ggml_mul(ctx, ggml_rms_norm(ctx, last, shape_.rms_eps), m_.tensor(tensors_ + ".norm"));
            logits = ggml_mul_mat(ctx, head, hidden);
            g.output(hidden);
            g.output(logits);
        }
        g.compute(backend_, allocr_);
        if (hidden) {
            hidden_ = Graph::read(hidden);
            logits_ = Graph::read(logits);
        }
        n_past_ += count;
    }
}
