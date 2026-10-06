#include "talker.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

/*
 * Activations are channel-first ([hidden, tokens]). The talker's position embedding is Qwen's
 * multimodal RoPE, but a text-only sequence gives its three sections the same position, which makes
 * it the ordinary rotate-half RoPE (GGML_ROPE_TYPE_NEOX).
 */

struct GraphCtx {
    ggml_context * ctx;
    ggml_cgraph * gf;
};

namespace {

constexpr int kGraphSize = 8192;

/**
 * The talker's cache grows in steps of this many positions, starting with room for the prompt and as many frames
 * (20 s of speech), and doubling as the speech grows.
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

GraphCtx new_graph() {
    ggml_init_params params = {ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false),
                               nullptr, true};
    ggml_context * ctx = ggml_init(params);
    return {ctx, ggml_new_graph_custom(ctx, kGraphSize, false)};
}

DecoderShape read_shape(const ModelFile & m, const std::string & p) {
    DecoderShape s;
    s.hidden = (int) m.u32(p + ".hidden_size");
    s.ffn = (int) m.u32(p + ".intermediate_size");
    s.n_layer = (int) m.u32(p + ".num_hidden_layers");
    s.n_head = (int) m.u32(p + ".num_attention_heads");
    s.n_kv_head = (int) m.u32(p + ".num_key_value_heads");
    s.head_dim = (int) m.u32(p + ".head_dim");
    s.rms_eps = m.f32(p + ".rms_norm_eps");
    s.rope_theta = m.f32(p + ".rope_theta");
    return s;
}

ggml_tensor * input_i32(ggml_context * ctx, int64_t n) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_input(t);
    return t;
}

}  // namespace

/**
 * The keys and values of every layer of one stack for `capacity` positions: the keys a row per position,
 * [kv heads * head dim, capacity], and the values transposed, a row per channel, [capacity, kv heads * head dim].
 * A step's attention reads both through views, the keys as [head dim, positions, kv heads] and the values as
 * [positions, head dim, kv heads], the operands of its two matrix products, so it reads the cache once and copies
 * none of it. Values kept a row per position need a copy into that layout at every step, with which a step of the 1.7B
 * talker 8192 frames into its speech took 205 ms instead of 28 on Metal and 259 instead of 40 on the CPU of an Apple M5
 * (2026-10-06). ggml_flash_attn_ext reads values a row per position, but it gains at most 3 ms a step there on Metal
 * and takes two to four times as long as the two products on the CPU, where it also sums the values in half precision.
 */
struct Talker::Cache {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor *> k, v;
    int64_t capacity = 0;

    /** A cache with room for `positions` positions, room(positions) in all. */
    Cache(ggml_backend_t backend, ggml_type type, const DecoderShape & s, int64_t positions) : capacity(room(positions)) {
        ggml_init_params params = {ggml_tensor_overhead() * (2 * s.n_layer + 1), nullptr, true};
        ctx = ggml_init(params);
        for (int l = 0; l < s.n_layer; l++) {
            k.push_back(ggml_new_tensor_2d(ctx, type, (int64_t) s.n_kv_head * s.head_dim, capacity));
            v.push_back(ggml_new_tensor_2d(ctx, type, capacity, (int64_t) s.n_kv_head * s.head_dim));
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) {
            ggml_free(ctx);
            throw std::runtime_error("cannot allocate a key/value cache of " + std::to_string(capacity) + " positions");
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

Talker::Talker(const ModelFile & m, ggml_backend_t backend) : backend_(backend), m_(m) {
    talker_ = read_shape(m, "qwen3-tts.talker");
    cp_ = read_shape(m, "qwen3-tts.code_predictor");
    n_groups_ = (int) m.u32("qwen3-tts.talker.num_code_groups");
    vocab_ = (int) m.u32("qwen3-tts.talker.vocab_size");
    cp_vocab_ = (int) m.u32("qwen3-tts.code_predictor.vocab_size");
    max_positions_ = (int) m.u32("qwen3-tts.talker.max_position_embeddings");
    cp_projected_ = cp_.hidden != talker_.hidden;
    cp_cache_ = std::make_unique<Cache>(backend_, GGML_TYPE_F32, cp_, n_groups_ + 1);
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
}

Talker::~Talker() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

int64_t Talker::cache_capacity() const {
    return cache_ ? cache_->capacity : 0;
}

void Talker::resize_cache(int64_t positions) {
    auto next = std::make_unique<Cache>(backend_, GGML_TYPE_F16, talker_, positions);
    if (n_past_ > 0) {
        // The positions so far are copied on the device, layer by layer, into the new buffer.
        GraphCtx g = new_graph();
        for (int l = 0; l < talker_.n_layer; l++) {
            ggml_build_forward_expand(g.gf, ggml_cpy(g.ctx, cache_->keys(g.ctx, l, 0, n_past_), next->keys(g.ctx, l, 0, n_past_)));
            ggml_build_forward_expand(g.gf, ggml_cpy(g.ctx, cache_->values(g.ctx, l, 0, n_past_), next->values(g.ctx, l, 0, n_past_)));
        }
        const bool computed = ggml_gallocr_alloc_graph(allocr_, g.gf) && ggml_backend_graph_compute(backend_, g.gf) == GGML_STATUS_SUCCESS;
        ggml_free(g.ctx);
        if (!computed) throw std::runtime_error("cannot copy the talker's key/value cache into a larger one");
    }
    cache_ = std::move(next);
}

ggml_tensor * Talker::run_stack(GraphCtx & g, const std::string & prefix, const DecoderShape & s, ggml_tensor * x,
                                ggml_tensor * pos, ggml_tensor * mask, Cache & cache, int64_t n_past, int64_t n_tokens) {
    const ModelFile & m = m_;
    ggml_context * ctx = g.ctx;
    const int64_t kv_dim = (int64_t) s.n_kv_head * s.head_dim;
    const int64_t n_kv = n_past + n_tokens;
    for (int l = 0; l < s.n_layer; l++) {
        const std::string b = prefix + ".blk." + std::to_string(l) + ".";
        ggml_tensor * h = ggml_mul(ctx, ggml_rms_norm(ctx, x, s.rms_eps), m.tensor(b + "attn_norm"));
        ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, m.tensor(b + "attn_q"), h), s.head_dim, s.n_head, n_tokens);
        ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, m.tensor(b + "attn_k"), h), s.head_dim, s.n_kv_head, n_tokens);
        ggml_tensor * v = ggml_mul_mat(ctx, m.tensor(b + "attn_v"), h);
        q = ggml_mul(ctx, ggml_rms_norm(ctx, q, s.rms_eps), m.tensor(b + "attn_q_norm"));
        k = ggml_mul(ctx, ggml_rms_norm(ctx, k, s.rms_eps), m.tensor(b + "attn_k_norm"));
        q = ggml_rope_ext(ctx, q, pos, nullptr, s.head_dim, GGML_ROPE_TYPE_NEOX, 0, s.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, s.head_dim, GGML_ROPE_TYPE_NEOX, 0, s.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, k, kv_dim, n_tokens), cache.keys(ctx, l, n_past, n_tokens)));
        ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_transpose(ctx, v), cache.values(ctx, l, n_past, n_tokens)));

        ggml_tensor * kc = cache.k[l];
        ggml_tensor * vc = cache.v[l];
        ggml_tensor * keys = ggml_view_3d(ctx, kc, s.head_dim, s.n_kv_head, n_kv, ggml_row_size(kc->type, s.head_dim), kc->nb[1], 0);
        ggml_tensor * kh = ggml_permute(ctx, keys, 0, 2, 1, 3);
        ggml_tensor * vt = ggml_view_3d(ctx, vc, n_kv, s.head_dim, s.n_kv_head, vc->nb[1], s.head_dim * vc->nb[1], 0);
        // The query is a permuted view like the keys: Vulkan multiplies the keys by a single token in place only when
        // both are permuted alike, and otherwise copies the keys into a contiguous tensor first.
        ggml_tensor * qh = ggml_permute(ctx, q, 0, 2, 1, 3);
        ggml_tensor * kq = ggml_mul_mat(ctx, kh, qh);
        kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f / std::sqrt((float) s.head_dim), 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, vt, kq);
        ggml_tensor * o = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)),
                                          (int64_t) s.n_head * s.head_dim, n_tokens);
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, m.tensor(b + "attn_o"), o));

        h = ggml_mul(ctx, ggml_rms_norm(ctx, x, s.rms_eps), m.tensor(b + "ffn_norm"));
        ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, m.tensor(b + "ffn_gate"), h));
        h = ggml_mul_mat(ctx, m.tensor(b + "ffn_down"), ggml_mul(ctx, gate, ggml_mul_mat(ctx, m.tensor(b + "ffn_up"), h)));
        x = ggml_add(ctx, x, h);
    }
    return x;
}

std::vector<float> Talker::text_embeddings(const std::vector<int32_t> & ids) {
    const ModelFile & m = m_;
    GraphCtx g = new_graph();
    ggml_tensor * idx = input_i32(g.ctx, (int64_t) ids.size());
    ggml_tensor * x = ggml_get_rows(g.ctx, m.tensor("talker.text_embd"), idx);
    x = ggml_add(g.ctx, ggml_mul_mat(g.ctx, m.tensor("talker.text_proj.fc1.weight"), x), m.tensor("talker.text_proj.fc1.bias"));
    x = ggml_silu(g.ctx, x);
    x = ggml_add(g.ctx, ggml_mul_mat(g.ctx, m.tensor("talker.text_proj.fc2.weight"), x), m.tensor("talker.text_proj.fc2.bias"));
    ggml_set_output(x);
    ggml_build_forward_expand(g.gf, x);
    if (!ggml_gallocr_alloc_graph(allocr_, g.gf)) throw std::runtime_error("cannot allocate the text embedding graph");
    ggml_backend_tensor_set(idx, ids.data(), 0, ids.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(backend_, g.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("text embedding failed");
    std::vector<float> out(ggml_nelements(x));
    ggml_backend_tensor_get(x, out.data(), 0, ggml_nbytes(x));
    ggml_free(g.ctx);
    return out;
}

std::vector<float> Talker::codec_embeddings(const std::vector<int32_t> & ids) {
    GraphCtx g = new_graph();
    ggml_tensor * idx = input_i32(g.ctx, (int64_t) ids.size());
    ggml_tensor * x = ggml_get_rows(g.ctx, m_.tensor("talker.codec_embd"), idx);
    ggml_set_output(x);
    ggml_build_forward_expand(g.gf, x);
    if (!ggml_gallocr_alloc_graph(allocr_, g.gf)) throw std::runtime_error("cannot allocate the codec embedding graph");
    ggml_backend_tensor_set(idx, ids.data(), 0, ids.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(backend_, g.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("codec embedding failed");
    std::vector<float> out(ggml_nelements(x));
    ggml_backend_tensor_get(x, out.data(), 0, ggml_nbytes(x));
    ggml_free(g.ctx);
    return out;
}

void Talker::prefill(const std::vector<float> & embeds, int n, int64_t positions) {
    if (positions > max_positions_ || n > positions) {
        throw std::logic_error("an utterance of " + std::to_string(positions) + " positions with a prompt of " + std::to_string(n) +
                               " exceeds the talker's " + std::to_string(max_positions_));
    }
    positions_ = positions;
    n_past_ = 0;
    // A cache that a long utterance grew is given back, so that a model holds what its current utterance needs.
    const int64_t start = std::min(positions, round_up(n + kCacheStep, kCacheStep));
    if (cache_capacity() != Cache::room(start)) resize_cache(start);
    run_talker(&embeds, nullptr, nullptr, n);
}

void Talker::step(const int32_t * codes, const std::vector<float> & extra) {
    run_talker(nullptr, codes, &extra, 1);
}

void Talker::run_talker(const std::vector<float> * embeds, const int32_t * codes, const std::vector<float> * extra,
                        int64_t n) {
    if (n_past_ + n > positions_) throw std::logic_error("the talker was fed past the positions of its utterance");
    if (n_past_ + n > cache_->capacity) {
        resize_cache(std::min(positions_, std::max(n_past_ + n, round_up(2 * cache_->capacity, kCacheStep))));
    }
    const ModelFile & m = m_;
    GraphCtx g = new_graph();
    ggml_context * ctx = g.ctx;

    ggml_tensor * in_embeds = nullptr, * in_extra = nullptr;
    std::vector<ggml_tensor *> in_codes;
    ggml_tensor * x;
    if (embeds) {
        in_embeds = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, talker_.hidden, n);
        ggml_set_input(in_embeds);
        x = in_embeds;
    } else {
        // A frame's input is the sum of its codes' embeddings: the first from the talker's table, the
        // others from the code predictor's, which live in the talker's hidden space.
        for (int q = 0; q < n_groups_; q++) in_codes.push_back(input_i32(ctx, 1));
        x = ggml_get_rows(ctx, m.tensor("talker.codec_embd"), in_codes[0]);
        for (int q = 1; q < n_groups_; q++) {
            x = ggml_add(ctx, x, ggml_get_rows(ctx, m.tensor("cp.codec_embd." + std::to_string(q - 1)), in_codes[q]));
        }
        in_extra = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, talker_.hidden);
        ggml_set_input(in_extra);
        x = ggml_add(ctx, x, in_extra);
    }
    ggml_tensor * pos = input_i32(ctx, n);
    ggml_tensor * mask = nullptr;
    if (n > 1) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_past_ + n, n);
        ggml_set_input(mask);
    }
    x = run_stack(g, "talker", talker_, x, pos, mask, *cache_, n_past_, n);
    x = ggml_mul(ctx, ggml_rms_norm(ctx, x, talker_.rms_eps), m.tensor("talker.norm"));
    ggml_tensor * last = ggml_view_2d(ctx, x, talker_.hidden, 1, x->nb[1], (n - 1) * x->nb[1]);
    ggml_tensor * hidden = ggml_cont(ctx, last);
    ggml_tensor * logits = ggml_mul_mat(ctx, m.tensor("talker.codec_head"), last);
    ggml_set_output(hidden);
    ggml_set_output(logits);
    ggml_build_forward_expand(g.gf, hidden);
    ggml_build_forward_expand(g.gf, logits);
    if (!ggml_gallocr_alloc_graph(allocr_, g.gf)) throw std::runtime_error("cannot allocate the talker graph");

    if (in_embeds) ggml_backend_tensor_set(in_embeds, embeds->data(), 0, ggml_nbytes(in_embeds));
    for (int q = 0; q < (int) in_codes.size(); q++) ggml_backend_tensor_set(in_codes[q], codes + q, 0, sizeof(int32_t));
    if (in_extra) ggml_backend_tensor_set(in_extra, extra->data(), 0, ggml_nbytes(in_extra));
    std::vector<int32_t> positions(n);
    for (int64_t i = 0; i < n; i++) positions[i] = (int32_t) (n_past_ + i);
    ggml_backend_tensor_set(pos, positions.data(), 0, n * sizeof(int32_t));
    if (mask) {
        std::vector<float> md((n_past_ + n) * n);
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = 0; j < n_past_ + n; j++) md[i * (n_past_ + n) + j] = j <= n_past_ + i ? 0.0f : -INFINITY;
        ggml_backend_tensor_set(mask, md.data(), 0, md.size() * sizeof(float));
    }
    if (ggml_backend_graph_compute(backend_, g.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("the talker failed");
    logits_.resize(vocab_);
    hidden_.resize(talker_.hidden);
    ggml_backend_tensor_get(logits, logits_.data(), 0, ggml_nbytes(logits));
    ggml_backend_tensor_get(hidden, hidden_.data(), 0, ggml_nbytes(hidden));
    n_past_ += n;
    ggml_free(ctx);
}

const std::vector<float> & Talker::cp_begin(int32_t code0) {
    cp_past_ = 0;
    return run_cp(code0, 0);
}

const std::vector<float> & Talker::cp_next(int group, int32_t code) {
    return run_cp(code, group);
}

const std::vector<float> & Talker::run_cp(int32_t code, int group) {
    const ModelFile & m = m_;
    GraphCtx g = new_graph();
    ggml_context * ctx = g.ctx;
    ggml_tensor * in_code = input_i32(ctx, 1);
    ggml_tensor * in_hidden = nullptr;
    ggml_tensor * x;
    int64_t n;
    if (group == 0) {
        // The first step reads two positions: the talker's hidden state and the first code's embedding.
        in_hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, talker_.hidden, 1);
        ggml_set_input(in_hidden);
        ggml_tensor * e = ggml_get_rows(ctx, m.tensor("talker.codec_embd"), in_code);
        x = ggml_concat(ctx, in_hidden, e, 1);
        n = 2;
    } else {
        x = ggml_get_rows(ctx, m.tensor("cp.codec_embd." + std::to_string(group - 1)), in_code);
        n = 1;
    }
    if (cp_projected_) x = ggml_add(ctx, ggml_mul_mat(ctx, m.tensor("cp.in_proj.weight"), x), m.tensor("cp.in_proj.bias"));
    ggml_tensor * pos = input_i32(ctx, n);
    ggml_tensor * mask = nullptr;
    if (n > 1) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cp_past_ + n, n);
        ggml_set_input(mask);
    }
    x = run_stack(g, "cp", cp_, x, pos, mask, *cp_cache_, cp_past_, n);
    x = ggml_mul(ctx, ggml_rms_norm(ctx, x, cp_.rms_eps), m.tensor("cp.norm"));
    ggml_tensor * last = ggml_view_2d(ctx, x, cp_.hidden, 1, x->nb[1], (n - 1) * x->nb[1]);
    ggml_tensor * logits = ggml_mul_mat(ctx, m.tensor("cp.head." + std::to_string(group)), last);
    ggml_set_output(logits);
    ggml_build_forward_expand(g.gf, logits);
    if (!ggml_gallocr_alloc_graph(allocr_, g.gf)) throw std::runtime_error("cannot allocate the code predictor graph");

    ggml_backend_tensor_set(in_code, &code, 0, sizeof(int32_t));
    if (in_hidden) ggml_backend_tensor_set(in_hidden, hidden_.data(), 0, ggml_nbytes(in_hidden));
    std::vector<int32_t> positions(n);
    for (int64_t i = 0; i < n; i++) positions[i] = (int32_t) (cp_past_ + i);
    ggml_backend_tensor_set(pos, positions.data(), 0, n * sizeof(int32_t));
    if (mask) {
        std::vector<float> md((cp_past_ + n) * n);
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = 0; j < cp_past_ + n; j++) md[i * (cp_past_ + n) + j] = j <= cp_past_ + i ? 0.0f : -INFINITY;
        ggml_backend_tensor_set(mask, md.data(), 0, md.size() * sizeof(float));
    }
    if (ggml_backend_graph_compute(backend_, g.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("the code predictor failed");
    cp_logits_.resize(cp_vocab_);
    ggml_backend_tensor_get(logits, cp_logits_.data(), 0, ggml_nbytes(logits));
    cp_past_ += n;
    ggml_free(ctx);
    return cp_logits_;
}
