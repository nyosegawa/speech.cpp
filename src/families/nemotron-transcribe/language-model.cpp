#include "language-model.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "error.h"

namespace nemotron_transcribe {

namespace {

/** The rows one graph of a run takes at most. */
constexpr int64_t kBlockRows = 512;

/**
 * The rows times the positions of a block's attention scores at most, which take that times the heads times 4 bytes:
 * 512 rows over 8,192 positions, 512 MiB with 32 heads, where a block of 512 rows at the end of the longest prompt would
 * take 4.3 GB. A block after more rows takes fewer rows.
 */
constexpr int64_t kScoreArea = kBlockRows * 8192;

/** The rows of the block that starts after `past` rows, of `left` rows to run. */
int64_t block_rows(int64_t past, int64_t left) {
    return std::max<int64_t>(1, std::min({kBlockRows, left, kScoreArea / (past + kBlockRows)}));
}

std::string layer_name(int l) {
    return "blk." + std::to_string(l) + ".";
}

}  // namespace

/**
 * The recurrent state of each Mamba layer of the run, [state size, head dim, heads] as ggml_ssm_scan() keeps it, and the
 * inputs of its convolution's last rows, [width - 1, channels].
 */
struct LanguageModel::States {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor *> ssm, conv;

    States(ggml_backend_t backend, int layers, int state, int head_dim, int heads, int width, int64_t channels) {
        ggml_init_params params = {ggml_tensor_overhead() * (size_t) (2 * layers + 1), nullptr, true};
        ctx = ggml_init(params);
        if (!ctx) throw Error(Fault::OutOfMemory, "cannot create a ggml context");
        for (int l = 0; l < layers; l++) {
            ssm.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, state, head_dim, heads));
            conv.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width - 1, channels));
        }
        buffer = layers > 0 ? ggml_backend_alloc_ctx_tensors(ctx, backend) : nullptr;
        if (layers > 0 && !buffer) {
            ggml_free(ctx);
            throw Error(Fault::OutOfMemory, "cannot allocate the Mamba layers' states");
        }
    }
    ~States() {
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    States(const States &) = delete;
    States & operator=(const States &) = delete;
};

LanguageModel::LanguageModel(const ModelFile & m, ggml_backend_t backend, int first, int last, ggml_backend_t cpu) : backend_(backend), m_(m) {
    const std::string x = "nemotron-transcribe.llm.";
    types_ = m.str_array(x + "layer_types");
    first_ = first;
    last_ = last < 0 ? layers() : last;
    if (first_ < 0 || first_ >= last_ || last_ > layers()) throw std::invalid_argument("the language model's layers to run are not among its layers");
    h_ = m.size(x + "hidden_size");
    mamba_heads_ = m.size(x + "mamba_num_heads");
    mamba_head_dim_ = m.size(x + "mamba_head_dim");
    groups_ = m.size(x + "n_groups");
    state_ = m.size(x + "ssm_state_size");
    conv_width_ = m.size(x + "conv_kernel");
    heads_ = m.size(x + "num_attention_heads");
    kv_heads_ = m.size(x + "num_key_value_heads");
    head_dim_ = m.size(x + "head_dim");
    experts_ = m.size(x + "n_routed_experts");
    used_ = m.size(x + "num_experts_per_tok");
    norm_topk_ = m.boolean(x + "norm_topk_prob");
    // Transformers' NemotronH normalizes every block, the gated output of its Mamba mixers and the final state with
    // layer_norm_epsilon.
    eps_ = m.f32(x + "layer_norm_epsilon");
    routed_scale_ = m.f32(x + "routed_scaling_factor");
    int attention_layers = 0, mamba_layers = 0;
    slot_.assign((size_t) layers(), -1);
    for (int l = first_; l < last_; l++) {
        if (types_[(size_t) l] == "attention") slot_[(size_t) l] = attention_layers++;
        else if (types_[(size_t) l] == "mamba") slot_[(size_t) l] = mamba_layers++;
    }
    const int64_t channels = (int64_t) mamba_heads_ * mamba_head_dim_ + 2 * (int64_t) groups_ * state_;
    states_ = std::make_unique<States>(backend_, mamba_layers, state_, mamba_head_dim_, mamba_heads_, conv_width_, channels);
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (cpu) {
        ggml_backend_t backends[] = {backend_, cpu};
        sched_ = ggml_backend_sched_new(backends, nullptr, 2, 32768, false, false);
        if (!sched_) throw Error(Fault::OutOfMemory, "cannot create the scheduler of the language model's graphs");
    }
}

LanguageModel::~LanguageModel() {
    if (sched_) ggml_backend_sched_free(sched_);
    if (allocr_) ggml_gallocr_free(allocr_);
}

void LanguageModel::start(int64_t rows) {
    const int attention_layers = (int) std::count(types_.begin() + first_, types_.begin() + last_, "attention");
    if (attention_layers > 0 && (!keys_values_ || keys_values_->capacity() < rows)) {
        keys_values_.reset();
        keys_values_ = std::make_unique<KeysValues>(backend_, attention_layers, head_dim_, kv_heads_, rows);
    }
    if (states_->buffer) ggml_backend_buffer_clear(states_->buffer, 0);
    capacity_ = rows;
    n_past_ = 0;
}

void LanguageModel::reserve(int64_t rows) {
    start(rows);
    // The largest graph of such a sequence: the block whose scores take the most, kBlockRows rows over kScoreArea's
    // positions, or the last block of the sequence where it is shorter.
    const int64_t block = std::min(kBlockRows, rows);
    n_past_ = std::min(kScoreArea / kBlockRows, rows) - block;
    Graph g;
    build_rows(g, g.zeros(h_, block), nullptr);
    const bool reserved = sched_ ? ggml_backend_sched_reserve(sched_, g.cgraph()) : ggml_gallocr_reserve(allocr_, g.cgraph());
    if (!reserved) throw Error(Fault::OutOfMemory, "cannot reserve the memory of the language model's graphs on " + std::string(ggml_backend_name(backend_)));
    start(rows);
}

ggml_tensor * LanguageModel::mamba(Graph & g, int l, ggml_tensor * x) {
    ggml_context * c = g.ctx();
    const std::string p = layer_name(l);
    const int64_t n = x->ne[1], inner = (int64_t) mamba_heads_ * mamba_head_dim_, bc = (int64_t) groups_ * state_, channels = inner + 2 * bc;
    ggml_tensor * conv_state = states_->conv[(size_t) slot_[(size_t) l]];
    ggml_tensor * state = states_->ssm[(size_t) slot_[(size_t) l]];

    // The input projection gives the gate, the convolution's channels (the values x, then B and C) and dt, in that order.
    ggml_tensor * projected = mul_mat(c, m_.tensor(p + "ssm_in.weight"), x);
    const auto part = [&](int64_t offset, int64_t width) {
        return ggml_view_2d(c, projected, width, n, projected->nb[1], (size_t) offset * sizeof(float));
    };
    ggml_tensor * z = part(0, inner);
    ggml_tensor * dt = part(inner + channels, mamba_heads_);

    // The causal convolution runs over the rows before these, whose inputs the state keeps, and these.
    ggml_tensor * rows = ggml_concat(c, conv_state, ggml_cont(c, ggml_transpose(c, part(inner, channels))), 0);
    g.copy(ggml_view_2d(c, rows, conv_width_ - 1, channels, rows->nb[1], (size_t) n * rows->nb[0]), conv_state);
    ggml_tensor * mixed = ggml_ssm_conv(c, ggml_reshape_3d(c, rows, conv_width_ - 1 + n, channels, 1), m_.tensor(p + "ssm_conv1d.weight"));
    mixed = ggml_silu(c, ggml_add(c, ggml_reshape_2d(c, mixed, channels, n), m_.tensor(p + "ssm_conv1d.bias")));
    const auto split = [&](int64_t offset, int64_t dim, int64_t count) {
        return ggml_view_4d(c, mixed, dim, count, n, 1, (size_t) dim * sizeof(float), mixed->nb[1], mixed->nb[1] * n, (size_t) offset * sizeof(float));
    };
    ggml_tensor * values = split(0, mamba_head_dim_, mamba_heads_);
    ggml_tensor * b = split(inner, state_, groups_);
    ggml_tensor * cc = split(inner + bc, state_, groups_);

    // ggml_ssm_scan() takes dt before its softplus, which it applies itself, with the bias added; the time step limit
    // the layout requires, (0, inf), clamps nothing.
    dt = ggml_reshape_3d(c, ggml_add(c, ggml_cont(c, dt), m_.tensor(p + "ssm_dt.bias")), mamba_heads_, n, 1);
    const std::vector<int32_t> one = {0};
    ggml_tensor * out = ggml_ssm_scan(c, ggml_reshape_4d(c, state, state_, mamba_head_dim_, mamba_heads_, 1), ggml_cont(c, values), dt,
                                      m_.tensor(p + "ssm_a"), ggml_cont(c, b), ggml_cont(c, cc), g.input(one, 1), 1);
    const size_t outputs = (size_t) inner * n;
    ggml_tensor * y = ggml_view_3d(c, out, mamba_head_dim_, mamba_heads_, n, (size_t) mamba_head_dim_ * sizeof(float),
                                   (size_t) inner * sizeof(float), 0);
    g.copy(ggml_view_3d(c, out, state_, mamba_head_dim_, mamba_heads_, state->nb[1], state->nb[2], outputs * sizeof(float)), state);
    // The skip of each head, D times its values.
    y = ggml_add(c, y, ggml_mul(c, ggml_cont(c, values), m_.tensor(p + "ssm_d")));
    // MambaRMSNormGated with norm_before_gate=False: the gate's SiLU first, then RMSNorm over each group of channels.
    ggml_tensor * gated = ggml_mul(c, ggml_reshape_2d(c, y, inner, n), ggml_silu(c, ggml_cont(c, z)));
    gated = ggml_reshape_2d(c, ggml_rms_norm(c, ggml_reshape_3d(c, gated, inner / groups_, groups_, n), eps_), inner, n);
    gated = ggml_mul(c, gated, m_.tensor(p + "ssm_norm.weight"));
    return mul_mat(c, m_.tensor(p + "ssm_out.weight"), gated);
}

ggml_tensor * LanguageModel::attention(Graph & g, int l, ggml_tensor * x, ggml_tensor * mask) {
    ggml_context * c = g.ctx();
    const std::string p = layer_name(l);
    const int64_t n = x->ne[1];
    ggml_tensor * q = ggml_reshape_3d(c, mul_mat(c, m_.tensor(p + "attn_q.weight"), x), head_dim_, heads_, n);
    ggml_tensor * k = ggml_reshape_3d(c, mul_mat(c, m_.tensor(p + "attn_k.weight"), x), head_dim_, kv_heads_, n);
    ggml_tensor * v = ggml_reshape_3d(c, mul_mat(c, m_.tensor(p + "attn_v.weight"), x), head_dim_, kv_heads_, n);
    // NemotronH's attention takes no positions: the Mamba layers order the rows.
    ggml_tensor * o = keys_values_->attend(g, slot_[(size_t) l], q, k, v, n_past_, mask);
    return mul_mat(c, m_.tensor(p + "attn_output.weight"), o);
}

ggml_tensor * LanguageModel::experts(Graph & g, int l, ggml_tensor * x, ggml_tensor ** router) {
    ggml_context * c = g.ctx();
    const std::string p = layer_name(l);
    const int64_t n = x->ne[1];
    // NemotronHTopkRouter: the experts of the highest sigmoid scores with the correction bias, weighted by their scores
    // without it, normalized and scaled.
    *router = mul_mat(c, m_.tensor(p + "ffn_gate_inp.weight"), x);
    ggml_tensor * scores = ggml_sigmoid(c, *router);
    ggml_tensor * chosen = ggml_top_k(c, ggml_add(c, scores, m_.tensor(p + "exp_probs_b.bias")), used_);
    ggml_tensor * weights = ggml_get_rows(c, ggml_reshape_3d(c, scores, 1, experts_, n), chosen);
    if (norm_topk_) weights = ggml_div(c, weights, ggml_reshape_3d(c, ggml_sum_rows(c, ggml_reshape_2d(c, weights, used_, n)), 1, 1, n));
    weights = ggml_scale(c, weights, routed_scale_);
    const auto relu2 = [&](ggml_tensor * t) { return ggml_sqr(c, ggml_relu(c, t)); };
    ggml_tensor * up = mul_mat_id(c, m_.tensor(p + "ffn_up_exps.weight"), ggml_reshape_3d(c, x, h_, 1, n), chosen);
    ggml_tensor * down = ggml_mul(c, mul_mat_id(c, m_.tensor(p + "ffn_down_exps.weight"), relu2(up), chosen), weights);
    ggml_tensor * sum = nullptr;
    for (int e = 0; e < used_; e++) {
        ggml_tensor * one = ggml_view_2d(c, down, h_, n, down->nb[2], (size_t) e * down->nb[1]);
        sum = sum ? ggml_add(c, sum, one) : one;
    }
    ggml_tensor * shared = mul_mat(c, m_.tensor(p + "ffn_down_shexp.weight"), relu2(mul_mat(c, m_.tensor(p + "ffn_up_shexp.weight"), x)));
    return ggml_add(c, sum, shared);
}

std::vector<float> LanguageModel::embed(const std::vector<int32_t> & ids) {
    ggml_tensor * table = m_.tensor("token_embd.weight");
    for (int32_t id : ids) {
        if (id < 0 || id >= table->ne[1]) throw std::invalid_argument("a token is outside the language model's vocabulary");
    }
    Graph g;
    ggml_tensor * e = ggml_get_rows(g.ctx(), table, g.input(ids, (int64_t) ids.size()));
    g.output(e);
    g.compute(backend_, allocr_);
    return Graph::read(e);
}

std::vector<float> LanguageModel::run(const std::vector<float> & input, int64_t n, LanguageModelStages * stages) {
    if (n < 1 || (int64_t) input.size() != n * h_) throw std::invalid_argument("the language model takes an input of each row");
    if (n_past_ + n > capacity_) throw std::invalid_argument("the language model's rows outgrow the sequence it was started for");
    const int count = last_ - first_;
    if (stages) {
        for (auto * s : {&stages->router, &stages->layers}) s->assign((size_t) count, {});
    }
    std::vector<float> out;
    out.reserve(input.size());
    for (int64_t from = 0, rows = 0; from < n; from += rows) {
        rows = block_rows(n_past_, n - from);
        Graph g;
        std::vector<std::vector<ggml_tensor *>> kept((size_t) count);
        ggml_tensor * x = build_rows(g, g.input(std::vector<float>(input.begin() + from * h_, input.begin() + (from + rows) * h_), h_, rows),
                                     stages ? &kept : nullptr);
        if (sched_) g.compute(sched_);
        else g.compute(backend_, allocr_);
        if (stages) {
            for (int i = 0; i < count; i++) {
                std::vector<float> * where[] = {&stages->router[(size_t) i], &stages->layers[(size_t) i]};
                for (int s = 0; s < 2; s++) {
                    if (!kept[(size_t) i][(size_t) s]) continue;
                    const std::vector<float> v = Graph::read(kept[(size_t) i][(size_t) s]);
                    where[s]->insert(where[s]->end(), v.begin(), v.end());
                }
            }
        }
        const std::vector<float> y = Graph::read(x);
        out.insert(out.end(), y.begin(), y.end());
        n_past_ += rows;
    }
    return out;
}

ggml_tensor * LanguageModel::build_rows(Graph & g, ggml_tensor * x, std::vector<std::vector<ggml_tensor *>> * kept) {
    ggml_context * c = g.ctx();
    const int64_t rows = x->ne[1];
    // A graph holds only the inputs its nodes read, and only attention reads the mask.
    const bool attends = std::count(types_.begin() + first_, types_.begin() + last_, "attention") > 0;
    ggml_tensor * mask = attends ? causal_mask(g, n_past_, rows) : nullptr;
    for (int l = first_; l < last_; l++) {
        const std::string & type = types_[(size_t) l];
        ggml_tensor * h = ggml_mul(c, ggml_rms_norm(c, x, eps_), m_.tensor(layer_name(l) + "norm.weight"));
        ggml_tensor * router = nullptr;
        ggml_tensor * mixer = type == "mamba" ? mamba(g, l, h) : type == "attention" ? attention(g, l, h, mask) : experts(g, l, h, &router);
        x = ggml_add(c, x, mixer);
        if (kept) {
            if (router) g.output(router);
            g.output(x);
            (*kept)[(size_t) (l - first_)] = {router, x};
        }
    }
    g.output(x);
    return x;
}

std::vector<float> LanguageModel::logits(const std::vector<float> & hidden) {
    if ((int64_t) hidden.size() != h_) throw std::invalid_argument("the head takes the state of one row");
    Graph g;
    ggml_context * c = g.ctx();
    ggml_tensor * normed = ggml_mul(c, ggml_rms_norm(c, g.input(hidden, h_, 1), eps_), m_.tensor("output_norm.weight"));
    ggml_tensor * l = mul_mat(c, m_.tensor("output.weight"), normed);
    g.output(l);
    g.compute(backend_, allocr_);
    return Graph::read(l);
}

}  // namespace nemotron_transcribe
