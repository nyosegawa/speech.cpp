#include "text-model.h"

#include <cmath>
#include <stdexcept>

#include "error.h"

namespace drama3 {

TextModel::TextModel(const ModelFile & m, ggml_backend_t backend, int first, int last, ggml_backend_t cpu)
    : backend_(backend), m_(m), decoder_(m, "drama3.text.", backend, first, last, cpu) {
    const std::string a = "drama3.audio_decoder.";
    h_ = m.size("drama3.text.hidden_size");
    codebooks_ = m.count(a + "num_codebooks");
    codes_ = m.size(a + "vocab_size");
    seen_ = (int) m.u32(a + "vq_backbone_codebooks");
    semantic_begin_ = (int32_t) m.u32("drama3.semantic_start_token_id");
    semantic_end_ = (int32_t) m.u32("drama3.semantic_end_token_id");
}

std::vector<float> TextModel::embed(const std::vector<int32_t> & frames, int64_t n) {
    Graph g;
    ggml_tensor * e = build_embed(g, frames, n);
    g.output(e);
    g.compute(backend_, decoder_.allocator());
    return Graph::read(e);
}

ggml_tensor * TextModel::build_embed(Graph & g, const std::vector<int32_t> & frames, int64_t n) {
    const int width = 1 + codebooks_;
    if (n < 1 || (int64_t) frames.size() != n * width) throw std::invalid_argument("the text model embeds whole frames");
    ggml_tensor * table = m_.tensor("token_embd");
    std::vector<int32_t> ids((size_t) n), seen;
    std::vector<float> audio((size_t) n), text((size_t) n);
    for (int64_t i = 0; i < n; i++) {
        const int32_t id = frames[(size_t) (i * width)];
        if (id < 0 || id >= table->ne[1]) throw std::invalid_argument("a frame's main token is outside the vocabulary");
        ids[(size_t) i] = id;
        audio[(size_t) i] = id >= semantic_begin_ && id <= semantic_end_ ? 1.0f : 0.0f;
        text[(size_t) i] = 1.0f - audio[(size_t) i];
    }
    // The file holds the embeddings of codebooks 1 to 9, codebook b's from row (b - 1) * vocab_size.
    for (int b = 1; b <= seen_; b++) {
        for (int64_t i = 0; i < n; i++) {
            const int32_t code = frames[(size_t) (i * width + 1 + b)];
            if (code < 0 || code >= codes_) throw std::invalid_argument("a frame's code is outside its codebook");
            seen.push_back((b - 1) * codes_ + code);
        }
    }
    ggml_context * c = g.ctx();
    ggml_tensor * e = ggml_get_rows(c, table, g.input(ids, n));
    if (seen_ > 0) {
        ggml_tensor * rows = ggml_get_rows(c, m_.tensor("ad.codebook_embd"), g.input(seen, (int64_t) seen.size()));
        ggml_tensor * mixed = e;
        for (int b = 0; b < seen_; b++) mixed = ggml_add(c, mixed, ggml_view_2d(c, rows, h_, n, rows->nb[1], (size_t) (b * n) * rows->nb[1]));
        mixed = ggml_scale(c, mixed, 1.0f / std::sqrt((float) (1 + seen_)));
        e = ggml_add(c, ggml_mul(c, e, g.input(text, 1, n)), ggml_mul(c, mixed, g.input(audio, 1, n)));
    }
    return e;
}

void TextModel::reserve(int64_t rows, int64_t prompt_rows) {
    if (prompt_rows < 1 || prompt_rows > rows) throw std::invalid_argument("the text model reserves for a prompt within its sequence");
    // The embeddings of the longest prompt, the largest graph of such a sequence besides the decoder's own.
    std::vector<int32_t> frames((size_t) (prompt_rows * (1 + codebooks_)), 0);
    Graph embedding;
    embedding.output(build_embed(embedding, frames, prompt_rows));
    decoder_.reserve(rows, prompt_rows, embedding.cgraph());
}

std::vector<float> TextModel::logits(const std::vector<float> & hidden) {
    if ((int64_t) hidden.size() != h_) throw std::invalid_argument("the head takes the state of one row");
    Graph g;
    ggml_tensor * l = mul_mat(g.ctx(), m_.tensor("output"), g.input(hidden, h_, 1));
    g.output(l);
    g.compute(backend_, decoder_.allocator());
    return Graph::read(l);
}

}  // namespace drama3
