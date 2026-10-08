#include "detector.h"

#include <algorithm>
#include <string>

#include "error.h"
#include "layout.h"

namespace silero_vad {

namespace {

/** The tensors of one block's graph: each chunk's speech probability and the LSTM cell's state after the last. */
struct Block {
    ggml_tensor * probs;
    ggml_tensor * h;
    ggml_tensor * c;
};

/** The graph of `count` chunks of `inputs`, [context + chunk, count], from the cell's state `h` and `c`. */
Block build(Graph & g, const Network & network, const std::vector<float> & inputs, int64_t count, const std::vector<float> & h,
            const std::vector<float> & c) {
    ggml_context * ctx = g.ctx();
    ggml_tensor * input = g.input(inputs, network.context() + network.chunk(), count);
    const Network::Steps steps = network.lstm(ctx, network.encode(ctx, network.stft(ctx, input)), g.input(h, network.hidden()), g.input(c, network.hidden()));
    Block b{network.decode(ctx, steps.h), steps.last_h, steps.last_c};
    g.output(b.probs);
    g.output(b.h);
    g.output(b.c);
    return b;
}

}  // namespace

Detector::Detector(const std::string & path, ggml_backend_t backend)
    : backend_(backend),
      model_(path, backend, layout),
      network_(model_),
      sample_rate_((int) model_.u32("speech.sample_rate")),
      allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {
    if (!allocr_) throw Error(Fault::OutOfMemory, "cannot create a graph allocator");
    // The graph of every block has the operations of two chunks', which a backend may not compute: the cell's on a GPU
    // are hundreds of small operations in a row, and nothing here moves them to the CPU.
    Graph g;
    const std::vector<float> zeros((size_t) (2 * (network_.context() + network_.chunk())), 0.0f), state((size_t) network_.hidden(), 0.0f);
    build(g, network_, zeros, 2, state, state);
    for (int i = 0; i < ggml_graph_n_nodes(g.cgraph()); i++) {
        ggml_tensor * node = ggml_graph_node(g.cgraph(), i);
        if (!ggml_backend_supports_op(backend, node)) {
            ggml_gallocr_free(allocr_);
            throw Error(Fault::Device, std::string(ggml_backend_name(backend)) + " does not compute " + ggml_op_desc(node) + " as Silero VAD needs it; load " +
                                           model_.str("general.name") + " on the CPU (device \"cpu\")",
                        "device");
        }
    }
}

Detector::~Detector() {
    ggml_gallocr_free(allocr_);
}

int64_t Detector::chunks(const std::vector<float> & samples) const {
    return ((int64_t) samples.size() + network_.chunk() - 1) / network_.chunk();
}

std::vector<float> Detector::inputs(const std::vector<float> & samples, int64_t first, int64_t count) const {
    const int64_t chunk = network_.chunk(), context = network_.context(), width = context + chunk, n = (int64_t) samples.size();
    // The audio after `context` zeros, and zeros after it to the end of the last chunk, as get_speech_timestamps() pads
    // the last chunk and the model starts its context.
    std::vector<float> out((size_t) (count * width), 0.0f);
    for (int64_t k = 0; k < count; k++) {
        const int64_t start = (first + k) * chunk - context;
        for (int64_t i = std::max<int64_t>(0, -start); i < width && start + i < n; i++) out[(size_t) (k * width + i)] = samples[(size_t) (start + i)];
    }
    return out;
}

std::vector<float> Detector::probabilities(const std::vector<float> & samples, const std::function<bool(double done)> & progress) {
    const int64_t n = chunks(samples);
    std::vector<float> probs, h((size_t) network_.hidden(), 0.0f), c((size_t) network_.hidden(), 0.0f), block;
    probs.reserve((size_t) n);
    for (int64_t first = 0; first < n; first += kBlock) {
        const int64_t count = std::min(kBlock, n - first);
        Graph g;
        const Block b = build(g, network_, inputs(samples, first, count), count, h, c);
        g.compute(backend_, allocr_);
        Graph::read(b.probs, block);
        probs.insert(probs.end(), block.begin(), block.end());
        Graph::read(b.h, h);
        Graph::read(b.c, c);
        if (progress && !progress((double) (first + count) / (double) n)) break;
    }
    return probs;
}

}  // namespace silero_vad
