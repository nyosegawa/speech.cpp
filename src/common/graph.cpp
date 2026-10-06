#include "graph.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "error.h"

Graph::Graph(int max_nodes) {
    ggml_init_params params = {ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false),
                               nullptr, true};
    ctx_ = ggml_init(params);
    if (!ctx_) throw Error(Fault::OutOfMemory, "cannot create a ggml context");
    gf_ = ggml_new_graph_custom(ctx_, max_nodes, false);
}

Graph::~Graph() {
    if (ctx_) ggml_free(ctx_);
}

ggml_tensor * Graph::input(const std::vector<float> & data, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ggml_tensor * t = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    if ((int64_t) data.size() != ggml_nelements(t)) throw std::runtime_error("an input's data does not fit its shape");
    ggml_set_input(t);
    Upload u{t, std::vector<uint8_t>(ggml_nbytes(t))};
    std::memcpy(u.bytes.data(), data.data(), u.bytes.size());
    uploads_.push_back(std::move(u));
    return t;
}

ggml_tensor * Graph::input(const std::vector<int32_t> & data, int64_t ne0) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, ne0);
    if ((int64_t) data.size() != ne0) throw std::runtime_error("an input's data does not fit its shape");
    ggml_set_input(t);
    Upload u{t, std::vector<uint8_t>(ggml_nbytes(t))};
    std::memcpy(u.bytes.data(), data.data(), u.bytes.size());
    uploads_.push_back(std::move(u));
    return t;
}

ggml_tensor * Graph::zeros(int64_t ne0, int64_t ne1) {
    return input(std::vector<float>((size_t) (ne0 * ne1), 0.0f), ne0, ne1);
}

void Graph::output(ggml_tensor * t) {
    ggml_set_output(t);
    ggml_build_forward_expand(gf_, t);
}

void Graph::copy(ggml_tensor * src, ggml_tensor * dst) {
    ggml_build_forward_expand(gf_, ggml_cpy(ctx_, src, dst));
}

void Graph::compute(ggml_backend_t backend, ggml_gallocr_t allocr) {
    if (!ggml_gallocr_alloc_graph(allocr, gf_)) throw Error(Fault::OutOfMemory, "cannot allocate the memory of a graph on the device");
    for (const Upload & u : uploads_) ggml_backend_tensor_set(u.tensor, u.bytes.data(), 0, u.bytes.size());
    if (ggml_backend_graph_compute(backend, gf_) != GGML_STATUS_SUCCESS) {
        throw Error(Fault::Device, std::string("the device ") + ggml_backend_name(backend) + " failed to compute a graph");
    }
}

std::vector<float> Graph::read(const ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32) throw std::runtime_error("a result is not float32");
    std::vector<float> out(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    return out;
}

