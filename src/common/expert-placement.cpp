#include "expert-placement.h"

#include <cstdlib>

#include "backend.h"
#include "error.h"
#include "log.h"

namespace {

uint64_t gib_tenths(uint64_t bytes) {
    return (bytes * 10 + (1u << 29)) >> 30;
}

std::string gigabytes(uint64_t bytes) {
    const uint64_t tenths = gib_tenths(bytes);
    return std::to_string(tenths / 10) + "." + std::to_string(tenths % 10) + " GiB";
}

}  // namespace

bool is_routed_experts(const std::string & name) {
    const size_t at = name.rfind("_exps");
    return name.rfind("blk.", 0) == 0 && at != std::string::npos && at + 5 == name.size();
}

int layer_of(const std::string & name) {
    if (name.rfind("blk.", 0) != 0) return -1;
    char * end = nullptr;
    const long layer = std::strtol(name.c_str() + 4, &end, 10);
    return end && *end == '.' ? (int) layer : -1;
}

int expert_layers_on_device(const ModelFile & m, int layers, uint64_t working, uint64_t free, const std::string & device, const std::string & model,
                            const std::function<bool(const std::string & name)> & loads) {
    const uint64_t rest = m.weight_bytes([&](const std::string & name) { return loads(name) && !is_routed_experts(name); }) + working;
    if (rest > free) {
        throw Error(Fault::OutOfMemory, device + " has " + gigabytes(free) + " free, and " + model + " needs " + gigabytes(rest) +
                                            " there beside its experts; load it on the CPU or on a GPU with more memory free");
    }
    uint64_t left = free - rest;
    for (int l = 0; l < layers; l++) {
        const uint64_t experts = m.weight_bytes([&](const std::string & name) { return loads(name) && is_routed_experts(name) && layer_of(name) == l; });
        if (experts > left) return l;
        left -= experts;
    }
    return layers;
}

ExpertPlacement::ExpertPlacement(const ModelFile & m, int layers, uint64_t working, ggml_backend_t backend, int threads, const std::string & model,
                                 const std::function<bool(const std::string & name)> & loads)
    : layers_(layers), held_(layers), cpu_(nullptr, ggml_backend_free), pool_(nullptr, ggml_threadpool_free) {
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) return;
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(device, &free, &total);
    held_ = expert_layers_on_device(m, layers, working, free, ggml_backend_name(backend), model, loads);
    if (held_ == layers) return;
    log_message(LogLevel::Warn, model + " computes the experts of " + std::to_string(layers - held_) + " of its " + std::to_string(layers) +
                                    " layers on the CPU, since " + ggml_backend_name(backend) + " holds those of " + std::to_string(held_) +
                                    " beside the rest of the model; a GPU with more memory free runs it faster\n");
    cpu_.reset(start_device(find_device("cpu"), threads));
    // A pool that waits between graphs, since the CPU computes a graph of each layer for every row: ggml's threads
    // started for each graph made a row of Drama 3 take 95 ms where the pool takes 58, on an RTX 2080 with an i9-9900K
    // and 8 threads (2026-10-15).
    ggml_threadpool_params params = ggml_threadpool_params_default(threads);
    pool_.reset(ggml_threadpool_new(&params));
    if (!pool_) throw Error(Fault::OutOfMemory, "cannot start the CPU's threads for the experts");
    ggml_backend_cpu_set_threadpool(cpu_.get(), pool_.get());
}

ggml_backend_t ExpertPlacement::place(const std::string & name, ggml_backend_t backend) const {
    return cpu_ && is_routed_experts(name) && layer_of(name) >= held_ ? cpu_.get() : backend;
}

void ExpertPlacement::allow(bool on_cpu) const {
    if (!cpu_ || on_cpu) return;
    throw Error(Fault::OutOfRange,
                "the GPU holds the experts of " + std::to_string(held_) + " of the model's " + std::to_string(layers_) +
                    " layers and the CPU computes the rest, which experts_on_cpu false refuses; free the GPU's memory or load the model on a GPU with more",
                "experts_on_cpu");
}
