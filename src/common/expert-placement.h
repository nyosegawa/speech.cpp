#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "model-file.h"

/** Whether the tensor `name` holds a layer's routed experts, which a placement may leave in the CPU's memory. */
bool is_routed_experts(const std::string & name);

/** The layer of a layer's tensor `name`, "blk.<layer>." followed by the rest; -1 for a tensor of no layer. */
int layer_of(const std::string & name);

/**
 * The layers, from the first, whose routed experts a device with `free` bytes free holds beside `working` bytes and
 * every other tensor of the model file `m` that `loads` accepts: every one of its `layers` where the whole model fits,
 * fewer where it does not. A device that cannot hold the rest of the model with the experts of no layer throws an Error
 * of its memory naming `device` and the model, `model`.
 */
int expert_layers_on_device(const ModelFile & m, int layers, uint64_t working, uint64_t free, const std::string & device, const std::string & model,
                            const std::function<bool(const std::string & name)> & loads);

/**
 * Where the routed experts of a mixture-of-experts model are: on a GPU, the experts of as many layers from the first as
 * its free memory holds beside the rest of the model, and the CPU's memory the others, which the CPU computes with the
 * load's threads; on the CPU, every expert.
 */
class ExpertPlacement {
public:
    /**
     * The placement of the experts of the `layers` layers of the model file `m`, read for its metadata alone, loaded on
     * `backend` with `threads` threads beside `working` bytes and the tensors `loads` accepts. Where the CPU computes the
     * experts of some layers it says so as a warning, naming the model `model`.
     */
    ExpertPlacement(const ModelFile & m, int layers, uint64_t working, ggml_backend_t backend, int threads, const std::string & model,
                    const std::function<bool(const std::string & name)> & loads);

    /** The backend that loads the tensor `name`: the CPU's for the experts of a layer the GPU does not hold, else `backend`. */
    ggml_backend_t place(const std::string & name, ggml_backend_t backend) const;
    /** The CPU's backend, where it holds the experts of some layers, and nullptr where the model's device holds them all. */
    ggml_backend_t cpu() const { return cpu_.get(); }
    int layers() const { return layers_; }
    /** The layers, from the first, whose experts the model's device holds. */
    int held() const { return held_; }
    /**
     * Refuses a request whose experts_on_cpu is `on_cpu` false, out of range naming the option, where the CPU computes
     * the experts of some layers.
     */
    void allow(bool on_cpu) const;

private:
    int layers_ = 0, held_ = 0;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu_;
    std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool_;
};
