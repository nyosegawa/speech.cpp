#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

#include "ggml.h"

// How a model file stores its tensors: the weight types a file can have and the release that first reads each, and
// how a layout stores each tensor in a file of each weight type, which the reader and speech quantize both follow.

/** The first release whose files carry speech.layout; a file without it was written for a release before. */
constexpr const char * kFirstLayoutRelease = "0.7.0";

/** A tensor's shape as ggml gives it, ne[0] first; the axes past the ones given are 1. */
struct Shape {
    int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};

    Shape(std::initializer_list<int64_t> axes);
};

/**
 * A weight type of a model file, the type that holds most of its tensors' bytes: the value of general.file_type that
 * names it, as gguf-py's LlamaFileType numbers it, its ggml type, and the first release of speech.cpp that reads a file
 * of it.
 */
struct WeightType {
    uint32_t file_type;
    ggml_type type;
    const char * release;
};

/** The weight types of the files speech.cpp writes, F32 first: F32, F16, Q8_0, Q6_K, Q5_K and Q4_K. */
const std::vector<WeightType> & weight_types();

/** A ggml type as the model information writes it: F32, F16, Q8_0, Q4_K. */
std::string tensor_type_text(ggml_type type);

/**
 * A type a reader takes a tensor in, and the first release whose reader of the family takes the tensor in that type. A
 * release before the family's layout was first read, which the file's own speech.requires names, adds nothing.
 */
struct Since {
    ggml_type type;
    const char * release;
};

/**
 * How a layout stores a tensor: for a file of each weight type, the types the tensor takes in order, of which the file
 * holds it in the first whose blocks its rows are whole blocks of; and every type a reader takes the tensor in, whatever
 * the file's weight type, with the first release of the family that does, which a file that holds the tensor in that
 * type names in speech.requires. speech quantize writes the type the order gives.
 */
struct Storage {
    struct Choice {
        ggml_type file;
        std::vector<ggml_type> types;
    };
    std::vector<Choice> choices;
    std::vector<Since> since;
};

/**
 * Float32 in a file of every weight type, which every reader of the family takes: a norm, a bias, a kernel or any tensor
 * that an operation reads in float32.
 */
Storage float32_storage();

/**
 * A matrix that only ggml_mul_mat() and ggml_get_rows() read, which ggml computes from every weight type on the CPU,
 * Metal and Vulkan: in the file's own type; in Q8_0 instead of a K-quant where its rows are not whole blocks of 256
 * values; and in F16 instead of Q8_0 where they are not whole blocks of 32; each type read from the release `since`
 * gives it, which differs from family to family (docs/adr/0040).
 */
Storage quantized_storage(std::vector<Since> since);

/**
 * Float32 in an F32 file and float16 in a file of every other weight type, which every reader of the family takes: a
 * weight that an operation reads in F16 or F32 alone, such as a convolution's kernel through ggml_im2col(), or one a
 * family keeps out of quantization.
 */
Storage half_storage();

/** A tensor a layout calls for: its name, its shape, and how the layout stores it. */
struct TensorSpec {
    std::string name;
    Shape shape;
    std::reference_wrapper<const Storage> storage;

    /** The types a file may hold the tensor in. */
    std::vector<ggml_type> types() const;
    /** The type a file of the weight type `file` holds the tensor in. */
    ggml_type type_in(ggml_type file) const;
    /** The first release whose reader of the family takes the tensor in `type`, one of types(). */
    const char * first_release(ggml_type type) const;
};

/** The numbers of a release, MAJOR.MINOR.PATCH as VERSION writes it, or none for text of another form. */
std::optional<std::array<unsigned long, 3>> release_numbers(const std::string & release);

/**
 * Whether the release `a` comes after the release `b`. A release that is not MAJOR.MINOR.PATCH throws an Error of the
 * file, naming it.
 */
bool release_after(const std::string & a, const std::string & b);
