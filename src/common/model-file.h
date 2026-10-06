#pragma once

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

class ModelFile;

/** A tensor's shape as ggml gives it, ne[0] first; the axes past the ones given are 1. */
struct Shape {
    int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};

    Shape(std::initializer_list<int64_t> axes);
};

/** A tensor a layout calls for: its name, its shape, and the ggml types its converter may store it in. */
struct TensorSpec {
    std::string name;
    Shape shape;
    std::vector<ggml_type> types;
};

/**
 * What a reader takes of a GGUF file: the general.architecture it reads, the speech.layout it knows, what to tell
 * the owner of a file it cannot read, and the tensors a file calls for, which `tensors` names with the shape and the
 * types of each after it has read and checked the file's keys.
 */
struct Layout {
    const char * architecture;
    uint32_t version;
    /** What to do with a file this reader refuses, such as "convert it again with reference/x/convert.py". */
    const char * remedy;
    std::function<std::vector<TensorSpec>(const ModelFile & file)> tensors;
};

/**
 * The general.architecture of a GGUF file, read without loading its tensors. A file without speech.layout, written
 * for a release before layouts began, throws.
 */
std::string gguf_architecture(const std::string & path);

/**
 * A GGUF file of a known layout whose metadata stays readable and whose tensors, all of them or those a reader asks
 * for, live in one backend buffer. A key is read by its exact type: one that is missing or of another type throws
 * with a message naming it, where ggml's own getters would abort the process. Every failure throws an Error: Io for
 * a file that cannot be opened or read, File for one whose content this release cannot use.
 */
class ModelFile {
public:
    /**
     * Opens the file, checks its architecture and layout against `layout` and its tensors against the ones its keys
     * call for, a tensor missing, one not called for, and one of another shape or type included, and only then loads
     * the tensors onto `backend`: every tensor, or those whose names `keep` accepts.
     */
    ModelFile(const std::string & path, ggml_backend_t backend, const Layout & layout,
              const std::function<bool(const std::string & name)> & keep = {});
    /** Opens and checks the file as above and loads no tensor, for a reader of its metadata alone. */
    ModelFile(const std::string & path, const Layout & layout);
    ModelFile(const ModelFile &) = delete;
    ModelFile & operator=(const ModelFile &) = delete;

    /** A loaded tensor; one the file does not hold or that was not loaded throws. */
    ggml_tensor * tensor(const std::string & name) const;

    uint32_t u32(const std::string & key) const;
    /**
     * A u32 key that sizes the model, such as a width or a number of heads: 0 throws, and so does a value past the
     * range of int, in which the families hold sizes.
     */
    int size(const std::string & key) const;
    /**
     * A u32 key that numbers layers or blocks of tensors: 0 throws, and so does a number past the tensors the file
     * holds, since each layer holds one at least.
     */
    int count(const std::string & key) const;
    /**
     * Axis `axis` of the tensor `name` as the file stores it, for a layout to take a width that no key gives from one
     * tensor and check the others against it. A tensor the file does not hold throws, and so does a width of 0 or one
     * past the range of int.
     */
    int64_t width(const std::string & name, int axis) const;
    /** The number of tensors the file holds. */
    int64_t tensor_count() const;
    float f32(const std::string & key) const;
    bool boolean(const std::string & key) const;
    std::string str(const std::string & key) const;
    /** A string that names a kind; a value other than `values` throws. */
    std::string one_of(const std::string & key, std::initializer_list<const char *> values) const;
    std::vector<int32_t> i32_array(const std::string & key) const;
    std::vector<double> f64_array(const std::string & key) const;
    std::vector<std::string> str_array(const std::string & key) const;

    const std::string & path() const { return path_; }
    /** The layout's remedy, for a message about the file that a family's reader throws. */
    const std::string & remedy() const { return remedy_; }
    uint32_t layout_version() const { return version_; }

    /** The size of the file in bytes. */
    uint64_t file_bytes() const;
    /** The bytes of every tensor the file holds, as it stores them. */
    uint64_t weight_bytes() const;
    /** The number of metadata entries, the key of the one at `index`, and its value as JSON text. */
    size_t meta_count() const;
    std::string meta_key(size_t index) const;
    std::string meta_json(size_t index) const;

private:
    /** Opens and checks the file, which both constructors do. */
    void open(const Layout & layout);
    /** The key's id, once it is known to have `type` and, for an array, `element`. */
    int64_t key_id(const std::string & key, gguf_type type, gguf_type element = GGUF_TYPE_COUNT) const;
    /** "layout 1 of qwen3-tts", or the architecture alone while the layout is not yet read. */
    std::string layout_name() const;
    void check_tensors(const Layout & layout) const;
    void load(ggml_backend_t backend, const std::function<bool(const std::string & name)> & keep);

    std::string path_, architecture_, remedy_;
    uint32_t version_ = 0;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf_{nullptr, gguf_free};
    /** The shapes and types of every tensor of the file, which ggml reads with its metadata. */
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx_{nullptr, ggml_free};
    /** The copies of the tensors a reader keeps, when it keeps only some. */
    std::unique_ptr<ggml_context, decltype(&ggml_free)> kept_{nullptr, ggml_free};
    /** The context of the loaded tensors: ctx_, kept_, or none for a reader of the metadata alone. */
    ggml_context * loaded_ = nullptr;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer_{nullptr, ggml_backend_buffer_free};
};

/**
 * Reads and checks the keys every model file has: general.name, general.license, general.source.url,
 * speech.sample_rate, speech.languages, and speech.task and speech.language_use, which must be the family's `task`
 * and `language_use`.
 */
void check_model_keys(const ModelFile & file, const char * task, const char * language_use);

/** Appends a tensor of each of `parts`, named `prefix` and the part's name, to `tensors`: the tensors of one block. */
void add_block(std::vector<TensorSpec> & tensors, const std::string & prefix, std::initializer_list<TensorSpec> parts);

/**
 * Appends, for each of 0 to `count` - 1, a tensor of each of `parts`, named `prefix`, the number and the part's name
 * after a dot, or `prefix` and the number alone for a part of an empty name, to `tensors`: the tensors of numbered
 * blocks of the same shapes.
 */
void add_numbered(std::vector<TensorSpec> & tensors, const std::string & prefix, int count, std::initializer_list<TensorSpec> parts);
