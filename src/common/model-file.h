#pragma once

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

class ModelFile;

/**
 * What a reader takes of a GGUF file: the general.architecture it reads, the speech.layout it knows, what to tell
 * the owner of a file it cannot read, and the tensors a file calls for, which `tensors` names after it has read and
 * checked the file's keys.
 */
struct Layout {
    const char * architecture;
    uint32_t version;
    /** What to do with a file this reader refuses, such as "convert it again with reference/x/convert.py". */
    const char * remedy;
    std::vector<std::string> (*tensors)(const ModelFile & file);
};

/**
 * The general.architecture of a GGUF file, read without loading its tensors. A file without speech.layout, written
 * for a release before layouts began, throws.
 */
std::string gguf_architecture(const std::string & path);

/**
 * A GGUF file of a known layout whose tensors live in one backend buffer, and whose metadata stays readable. A key
 * is read by its exact type: one that is missing or of another type throws with a message naming it, where ggml's
 * own getters would abort the process.
 */
class ModelFile {
public:
    /**
     * Opens the file, checks its architecture and layout against `layout` and its tensors against the ones its keys
     * call for, a tensor missing or one not called for included, and only then loads the tensors onto `backend`.
     */
    ModelFile(const std::string & path, ggml_backend_t backend, const Layout & layout);
    ModelFile(const ModelFile &) = delete;
    ModelFile & operator=(const ModelFile &) = delete;

    ggml_tensor * tensor(const std::string & name) const;

    uint32_t u32(const std::string & key) const;
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

private:
    /** The key's id, once it is known to have `type` and, for an array, `element`. */
    int64_t key_id(const std::string & key, gguf_type type, gguf_type element = GGUF_TYPE_COUNT) const;
    /** "layout 1 of qwen3-tts", or the architecture alone while the layout is not yet read. */
    std::string layout_name() const;
    void check_tensors(const Layout & layout) const;
    void load(ggml_backend_t backend);

    std::string path_, architecture_, remedy_;
    uint32_t version_ = 0;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf_{nullptr, gguf_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx_{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer_{nullptr, ggml_backend_buffer_free};
};

/**
 * Reads and checks the keys every model file has: general.name, general.license, general.source.url,
 * speech.sample_rate, speech.languages, and speech.task and speech.language_use, which must be the family's `task`
 * and `language_use`.
 */
void check_model_keys(const ModelFile & file, const char * task, const char * language_use);

/**
 * Appends `prefix` followed by each of 0 to `count` - 1 and each of `suffixes` after a dot, or alone for an empty
 * suffix, to `names`: the tensors of numbered blocks.
 */
void add_numbered(std::vector<std::string> & names, const std::string & prefix, uint32_t count,
                  std::initializer_list<const char *> suffixes);
