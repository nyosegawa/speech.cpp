#include "model-file.h"

#include <algorithm>
#include <cstdio>
#include <set>
#include <stdexcept>

namespace {

/** The first release whose files carry speech.layout; a file without it was written for a release before. */
constexpr const char * kFirstLayoutRelease = "0.7.0";

/** fseek() takes a long, which is 32 bits on Windows, so it cannot reach a tensor past 2 GiB there. */
bool seek(FILE * f, uint64_t offset) {
#ifdef _WIN32
    return _fseeki64(f, (int64_t) offset, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t) offset, SEEK_SET) == 0;
#endif
}

using Gguf = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

Gguf open_gguf(const std::string & path, ggml_context ** ctx) {
    gguf_init_params params = {/*no_alloc =*/true, /*ctx =*/ctx};
    Gguf gguf(gguf_init_from_file(path.c_str(), params), gguf_free);
    if (!gguf) throw std::runtime_error("cannot read " + path + " as a GGUF file; check the path and that the download is complete");
    return gguf;
}

/** Throws unless the file has speech.layout, which every file written for a release with layouts has. */
void require_layout_key(const gguf_context * gguf, const std::string & path, const std::string & remedy) {
    if (gguf_find_key(gguf, "speech.layout") < 0) {
        throw std::runtime_error(path + " has no speech.layout: it was written for a release of speech.cpp before " +
                                 kFirstLayoutRelease + ", which this release does not read; " + remedy);
    }
}

/** A key's type as the layouts write it: u32, or [i32] for an array. */
std::string type_text(gguf_type type, gguf_type element) {
    return type == GGUF_TYPE_ARRAY ? std::string("[") + gguf_type_name(element) + "]" : gguf_type_name(type);
}

std::string joined(const std::vector<std::string> & items, size_t at_most) {
    std::string out;
    for (size_t i = 0; i < items.size() && i < at_most; i++) out += (i ? ", " : "") + items[i];
    if (items.size() > at_most) out += " and " + std::to_string(items.size() - at_most) + " more";
    return out;
}

}  // namespace

std::string gguf_architecture(const std::string & path) {
    const Gguf gguf = open_gguf(path, nullptr);
    require_layout_key(gguf.get(), path, "convert it again or download it again");
    const int64_t id = gguf_find_key(gguf.get(), "general.architecture");
    if (id < 0 || gguf_get_kv_type(gguf.get(), id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(path + " names no general.architecture as a string");
    }
    return gguf_get_val_str(gguf.get(), id);
}

ModelFile::ModelFile(const std::string & path, ggml_backend_t backend, const Layout & layout)
    : path_(path), architecture_(layout.architecture), remedy_(layout.remedy) {
    ggml_context * ctx = nullptr;
    gguf_ = open_gguf(path, &ctx);
    ctx_.reset(ctx);
    require_layout_key(gguf_.get(), path_, remedy_);
    const int64_t id = gguf_find_key(gguf_.get(), "general.architecture");
    if (id < 0 || gguf_get_kv_type(gguf_.get(), id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(path_ + " names no general.architecture as a string; " + remedy_);
    }
    const std::string architecture = gguf_get_val_str(gguf_.get(), id);
    if (architecture != layout.architecture) {
        throw std::runtime_error(path_ + " is a file of " + architecture + ", where a file of " + layout.architecture + " is expected");
    }
    version_ = u32("speech.layout");
    if (version_ > layout.version) {
        throw std::runtime_error(path_ + " has layout " + std::to_string(version_) + " of " + architecture_ + ", which speech.cpp " +
                                 str("speech.requires") + " and later read; this is " + SPEECH_VERSION);
    }
    if (version_ != layout.version) {
        throw std::runtime_error(path_ + " has layout " + std::to_string(version_) + " of " + architecture_ +
                                 ", which no release of speech.cpp writes; " + remedy_);
    }
    check_tensors(layout);
    load(backend);
}

void ModelFile::check_tensors(const Layout & layout) const {
    const std::vector<std::string> wanted = layout.tensors(*this);
    const std::set<std::string> want(wanted.begin(), wanted.end());
    std::set<std::string> have;
    std::vector<std::string> missing, extra;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf_.get()); i++) {
        const std::string name = gguf_get_tensor_name(gguf_.get(), i);
        have.insert(name);
        if (!want.count(name)) extra.push_back(name);
    }
    for (const std::string & name : wanted) {
        if (!have.count(name)) missing.push_back(name);
    }
    if (missing.empty() && extra.empty()) return;
    std::string what;
    if (!missing.empty()) what += "it lacks " + joined(missing, 5);
    if (!extra.empty()) what += std::string(missing.empty() ? "" : ", and ") + "its keys call for no " + joined(extra, 5);
    throw std::runtime_error(path_ + " does not hold the tensors that " + layout_name() + " calls for: " + what + "; " + remedy_);
}

void ModelFile::load(ggml_backend_t backend) {
    buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(), backend));
    if (!buffer_) throw std::runtime_error("cannot allocate the weights of " + path_ + " on " + ggml_backend_name(backend));
    ggml_backend_buffer_set_usage(buffer_.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::unique_ptr<FILE, decltype(&std::fclose)> f(ggml_fopen(path_.c_str(), "rb"), &std::fclose);
    if (!f) throw std::runtime_error("cannot open " + path_);
    const uint64_t data_offset = gguf_get_data_offset(gguf_.get());
    std::vector<uint8_t> staging;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf_.get()); i++) {
        const char * name = gguf_get_tensor_name(gguf_.get(), i);
        ggml_tensor * t = ggml_get_tensor(ctx_.get(), name);
        const size_t size = ggml_nbytes(t);
        staging.resize(size);
        if (!seek(f.get(), data_offset + gguf_get_tensor_offset(gguf_.get(), i)) || std::fread(staging.data(), 1, size, f.get()) != size) {
            throw std::runtime_error(std::string("cannot read the tensor ") + name + " of " + path_ + "; the file is shorter than its header says");
        }
        ggml_backend_tensor_set(t, staging.data(), 0, size);
    }
}

std::string ModelFile::layout_name() const {
    return version_ ? "layout " + std::to_string(version_) + " of " + architecture_ : architecture_;
}

ggml_tensor * ModelFile::tensor(const std::string & name) const {
    ggml_tensor * t = ggml_get_tensor(ctx_.get(), name.c_str());
    if (!t) throw std::runtime_error("the tensor " + name + " is missing from " + path_);
    return t;
}

int64_t ModelFile::key_id(const std::string & key, gguf_type type, gguf_type element) const {
    const int64_t id = gguf_find_key(gguf_.get(), key.c_str());
    const std::string layout = layout_name();
    if (id < 0) throw std::runtime_error(path_ + " has no key " + key + ", which " + layout + " requires; " + remedy_);
    const gguf_type actual = gguf_get_kv_type(gguf_.get(), id);
    const gguf_type actual_element = actual == GGUF_TYPE_ARRAY ? gguf_get_arr_type(gguf_.get(), id) : GGUF_TYPE_COUNT;
    if (actual != type || actual_element != element) {
        throw std::runtime_error("the key " + key + " of " + path_ + " has the type " + type_text(actual, actual_element) + ", where " +
                                 layout + " gives it the type " + type_text(type, element) + "; " + remedy_);
    }
    return id;
}

uint32_t ModelFile::u32(const std::string & key) const {
    return gguf_get_val_u32(gguf_.get(), key_id(key, GGUF_TYPE_UINT32));
}

float ModelFile::f32(const std::string & key) const {
    return gguf_get_val_f32(gguf_.get(), key_id(key, GGUF_TYPE_FLOAT32));
}

bool ModelFile::boolean(const std::string & key) const {
    return gguf_get_val_bool(gguf_.get(), key_id(key, GGUF_TYPE_BOOL));
}

std::string ModelFile::str(const std::string & key) const {
    return gguf_get_val_str(gguf_.get(), key_id(key, GGUF_TYPE_STRING));
}

std::string ModelFile::one_of(const std::string & key, std::initializer_list<const char *> values) const {
    const std::string value = str(key);
    std::vector<std::string> known;
    for (const char * v : values) {
        if (value == v) return value;
        known.push_back(std::string("\"") + v + "\"");
    }
    throw std::runtime_error("the key " + key + " of " + path_ + " is \"" + value + "\", which " + layout_name() + " does not know; it knows " +
                             joined(known, known.size()) + "; " + remedy_);
}

std::vector<int32_t> ModelFile::i32_array(const std::string & key) const {
    const int64_t id = key_id(key, GGUF_TYPE_ARRAY, GGUF_TYPE_INT32);
    const int32_t * data = (const int32_t *) gguf_get_arr_data(gguf_.get(), id);
    return std::vector<int32_t>(data, data + gguf_get_arr_n(gguf_.get(), id));
}

std::vector<double> ModelFile::f64_array(const std::string & key) const {
    const int64_t id = key_id(key, GGUF_TYPE_ARRAY, GGUF_TYPE_FLOAT64);
    const double * data = (const double *) gguf_get_arr_data(gguf_.get(), id);
    return std::vector<double>(data, data + gguf_get_arr_n(gguf_.get(), id));
}

std::vector<std::string> ModelFile::str_array(const std::string & key) const {
    const int64_t id = key_id(key, GGUF_TYPE_ARRAY, GGUF_TYPE_STRING);
    std::vector<std::string> out(gguf_get_arr_n(gguf_.get(), id));
    for (size_t i = 0; i < out.size(); i++) out[i] = gguf_get_arr_str(gguf_.get(), id, i);
    return out;
}

void check_model_keys(const ModelFile & file, const char * task, const char * language_use) {
    for (const char * key : {"general.name", "general.license", "general.source.url"}) {
        if (file.str(key).empty()) throw std::runtime_error("the key " + std::string(key) + " of " + file.path() + " is empty; " + file.remedy());
    }
    const std::string file_task = file.one_of("speech.task", {"synthesis", "recognition"});
    if (file_task != task) {
        throw std::runtime_error(file.path() + " says its speech.task is " + file_task + ", where its family does " + task + "; " +
                                 file.remedy());
    }
    const std::string file_use = file.one_of("speech.language_use", {"steers", "checked"});
    if (file_use != language_use) {
        throw std::runtime_error(file.path() + " says its speech.language_use is " + file_use + ", where its family's is " + language_use +
                                 "; " + file.remedy());
    }
    if (file.u32("speech.sample_rate") == 0) throw std::runtime_error("speech.sample_rate of " + file.path() + " is 0; " + file.remedy());
    const std::vector<std::string> languages = file.str_array("speech.languages");
    if (languages.empty() || !std::is_sorted(languages.begin(), languages.end())) {
        throw std::runtime_error("speech.languages of " + file.path() + " is empty or not sorted; " + file.remedy());
    }
}

void add_numbered(std::vector<std::string> & names, const std::string & prefix, uint32_t count,
                  std::initializer_list<const char *> suffixes) {
    for (uint32_t i = 0; i < count; i++) {
        for (const char * suffix : suffixes) names.push_back(prefix + std::to_string(i) + (*suffix ? std::string(".") + suffix : ""));
    }
}
