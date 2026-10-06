#include "model-file.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>
#include <stdexcept>

#include "error.h"
#include "json.h"
#include "log.h"

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

Error file_error(const std::string & message) {
    return Error(Fault::File, message);
}

/**
 * The file's metadata. gguf_init_from_file() answers a file it cannot open and one that is not GGUF alike, so the
 * file is opened first to tell the two apart.
 */
Gguf open_gguf(const std::string & path, ggml_context ** ctx) {
    route_ggml_log();
    std::unique_ptr<FILE, decltype(&std::fclose)> f(ggml_fopen(path.c_str(), "rb"), &std::fclose);
    if (!f) throw Error(Fault::Io, "cannot open " + path + "; check the path and that the file can be read");
    f.reset();
    gguf_init_params params = {/*no_alloc =*/true, /*ctx =*/ctx};
    Gguf gguf(gguf_init_from_file(path.c_str(), params), gguf_free);
    if (!gguf) throw file_error("cannot read " + path + " as a GGUF file; check that it is a model file and that the download is complete");
    return gguf;
}

/** Throws unless the file has speech.layout, which every file written for a release with layouts has. */
void require_layout_key(const gguf_context * gguf, const std::string & path, const std::string & remedy) {
    if (gguf_find_key(gguf, "speech.layout") < 0) {
        throw file_error(path + " has no speech.layout: it was written for a release of speech.cpp before " + kFirstLayoutRelease +
                         ", which this release does not read; " + remedy);
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

/** A shape as README.md writes it, "[2048, 1024]", without the axes of 1 past the last other one. */
std::string shape_text(const int64_t * ne) {
    int axes = GGML_MAX_DIMS;
    while (axes > 1 && ne[axes - 1] == 1) axes--;
    std::string out = "[";
    for (int i = 0; i < axes; i++) out += (i ? ", " : "") + std::to_string(ne[i]);
    return out + "]";
}

/** A ggml type as README.md writes it: F32, F16, Q8_0. */
std::string tensor_type_text(ggml_type type) {
    std::string name = ggml_type_name(type);
    for (char & c : name) c = (char) std::toupper((unsigned char) c);
    return name;
}

/** "F32", or "Q8_0, F16 or F32". */
std::string types_text(const std::vector<ggml_type> & types) {
    std::string out;
    for (size_t i = 0; i < types.size(); i++) out += (i == 0 ? "" : i + 1 == types.size() ? " or " : ", ") + tensor_type_text(types[i]);
    return out;
}

/** The bytes of one value of a type that is neither a string nor an array. */
size_t scalar_size(gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default: throw std::logic_error("scalar_size() was given a string or an array");
    }
}

/** One value of type `type` at `data` as JSON. */
std::string scalar_json(gguf_type type, const void * data) {
    switch (type) {
        case GGUF_TYPE_UINT8: return std::to_string(*(const uint8_t *) data);
        case GGUF_TYPE_INT8: return std::to_string(*(const int8_t *) data);
        case GGUF_TYPE_UINT16: return std::to_string(*(const uint16_t *) data);
        case GGUF_TYPE_INT16: return std::to_string(*(const int16_t *) data);
        case GGUF_TYPE_UINT32: return std::to_string(*(const uint32_t *) data);
        case GGUF_TYPE_INT32: return std::to_string(*(const int32_t *) data);
        case GGUF_TYPE_UINT64: return std::to_string(*(const uint64_t *) data);
        case GGUF_TYPE_INT64: return std::to_string(*(const int64_t *) data);
        case GGUF_TYPE_FLOAT32: return json_number(*(const float *) data);
        case GGUF_TYPE_FLOAT64: return json_number(*(const double *) data);
        case GGUF_TYPE_BOOL: return *(const int8_t *) data ? "true" : "false";
        default: throw std::logic_error("scalar_json() was given a string or an array");
    }
}

}  // namespace

std::string gguf_architecture(const std::string & path) {
    const Gguf gguf = open_gguf(path, nullptr);
    require_layout_key(gguf.get(), path, "convert it again or download it again");
    const int64_t id = gguf_find_key(gguf.get(), "general.architecture");
    if (id < 0 || gguf_get_kv_type(gguf.get(), id) != GGUF_TYPE_STRING) {
        throw file_error(path + " names no general.architecture as a string");
    }
    return gguf_get_val_str(gguf.get(), id);
}

ModelFile::ModelFile(const std::string & path, ggml_backend_t backend, const Layout & layout,
                     const std::function<bool(const std::string & name)> & keep)
    : path_(path), architecture_(layout.architecture), remedy_(layout.remedy) {
    open(layout);
    load(backend, keep);
}

ModelFile::ModelFile(const std::string & path, const Layout & layout)
    : path_(path), architecture_(layout.architecture), remedy_(layout.remedy) {
    open(layout);
}

void ModelFile::open(const Layout & layout) {
    ggml_context * ctx = nullptr;
    gguf_ = open_gguf(path_, &ctx);
    ctx_.reset(ctx);
    require_layout_key(gguf_.get(), path_, remedy_);
    const int64_t id = gguf_find_key(gguf_.get(), "general.architecture");
    if (id < 0 || gguf_get_kv_type(gguf_.get(), id) != GGUF_TYPE_STRING) {
        throw file_error(path_ + " names no general.architecture as a string; " + remedy_);
    }
    const std::string architecture = gguf_get_val_str(gguf_.get(), id);
    if (architecture != layout.architecture) {
        throw file_error(path_ + " is a file of " + architecture + ", where a file of " + layout.architecture + " is expected");
    }
    version_ = u32("speech.layout");
    if (version_ > layout.version) {
        throw file_error(path_ + " has layout " + std::to_string(version_) + " of " + architecture_ + ", which speech.cpp " +
                         str("speech.requires") + " and later read; this is " + SPEECH_VERSION);
    }
    if (version_ != layout.version) {
        throw file_error(path_ + " has layout " + std::to_string(version_) + " of " + architecture_ + ", which no release of speech.cpp writes; " +
                         remedy_);
    }
    check_tensors(layout);
}

void ModelFile::check_tensors(const Layout & layout) const {
    const std::vector<TensorSpec> wanted = layout.tensors(*this);
    std::set<std::string> want;
    for (const TensorSpec & t : wanted) want.insert(t.name);
    std::set<std::string> have;
    std::vector<std::string> missing, extra;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf_.get()); i++) {
        const std::string name = gguf_get_tensor_name(gguf_.get(), i);
        have.insert(name);
        if (!want.count(name)) extra.push_back(name);
    }
    for (const TensorSpec & t : wanted) {
        if (!have.count(t.name)) missing.push_back(t.name);
    }
    if (!missing.empty() || !extra.empty()) {
        std::string what;
        if (!missing.empty()) what += "it lacks " + joined(missing, 5);
        if (!extra.empty()) what += std::string(missing.empty() ? "" : ", and ") + "its keys call for no " + joined(extra, 5);
        throw file_error(path_ + " does not hold the tensors that " + layout_name() + " calls for: " + what + "; " + remedy_);
    }
    for (const TensorSpec & spec : wanted) {
        const ggml_tensor * t = ggml_get_tensor(ctx_.get(), spec.name.c_str());
        if (!std::equal(spec.shape.ne, spec.shape.ne + GGML_MAX_DIMS, t->ne)) {
            throw file_error("the tensor " + spec.name + " of " + path_ + " has the shape " + shape_text(t->ne) + ", where " + layout_name() +
                             " gives it the shape " + shape_text(spec.shape.ne) + "; " + remedy_);
        }
        if (std::find(spec.types.begin(), spec.types.end(), t->type) == spec.types.end()) {
            throw file_error("the tensor " + spec.name + " of " + path_ + " has the type " + tensor_type_text(t->type) + ", where " + layout_name() +
                             " stores it in " + types_text(spec.types) + "; " + remedy_);
        }
    }
}

void ModelFile::load(ggml_backend_t backend, const std::function<bool(const std::string & name)> & keep) {
    loaded_ = ctx_.get();
    if (keep) {
        const int64_t n = gguf_get_n_tensors(gguf_.get());
        ggml_init_params params = {ggml_tensor_overhead() * (size_t) std::max<int64_t>(n, 1), nullptr, true};
        kept_.reset(ggml_init(params));
        if (!kept_) throw Error(Fault::OutOfMemory, "cannot create a ggml context for the tensors of " + path_);
        for (int64_t i = 0; i < n; i++) {
            const char * name = gguf_get_tensor_name(gguf_.get(), i);
            if (keep(name)) ggml_set_name(ggml_dup_tensor(kept_.get(), ggml_get_tensor(ctx_.get(), name)), name);
        }
        loaded_ = kept_.get();
    }
    buffer_.reset(ggml_backend_alloc_ctx_tensors(loaded_, backend));
    if (!buffer_) throw Error(Fault::OutOfMemory, "cannot allocate the weights of " + path_ + " on " + ggml_backend_name(backend));
    ggml_backend_buffer_set_usage(buffer_.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::unique_ptr<FILE, decltype(&std::fclose)> f(ggml_fopen(path_.c_str(), "rb"), &std::fclose);
    if (!f) throw Error(Fault::Io, "cannot open " + path_);
    const uint64_t data_offset = gguf_get_data_offset(gguf_.get());
    std::vector<uint8_t> staging;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf_.get()); i++) {
        const char * name = gguf_get_tensor_name(gguf_.get(), i);
        ggml_tensor * t = ggml_get_tensor(loaded_, name);
        if (!t) continue;
        const size_t size = ggml_nbytes(t);
        staging.resize(size);
        if (!seek(f.get(), data_offset + gguf_get_tensor_offset(gguf_.get(), i)) || std::fread(staging.data(), 1, size, f.get()) != size) {
            throw file_error(std::string("cannot read the tensor ") + name + " of " + path_ + "; the file is shorter than its header says");
        }
        ggml_backend_tensor_set(t, staging.data(), 0, size);
    }
}

std::string ModelFile::layout_name() const {
    return version_ ? "layout " + std::to_string(version_) + " of " + architecture_ : architecture_;
}

ggml_tensor * ModelFile::tensor(const std::string & name) const {
    ggml_tensor * t = loaded_ ? ggml_get_tensor(loaded_, name.c_str()) : nullptr;
    if (!t) throw std::logic_error("the tensor " + name + " of " + path_ + " is not loaded");
    return t;
}

int64_t ModelFile::key_id(const std::string & key, gguf_type type, gguf_type element) const {
    const int64_t id = gguf_find_key(gguf_.get(), key.c_str());
    const std::string layout = layout_name();
    if (id < 0) throw file_error(path_ + " has no key " + key + ", which " + layout + " requires; " + remedy_);
    const gguf_type actual = gguf_get_kv_type(gguf_.get(), id);
    const gguf_type actual_element = actual == GGUF_TYPE_ARRAY ? gguf_get_arr_type(gguf_.get(), id) : GGUF_TYPE_COUNT;
    if (actual != type || actual_element != element) {
        throw file_error("the key " + key + " of " + path_ + " has the type " + type_text(actual, actual_element) + ", where " + layout +
                         " gives it the type " + type_text(type, element) + "; " + remedy_);
    }
    return id;
}

uint32_t ModelFile::u32(const std::string & key) const {
    return gguf_get_val_u32(gguf_.get(), key_id(key, GGUF_TYPE_UINT32));
}

int ModelFile::size(const std::string & key) const {
    const uint32_t value = u32(key);
    if (value == 0 || value > (uint32_t) INT_MAX) {
        throw file_error("the key " + key + " of " + path_ + " is " + std::to_string(value) + ", where " + layout_name() + " takes a size from 1 to " +
                         std::to_string(INT_MAX) + "; " + remedy_);
    }
    return (int) value;
}

int ModelFile::count(const std::string & key) const {
    const uint32_t value = u32(key);
    if (value == 0 || value > tensor_count()) {
        throw file_error("the key " + key + " of " + path_ + " is " + std::to_string(value) + ", where " + layout_name() +
                         " takes a number of layers or blocks from 1 to the " + std::to_string(tensor_count()) + " tensors the file holds; " + remedy_);
    }
    return (int) value;
}

int64_t ModelFile::width(const std::string & name, int axis) const {
    const ggml_tensor * t = ggml_get_tensor(ctx_.get(), name.c_str());
    if (!t) throw file_error(path_ + " does not hold the tensor " + name + ", which " + layout_name() + " calls for; " + remedy_);
    if (t->ne[axis] == 0 || t->ne[axis] > INT_MAX) {
        throw file_error("the tensor " + name + " of " + path_ + " has the shape " + shape_text(t->ne) + ", where " + layout_name() +
                         " takes a width from 1 to " + std::to_string(INT_MAX) + " on its axis " + std::to_string(axis) + "; " + remedy_);
    }
    return t->ne[axis];
}

int64_t ModelFile::tensor_count() const {
    return gguf_get_n_tensors(gguf_.get());
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
    throw file_error("the key " + key + " of " + path_ + " is \"" + value + "\", which " + layout_name() + " does not know; it knows " +
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

uint64_t ModelFile::file_bytes() const {
    std::error_code error;
    const uint64_t size = std::filesystem::file_size(std::filesystem::u8path(path_), error);
    if (error) throw Error(Fault::Io, "cannot read the size of " + path_ + ": " + error.message());
    return size;
}

uint64_t ModelFile::weight_bytes() const {
    uint64_t bytes = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf_.get()); i++) bytes += gguf_get_tensor_size(gguf_.get(), i);
    return bytes;
}

size_t ModelFile::meta_count() const {
    return (size_t) gguf_get_n_kv(gguf_.get());
}

std::string ModelFile::meta_key(size_t index) const {
    return gguf_get_key(gguf_.get(), (int64_t) index);
}

std::string ModelFile::meta_json(size_t index) const {
    const gguf_context * g = gguf_.get();
    const int64_t id = (int64_t) index;
    const gguf_type type = gguf_get_kv_type(g, id);
    if (type == GGUF_TYPE_STRING) return json_string(gguf_get_val_str(g, id));
    if (type != GGUF_TYPE_ARRAY) return scalar_json(type, gguf_get_val_data(g, id));
    const gguf_type element = gguf_get_arr_type(g, id);
    const size_t n = gguf_get_arr_n(g, id);
    std::string out = "[";
    if (element == GGUF_TYPE_STRING) {
        for (size_t i = 0; i < n; i++) out += (i ? "," : "") + json_string(gguf_get_arr_str(g, id, i));
    } else if (element == GGUF_TYPE_ARRAY) {
        throw file_error("the key " + meta_key(index) + " of " + path_ + " holds arrays of arrays, which no layout has; " + remedy_);
    } else {
        const auto * data = (const uint8_t *) gguf_get_arr_data(g, id);
        const size_t width = scalar_size(element);
        for (size_t i = 0; i < n; i++) out += (i ? "," : "") + scalar_json(element, data + i * width);
    }
    return out + "]";
}

void check_model_keys(const ModelFile & file, const char * task, const char * language_use) {
    for (const char * key : {"general.name", "general.license", "general.source.url"}) {
        if (file.str(key).empty()) throw file_error("the key " + std::string(key) + " of " + file.path() + " is empty; " + file.remedy());
    }
    const std::string file_task = file.one_of("speech.task", {"synthesis", "recognition"});
    if (file_task != task) {
        throw file_error(file.path() + " says its speech.task is " + file_task + ", where its family does " + task + "; " + file.remedy());
    }
    const std::string file_use = file.one_of("speech.language_use", {"steers", "checked"});
    if (file_use != language_use) {
        throw file_error(file.path() + " says its speech.language_use is " + file_use + ", where its family's is " + language_use + "; " +
                         file.remedy());
    }
    file.size("speech.sample_rate");
    const std::vector<std::string> languages = file.str_array("speech.languages");
    if (languages.empty() || !std::is_sorted(languages.begin(), languages.end())) {
        throw file_error("speech.languages of " + file.path() + " is empty or not sorted; " + file.remedy());
    }
}

Shape::Shape(std::initializer_list<int64_t> axes) {
    if (axes.size() > GGML_MAX_DIMS) throw std::logic_error("a shape has more axes than ggml's tensors");
    std::copy(axes.begin(), axes.end(), ne);
}

void add_block(std::vector<TensorSpec> & tensors, const std::string & prefix, std::initializer_list<TensorSpec> parts) {
    for (const TensorSpec & part : parts) tensors.push_back({prefix + part.name, part.shape, part.types});
}

void add_numbered(std::vector<TensorSpec> & tensors, const std::string & prefix, int count, std::initializer_list<TensorSpec> parts) {
    for (int i = 0; i < count; i++) {
        for (const TensorSpec & part : parts) {
            tensors.push_back({prefix + std::to_string(i) + (part.name.empty() ? "" : "." + part.name), part.shape, part.types});
        }
    }
}
