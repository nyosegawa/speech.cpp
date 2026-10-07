#include "quantize.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

#include "error.h"
#include "gguf.h"
#include "log.h"

// speech quantize's writing of a model file in another weight type. The metadata is copied from the file's own bytes
// rather than written through gguf's writer: gguf writes a tensor's axes up to its last above 1 (ggml_n_dims()), where
// gguf-py, which wrote every released file, writes every axis of the numpy array, so that a kernel of width 1, [768,
// 768, 1] in a released file, would be written as [768, 768], and the file would differ from the released one in its
// bytes and its SHA-256, which callers pin.

namespace {

using File = std::unique_ptr<FILE, decltype(&std::fclose)>;
using Gguf = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

/** The bytes of a GGUF file's metadata read in order, every read checked against their end. */
class MetaReader {
public:
    MetaReader(const std::vector<uint8_t> & bytes, const std::string & path) : bytes_(bytes), path_(path) {}

    size_t at() const { return at_; }

    template <typename T>
    T read() {
        T value;
        std::memcpy(&value, take(sizeof(T)), sizeof(T));
        return value;
    }

    std::string string() {
        const uint64_t n = read<uint64_t>();
        return std::string((const char *) take(n), n);
    }

    void skip_value(gguf_type type) {
        if (type == GGUF_TYPE_STRING) {
            string();
        } else if (type == GGUF_TYPE_ARRAY) {
            const auto element = (gguf_type) read<uint32_t>();
            const uint64_t n = read<uint64_t>();
            if (element == GGUF_TYPE_ARRAY) throw Error(Fault::File, path_ + " holds arrays of arrays, which no layout has", "model_path");
            if (element == GGUF_TYPE_STRING) {
                for (uint64_t i = 0; i < n; i++) string();
            } else {
                const size_t size = gguf_scalar_size(element);
                if (n > (bytes_.size() - at_) / size) throw ended();
                take(n * size);
            }
        } else {
            take(gguf_scalar_size(type));
        }
    }

private:
    Error ended() const { return Error(Fault::File, "the metadata of " + path_ + " ends before its header says", "model_path"); }

    const uint8_t * take(uint64_t n) {
        if (n > bytes_.size() - at_) throw ended();
        const uint8_t * p = bytes_.data() + at_;
        at_ += n;
        return p;
    }

    const std::vector<uint8_t> & bytes_;
    const std::string & path_;
    size_t at_ = 0;
};

/** Where each key and each tensor's information lies in a file's metadata. */
struct MetaLayout {
    struct Entry {
        std::string key;
        size_t begin, end;
    };
    struct Info {
        std::string name;
        /** Where the information begins and where its type follows its name and shape. */
        size_t begin, type_at;
    };
    uint32_t version = 0;
    std::vector<Entry> entries;
    std::vector<Info> infos;
};

MetaLayout read_meta_layout(const std::vector<uint8_t> & bytes, const std::string & path) {
    MetaReader r(bytes, path);
    MetaLayout m;
    if (r.read<uint32_t>() != 0x46554747) throw Error(Fault::File, path + " is not a GGUF file", "model_path");
    m.version = r.read<uint32_t>();
    const uint64_t tensors = r.read<uint64_t>(), entries = r.read<uint64_t>();
    for (uint64_t i = 0; i < entries; i++) {
        const size_t begin = r.at();
        std::string key = r.string();
        r.skip_value((gguf_type) r.read<uint32_t>());
        m.entries.push_back({std::move(key), begin, r.at()});
    }
    for (uint64_t i = 0; i < tensors; i++) {
        const size_t begin = r.at();
        std::string name = r.string();
        const uint32_t axes = r.read<uint32_t>();
        for (uint32_t a = 0; a < axes; a++) r.read<uint64_t>();
        m.infos.push_back({std::move(name), begin, r.at()});
        r.read<uint32_t>();
        r.read<uint64_t>();
    }
    return m;
}

void put_bytes(std::vector<uint8_t> & out, const void * data, size_t n) {
    out.insert(out.end(), (const uint8_t *) data, (const uint8_t *) data + n);
}

template <typename T>
void put(std::vector<uint8_t> & out, T value) {
    put_bytes(out, &value, sizeof(T));
}

void put_string(std::vector<uint8_t> & out, const std::string & s) {
    put<uint64_t>(out, s.size());
    put_bytes(out, s.data(), s.size());
}

void put_u32_entry(std::vector<uint8_t> & out, const std::string & key, uint32_t value) {
    put_string(out, key);
    put<uint32_t>(out, GGUF_TYPE_UINT32);
    put<uint32_t>(out, value);
}

void pad(std::vector<uint8_t> & out, size_t alignment) {
    out.resize(GGML_PAD(out.size(), alignment), 0);
}

/**
 * The threads of a conversion, joined when it ends however it ends: a std::thread that is destroyed while it can still be
 * joined ends the process, which a failure to start a later thread would otherwise do to the ones started before it.
 */
struct Workers {
    std::vector<std::thread> threads;

    ~Workers() {
        for (std::thread & t : threads) t.join();
    }
};

/**
 * Converts `values`, rows of `row` values, to `type`, the rows spread over the machine's threads; ggml converts each row
 * on its own, so the bytes do not depend on the threads.
 */
std::vector<uint8_t> convert(const std::vector<float> & values, int64_t row, ggml_type type) {
    const int64_t rows = (int64_t) values.size() / row;
    std::vector<uint8_t> out(ggml_row_size(type, row) * rows);
    const int64_t threads = std::max<int64_t>(1, std::min<int64_t>(std::thread::hardware_concurrency(), rows));
    {
        Workers workers;
        workers.threads.reserve((size_t) threads);
        for (int64_t t = 0; t < threads; t++) {
            const int64_t first = rows * t / threads, last = rows * (t + 1) / threads;
            try {
                workers.threads.emplace_back([&, first, last] { ggml_quantize_chunk(type, values.data(), out.data(), first * row, last - first, row, nullptr); });
            } catch (const std::system_error & e) {
                throw Error(Fault::OutOfMemory, std::string("the host cannot start another thread to quantize with (") + e.what() +
                                                    "); close other programs and run it again");
            }
        }
    }
    return out;
}

/** Writes `bytes` to `f`, the file `path`. */
void write_bytes(FILE * f, const std::vector<uint8_t> & bytes, const std::string & path) {
    if (std::fwrite(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
        throw Error(Fault::Io, "cannot write " + path + "; check that its disk has room", "out_path");
    }
}

/**
 * Refuses a file whose tensors are not all F32: a quantized file is not quantized again, and a type made of F16 weights
 * would hold other bytes than the same type made of the F32 weights.
 */
void require_f32(const gguf_context * gguf, const ModelIdentity & identity, const std::string & in, const Layout & layout) {
    int64_t other = -1;
    bool quantized = false;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); i++) {
        const ggml_type t = gguf_get_tensor_type(gguf, i);
        if (t != GGML_TYPE_F32 && other < 0) other = i;
        quantized |= ggml_is_quantized(t);
    }
    if (other < 0) return;
    const std::string remedy = "; quantize the model's F32 file, which reference/" + std::string(layout.architecture) + "/convert.py writes";
    if (quantized) {
        throw Error(Fault::InvalidArgument, in + " holds " + identity.weight_type + " weights, which speech quantize does not quantize again" + remedy,
                    "model_path");
    }
    throw Error(Fault::InvalidArgument,
                in + " holds " + identity.weight_type + " weights (its tensor " + gguf_get_tensor_name(gguf, other) + " is " +
                    tensor_type_text(gguf_get_tensor_type(gguf, other)) +
                    "), and speech quantize makes every type from F32 weights alone, so that the files of a type made of one model hold the "
                    "same bytes" +
                    remedy,
                "model_path");
}

/** Writes `out` as quantize_model_file() says, and sets `created` once it has created the file. */
void write_model_file(const std::string & in, const std::string & out, const WeightType & type, const Layout & layout, bool & created) {
    const auto model = naming("model_path", [&] { return std::make_unique<const ModelFile>(in, layout); });
    const ModelIdentity identity = naming("model_path", [&] { return read_identity(*model); });
    ggml_context * raw_ctx = nullptr;
    route_ggml_log();
    const Gguf gguf(gguf_init_from_file(in.c_str(), {/*no_alloc =*/true, &raw_ctx}), gguf_free);
    const std::unique_ptr<ggml_context, decltype(&ggml_free)> meta_ctx(raw_ctx, ggml_free);
    if (!gguf) throw Error(Fault::File, "cannot read " + in + " as a GGUF file", "model_path");
    require_f32(gguf.get(), identity, in, layout);
    if (gguf_find_key(gguf.get(), GGUF_KEY_GENERAL_ALIGNMENT) >= 0) {
        throw Error(Fault::File, in + " sets " GGUF_KEY_GENERAL_ALIGNMENT ", which no converter of speech.cpp writes and speech quantize does not carry over",
                    "model_path");
    }
    std::unordered_map<std::string, TensorSpec> specs;
    for (TensorSpec & spec : layout.tensors(*model)) specs.emplace(spec.name, spec);
    const int64_t n = gguf_get_n_tensors(gguf.get()), keys = gguf_get_n_kv(gguf.get());
    std::vector<ggml_type> types(n);
    // The file names the latest of the releases that first read its layout, its general.file_type and each tensor's
    // type in its family.
    std::string release = naming("model_path", [&] { return model->str("speech.requires"); });
    const auto at_least = [&](const std::string & r) {
        if (release_after(r, release)) release = r;
    };
    at_least(type.release);
    for (int64_t i = 0; i < n; i++) {
        const TensorSpec & spec = specs.at(gguf_get_tensor_name(gguf.get(), i));
        types[i] = spec.type_in(type.type);
        at_least(spec.first_release(types[i]));
    }

    std::vector<uint8_t> bytes(gguf_get_data_offset(gguf.get()));
    const File source(ggml_fopen(in.c_str(), "rb"), &std::fclose);
    if (!source || std::fread(bytes.data(), 1, bytes.size(), source.get()) != bytes.size()) {
        throw Error(Fault::Io, "cannot read " + in + "; check that the file can be read", "model_path");
    }
    const MetaLayout meta_of_in = read_meta_layout(bytes, in);
    if ((int64_t) meta_of_in.entries.size() != keys || (int64_t) meta_of_in.infos.size() != n) {
        throw std::logic_error("the metadata of " + in + " has other keys or tensors than gguf read");
    }

    const bool quantized = ggml_is_quantized(type.type);
    const bool had_version = gguf_find_key(gguf.get(), "general.quantization_version") >= 0;
    std::vector<uint8_t> meta;
    put<uint32_t>(meta, 0x46554747);
    put<uint32_t>(meta, meta_of_in.version);
    put<uint64_t>(meta, (uint64_t) n);
    put<uint64_t>(meta, (uint64_t) (keys - (had_version ? 1 : 0) + (quantized ? 1 : 0)));
    for (int64_t i = 0; i < keys; i++) {
        const MetaLayout::Entry & e = meta_of_in.entries[i];
        if (e.key != gguf_get_key(gguf.get(), i)) throw std::logic_error("the keys of " + in + " are not in the order gguf read them");
        if (e.key == "general.file_type") {
            put_u32_entry(meta, e.key, type.file_type);
            if (quantized) put_u32_entry(meta, "general.quantization_version", GGML_QNT_VERSION);
        } else if (e.key == "speech.requires") {
            put_string(meta, e.key);
            put<uint32_t>(meta, GGUF_TYPE_STRING);
            put_string(meta, release);
        } else if (e.key != "general.quantization_version") {
            put_bytes(meta, bytes.data() + e.begin, e.end - e.begin);
        }
    }
    const size_t alignment = gguf_get_alignment(gguf.get());
    uint64_t offset = 0;
    for (int64_t i = 0; i < n; i++) {
        const MetaLayout::Info & info = meta_of_in.infos[i];
        const ggml_tensor * t = ggml_get_tensor(meta_ctx.get(), info.name.c_str());
        if (!t || info.name != gguf_get_tensor_name(gguf.get(), i)) throw std::logic_error("the tensors of " + in + " are not in the order gguf read them");
        put_bytes(meta, bytes.data() + info.begin, info.type_at - info.begin);
        put<uint32_t>(meta, types[i]);
        put<uint64_t>(meta, offset);
        offset += GGML_PAD(ggml_row_size(types[i], t->ne[0]) * (ggml_nelements(t) / t->ne[0]), alignment);
    }
    pad(meta, alignment);

    const File target(ggml_fopen(out.c_str(), "wb"), &std::fclose);
    if (!target) throw Error(Fault::Io, "cannot create " + out + "; check that its folder exists and is writable", "out_path");
    created = true;
    write_bytes(target.get(), meta, out);
    std::vector<float> values;
    for (int64_t i = 0; i < n; i++) {
        const ggml_tensor * t = ggml_get_tensor(meta_ctx.get(), gguf_get_tensor_name(gguf.get(), i));
        values.resize(ggml_nelements(t));
        const size_t size = values.size() * sizeof(float);
        if (!seek_file(source.get(), gguf_get_data_offset(gguf.get()) + gguf_get_tensor_offset(gguf.get(), i)) ||
            std::fread(values.data(), 1, size, source.get()) != size) {
            throw Error(Fault::File, "cannot read the tensor " + std::string(t->name) + " of " + in + "; the file is shorter than its header says",
                        "model_path");
        }
        // ggml's quantizers assert that a value is finite where assertions are on and write garbage where they are off.
        const auto bad = std::find_if(values.begin(), values.end(), [](float v) { return !std::isfinite(v); });
        if (bad != values.end()) {
            throw Error(Fault::File,
                        "the tensor " + std::string(t->name) + " of " + in + " holds " + (std::isnan(*bad) ? "NaN" : *bad > 0 ? "infinity" : "-infinity") +
                            " at value " + std::to_string(bad - values.begin()) + ", which no weight of a model is; " + model->remedy(),
                        "model_path");
        }
        std::vector<uint8_t> data = convert(values, t->ne[0], types[i]);
        pad(data, alignment);
        write_bytes(target.get(), data, out);
    }
    if (std::fflush(target.get()) != 0) throw Error(Fault::Io, "cannot write " + out + "; check that its disk has room", "out_path");
}

}  // namespace

void quantize_model_file(const std::string & in, const std::string & out, const WeightType & type, const Layout & layout) {
    if (type.type == GGML_TYPE_F32) throw std::logic_error("quantize_model_file() was asked for F32, the type it writes from");
    std::error_code error;
    if (std::filesystem::equivalent(std::filesystem::u8path(in), std::filesystem::u8path(out), error)) {
        throw Error(Fault::InvalidArgument, out + " is the model file itself; give speech quantize another path to write", "out_path");
    }
    bool created = false;
    try {
        write_model_file(in, out, type, layout, created);
        try {
            read_identity(ModelFile(out, layout));
        } catch (const Error & e) {
            throw std::logic_error("speech quantize wrote " + out + ", which its own reader refuses: " + e.what());
        }
    } catch (...) {
        if (created) std::filesystem::remove(std::filesystem::u8path(out), error);
        throw;
    }
}
