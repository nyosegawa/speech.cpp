// Checks what quantize_model_file(), which speech quantize runs, writes of a file and refuses of its weights. A model
// file of a layout of the check's own, which speech.requires says 0.7.0 reads, holds a matrix that a quantized file
// quantizes, which its storage says releases from 0.7.0 read in F32 and F16 and from 0.8.0 in Q8_0 and the K-quants, as
// FastConformer's says, and a vector it keeps in F32. With finite weights it is written in every weight type speech
// quantize writes, naming in speech.requires 0.7.0 in F16 and 0.8.0 in the others; with a NaN, an infinity or a negative
// infinity in either tensor it is refused, as a file error naming model_path with nothing written: ggml's quantizers
// abort on such a value where assertions are on and write garbage where they are off.
//
// usage: quantize-check <work dir>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "error.h"
#include "gguf.h"
#include "model-file.h"
#include "quantize.h"

namespace {

constexpr int64_t kRow = 256, kRows = 16;

const Storage kMatrix = quantized_storage({{GGML_TYPE_F32, "0.7.0"},
                                           {GGML_TYPE_F16, "0.7.0"},
                                           {GGML_TYPE_Q8_0, "0.8.0"},
                                           {GGML_TYPE_Q6_K, "0.8.0"},
                                           {GGML_TYPE_Q5_K, "0.8.0"},
                                           {GGML_TYPE_Q4_K, "0.8.0"}});
const Storage kVector = float32_storage();

const Layout kLayout = {"quantize-check", 1, "make it again with quantize-check", [](const ModelFile &) {
                            return std::vector<TensorSpec>{{"matrix", {kRow, kRows}, kMatrix}, {"vector", {kRow}, kVector}};
                        }};

/** Writes the check's F32 model file at `path`, with `bad` at value `at` of `tensor` when `tensor` is not empty. */
void write_model(const std::string & path, const std::string & tensor, int64_t at, float bad) {
    ggml_init_params params = {ggml_tensor_overhead() * 2 + (size_t) (kRow * kRows + kRow) * sizeof(float) + 1024, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    ggml_tensor * matrix = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kRow, kRows);
    ggml_tensor * vector = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, kRow);
    ggml_set_name(matrix, "matrix");
    ggml_set_name(vector, "vector");
    for (ggml_tensor * t : {matrix, vector}) {
        float * v = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); i++) v[i] = std::sin(0.37f * (float) i) * 0.05f;
        if (tensor == ggml_get_name(t)) v[at] = bad;
    }
    std::unique_ptr<gguf_context, decltype(&gguf_free)> g(gguf_init_empty(), gguf_free);
    gguf_set_val_str(g.get(), "general.architecture", "quantize-check");
    gguf_set_val_str(g.get(), "general.name", "quantize-check");
    gguf_set_val_str(g.get(), "general.organization", "speech.cpp");
    gguf_set_val_str(g.get(), "general.basename", "quantize-check");
    gguf_set_val_str(g.get(), "general.size_label", "1.3K");
    gguf_set_val_str(g.get(), "general.license", "MIT");
    gguf_set_val_str(g.get(), "general.source.url", "https://huggingface.co/speech-cpp/quantize-check/tree/0");
    gguf_set_val_str(g.get(), "general.source.repo_url", "https://huggingface.co/speech-cpp/quantize-check");
    gguf_set_val_u32(g.get(), "general.file_type", 0);
    gguf_set_val_u32(g.get(), "speech.layout", 1);
    gguf_set_val_str(g.get(), "speech.requires", "0.7.0");
    gguf_add_tensor(g.get(), matrix);
    gguf_add_tensor(g.get(), vector);
    if (!gguf_write_to_file(g.get(), path.c_str(), false)) throw std::runtime_error("cannot write " + path);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <work dir>\n", argv[0]);
        return 2;
    }
    const std::filesystem::path work = std::filesystem::u8path(argv[1]);
    std::filesystem::create_directories(work);
    const std::string in = (work / "quantize-check-F32.gguf").u8string(), out = (work / "quantize-check-out.gguf").u8string();
    int failed = 0;
    try {
        write_model(in, "", 0, 0);
        for (const WeightType & type : weight_types()) {
            if (type.type == GGML_TYPE_F32) continue;
            quantize_model_file(in, out, type, kLayout);
            const ModelFile file(out, kLayout);
            const std::string written = read_identity(file).weight_type, release = file.str("speech.requires");
            const std::string wanted = type.type == GGML_TYPE_F16 ? "0.7.0" : "0.8.0";
            std::printf("finite weights in %s: written, %s weights, speech.requires %s\n", tensor_type_text(type.type).c_str(), written.c_str(),
                        release.c_str());
            if (written != tensor_type_text(type.type) || release != wanted) {
                std::printf("FAIL: %s, where %s weights and speech.requires %s are wanted\n", tensor_type_text(type.type).c_str(),
                            tensor_type_text(type.type).c_str(), wanted.c_str());
                failed++;
            }
            std::filesystem::remove(std::filesystem::u8path(out));
        }
        const float infinity = std::numeric_limits<float>::infinity();
        for (const auto & [tensor, at] : {std::make_pair("matrix", kRow + 44), std::make_pair("vector", (int64_t) 100)}) {
            for (const float bad : {std::numeric_limits<float>::quiet_NaN(), infinity, -infinity}) {
                write_model(in, tensor, at, bad);
                for (const WeightType & type : weight_types()) {
                    if (type.type == GGML_TYPE_F32) continue;
                    std::string outcome;
                    try {
                        quantize_model_file(in, out, type, kLayout);
                        outcome = "written";
                    } catch (const Error & e) {
                        outcome = e.fault() == Fault::File && e.input() == "model_path" ? "" : std::string("refused otherwise: ") + e.what();
                        if (outcome.empty()) {
                            std::printf("%g in the %s, %s: refused naming model_path: %s\n", bad, tensor, tensor_type_text(type.type).c_str(), e.what());
                        }
                    }
                    if (std::filesystem::exists(std::filesystem::u8path(out))) {
                        outcome += outcome.empty() ? "" : ", and ";
                        outcome += "a file is left where the output goes";
                        std::filesystem::remove(std::filesystem::u8path(out));
                    }
                    if (!outcome.empty()) {
                        std::printf("FAIL: %g in the %s, %s: %s\n", bad, tensor, tensor_type_text(type.type).c_str(), outcome.c_str());
                        failed++;
                    }
                }
            }
        }
        std::filesystem::remove(std::filesystem::u8path(in));
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    if (failed) {
        std::printf("FAIL: %d cases\n", failed);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
