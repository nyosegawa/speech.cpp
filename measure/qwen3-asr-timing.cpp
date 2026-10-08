// Times Qwen3-ASR on the audio of dumps of reference/qwen3-asr/dump.py, with the language left to the model and forced
// as each dump's forced request forces it: for each, the median over the runs of the whole recognition, the frontend,
// the encoder, the prefill (the decoder's input and its prompt) and the decoding, and of the decoding's steps per
// second, a step being a token fed back to the decoder, as llama.cpp's llama-server counts them. One recognition of the
// first dump goes first, untimed, so that the times leave out the compilation of a GPU's kernels. It prints one JSON
// object per line; checks/llama-server-timing.py prints the same of llama.cpp's server.
//
// usage: qwen3-asr-timing <model.gguf> <gpu|cpu|device name> <runs> <dump dir>...

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "json-reader.h"
#include "json.h"
#include "npy.h"
#include "qwen3-asr/recognizer.h"
#include "reference-dumps.h"

using namespace qwen3_asr;

namespace {

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

/** The language that the forced request of the dump `dir` forces, as an index of the model's languages. */
size_t forced_language(const std::filesystem::path & dir, const ModelFile & m) {
    const JsonValue meta = parse_json(dump_text(dir / "forced" / "meta.json"));
    const std::string name = meta.member("language")->text;
    const std::vector<std::string> names = m.str_array("qwen3-asr.language_names");
    const auto it = std::find(names.begin(), names.end(), name);
    if (it == names.end()) throw std::runtime_error("the dump forces " + name + ", which the model does not name");
    return (size_t) (it - names.begin());
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 5) {
        std::fprintf(stderr, "usage: %s <model.gguf> <gpu|cpu|device name> <runs> <dump dir>...\n", args[0].c_str());
        return 2;
    }
    try {
        ggml_backend_t backend = init_backend(args[2]);
        const int runs = std::stoi(args[3]);
        {
            Recognizer recognizer(args[1], backend);
            const auto keep_going = [](double) { return true; };
            recognizer.recognize(read_npy((std::filesystem::u8path(args[4]) / "audio.npy").u8string()).f32, {}, keep_going);
            for (size_t a = 4; a < args.size(); a++) {
                const std::filesystem::path dir = std::filesystem::u8path(args[a]);
                const std::vector<float> audio = read_npy((dir / "audio.npy").u8string()).f32;
                RecognitionRequest forced;
                forced.language = forced_language(dir, recognizer.model());
                for (const RecognitionRequest & request : {RecognitionRequest{}, forced}) {
                    std::vector<double> total, frontend, encoder, prefill, decode, steps_per_second;
                    PartReport report;
                    std::string text;
                    for (int r = 0; r < runs; r++) {
                        std::vector<PartReport> parts;
                        const auto start = std::chrono::steady_clock::now();
                        text = recognizer.recognize(audio, request, keep_going, &parts).text;
                        total.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
                        report = parts.at(0);
                        frontend.push_back(report.frontend);
                        encoder.push_back(report.encoder);
                        prefill.push_back(report.prefill);
                        decode.push_back(report.decode);
                        steps_per_second.push_back((double) (report.ids.size() - 1) / report.decode);
                    }
                    std::printf("{\"input\":%s,\"seconds\":%.2f,\"forced\":%s,\"total\":%.3f,\"frontend\":%.3f,\"encoder\":%.3f,"
                                "\"prefill\":%.3f,\"prompt_rows\":%lld,\"decode\":%.3f,\"tokens\":%zu,\"steps_per_second\":%.1f,"
                                "\"text\":%s}\n",
                                json_string(dir.filename().u8string()).c_str(), (double) audio.size() / recognizer.sample_rate(),
                                request.language ? json_string(recognizer.languages()[*request.language]).c_str() : "null", median(total),
                                median(frontend), median(encoder), median(prefill), (long long) report.prompt_rows, median(decode),
                                report.ids.size(), median(steps_per_second), json_string(text).c_str());
                    std::fflush(stdout);
                }
            }
        }
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
