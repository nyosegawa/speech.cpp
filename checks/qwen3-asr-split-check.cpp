// Checks where Qwen3-ASR cuts audio too long for the model against qwen-asr's split_audio_into_chunks(): on the
// synthetic cases of reference/qwen3-asr/split_cases.py, under <reference out dir>/split/, and on the inputs of
// reference/qwen3-asr/dump.py that it split, under <reference out dir>/<the model's general.name>/. It then recognizes
// each such input with the model on the device and compares, part by part, the prompt's length, the ids written and
// the stop with the dump's, the parse of the dump's raw text with its language and text, and the joined text and the
// languages merged from the parts with the dump's, timing each part's stages and reporting the peak memory of the process.
//
// usage: qwen3-asr-split-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "json-reader.h"
#include "npy.h"
#include "qwen3-asr/layout.h"
#include "qwen3-asr/recognizer.h"
#include "qwen3-asr/split.h"
#include "reference-dumps.h"

#ifdef __APPLE__
#include <mach/mach.h>
#endif

using namespace qwen3_asr;

namespace {

/** The bounds of a split.npy, int32 as dump.py saves integers: where each part starts, and the end. */
std::vector<int64_t> read_split(const std::filesystem::path & file) {
    const std::vector<int32_t> bounds = read_npy(file.u8string()).i32;
    return std::vector<int64_t>(bounds.begin(), bounds.end());
}

/** The process's peak memory footprint in gigabytes, where the system reports it, and 0 elsewhere. */
double peak_gigabytes() {
#ifdef __APPLE__
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &count) == KERN_SUCCESS) return info.ledger_phys_footprint_peak / 1e9;
#endif
    return 0;
}

/**
 * The languages qwen-asr names in `names`, its merge_languages() of the parts' joined with commas, as indices of the
 * model's languages; a name the model does not hold throws.
 */
std::vector<size_t> languages_named(const ModelFile & m, const std::string & names) {
    const std::vector<std::string> known = m.str_array("qwen3-asr.language_names");
    std::vector<size_t> out;
    for (size_t at = 0; at < names.size();) {
        const size_t comma = std::min(names.find(',', at), names.size());
        const std::string name = names.substr(at, comma - at);
        const auto it = std::find(known.begin(), known.end(), name);
        if (it == known.end()) throw std::runtime_error("the dump names the language " + name + ", which the model does not name");
        out.push_back((size_t) (it - known.begin()));
        at = comma + 1;
    }
    return out;
}

std::string seconds_list(const std::vector<int64_t> & bounds, int rate) {
    std::string out;
    for (size_t i = 1; i + 1 < bounds.size(); i++) {
        char s[32];
        std::snprintf(s, sizeof s, "%s%.4f", out.empty() ? "" : ", ", (double) bounds[i] / rate);
        out += s;
    }
    return out.empty() ? "none" : out;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    try {
        bool ok = true;
        const std::filesystem::path root = std::filesystem::u8path(args[2]);
        const ModelFile meta_file(args[1], layout);
        const Splitter splitter(meta_file);
        const int rate = (int) meta_file.u32("speech.sample_rate");
        std::vector<std::filesystem::path> cases;
        for (const auto & e : std::filesystem::directory_iterator(root / "split")) cases.push_back(e.path());
        std::vector<std::filesystem::path> inputs;
        for (const auto & e : std::filesystem::directory_iterator(root / std::filesystem::u8path(meta_file.str("general.name")))) {
            if (std::filesystem::exists(e.path() / "split.npy")) inputs.push_back(e.path());
        }
        std::sort(cases.begin(), cases.end());
        std::sort(inputs.begin(), inputs.end());
        for (const auto & d : cases) {
            const Npy audio = read_npy((d / "audio.npy").u8string());
            const std::vector<int64_t> want = read_split(d / "split.npy");
            const auto start = std::chrono::steady_clock::now();
            const std::vector<int64_t> got = splitter.bounds(audio.f32);
            const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::printf("%-14s %8.2f s, cut at %s: %s (%.3f s)\n", d.filename().u8string().c_str(), (double) audio.f32.size() / rate,
                        seconds_list(got, rate).c_str(), got == want ? "qwen-asr's" : ("DIFFERS from qwen-asr's " + seconds_list(want, rate)).c_str(), took);
            ok = ok && got == want;
        }

        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
        std::printf("backend: %s\n", ggml_backend_name(backend));
        {
            Recognizer recognizer(args[1], backend);
            for (const auto & d : inputs) {
                const Npy audio = read_npy((d / "audio.npy").u8string());
                const std::vector<int64_t> want = read_split(d / "split.npy"), got = splitter.bounds(audio.f32);
                std::printf("%s, %.2f s, cut at %s: %s\n", d.filename().u8string().c_str(), (double) audio.f32.size() / rate,
                            seconds_list(got, rate).c_str(), got == want ? "the dump's" : ("DIFFERS from the dump's " + seconds_list(want, rate)).c_str());
                ok = ok && got == want;
                if (got != want) continue;
                std::vector<PartReport> parts;
                const auto start = std::chrono::steady_clock::now();
                const Recognition found = recognizer.recognize(audio.f32, {}, [](double) { return true; }, &parts);
                const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                bool limited = false;
                for (size_t k = 0; k < parts.size(); k++) {
                    const std::filesystem::path p = d / "auto" / ("part" + std::to_string(k));
                    const Npy prompt = read_npy((p / "prompt_ids.npy").u8string()), ids = read_npy((p / "ids.npy").u8string());
                    const PartReport & s = parts[k];
                    const bool same_prompt = s.prompt_rows == (int64_t) prompt.i32.size();
                    const JsonValue meta = parse_json(dump_text(p / "meta.json"));
                    const bool dump_limited = meta.member("stop")->text == "max_new_tokens";
                    const Parsed parse = recognizer.transcript().parse(dump_text(p / "raw.txt"), std::nullopt);
                    const bool parsed = parse.text == dump_text(p / "text.txt") &&
                                        (parse.language ? std::vector<size_t>{*parse.language} : std::vector<size_t>{}) ==
                                            languages_named(meta_file, meta.member("parsed_language")->text);
                    limited = limited || s.limited;
                    size_t same = 0;
                    while (same < s.ids.size() && same < ids.i32.size() && s.ids[same] == ids.i32[same]) same++;
                    std::printf("  part %zu: %lld prompt rows (%s), %zu ids written, the first %zu of them the dump's %zu, stop %s (the dump's %s), "
                                "the parse of the dump's raw text %s\n",
                                k, (long long) s.prompt_rows, same_prompt ? "the dump's" : "DIFFER", s.ids.size(), same, ids.i32.size(),
                                s.limited ? "model_limit" : "complete", dump_limited ? "model_limit" : "complete",
                                parsed ? "its language and text" : "DIFFERS");
                    std::printf("    frontend %.3f s, encoder %.3f s, prefill %.3f s, decoding %.3f s (%.1f tokens per second)\n", s.frontend,
                                s.encoder, s.prefill, s.decode, s.ids.size() / s.decode);
                    ok = ok && same_prompt && parsed;
                }
                const std::string want_text = dump_text(d / "auto" / "text.txt");
                const bool same_languages =
                    found.languages == languages_named(meta_file, parse_json(dump_text(d / "auto" / "meta.json")).member("language")->text);
                std::string tags;
                for (size_t language : found.languages) tags += (tags.empty() ? "" : ",") + recognizer.languages()[language];
                std::printf("  text %s, %zu bytes, stop %s, languages %s (%s), in %.1f s; peak memory footprint %.2f GB\n",
                            found.text == want_text ? "the dump's" : "DIFFERS", found.text.size(), found.limited ? "model_limit" : "complete",
                            tags.empty() ? "none" : tags.c_str(), same_languages ? "the dump's" : "DIFFER", took, peak_gigabytes());
                ok = ok && found.limited == limited && found.text.size() > 0 && same_languages;
            }
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
