#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "commands.h"
#include "json-reader.h"
#include "json.h"

// The subcommands about models and devices rather than requests: speech voice makes a voice file, speech info prints
// what a model file says without loading it, and speech devices lists the devices a model can run on.

namespace {

std::string list(size_t count, const std::function<std::string(size_t)> & get) {
    std::string out;
    for (size_t i = 0; i < count; i++) out += (i ? ", " : "") + get(i);
    return out;
}

std::string gigabytes(uint64_t bytes) {
    char s[32];
    std::snprintf(s, sizeof s, "%.2f GB", bytes / 1e9);
    return s;
}

/** The file name the official runtime gives a speaker-inversion embedding, and the only one it reads one from. */
constexpr const char * kEmbeddingSuffix = ".speaker.safetensors";

bool is_embedding(const std::string & path) {
    const std::string suffix = kEmbeddingSuffix;
    return path.size() > suffix.size() && path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/** A speaker-inversion embedding: `tokens` vectors of `dim` values, row by row. */
struct Embedding {
    std::vector<float> values;
    size_t tokens = 0, dim = 0;
};

/**
 * The embedding of a .speaker.safetensors file as the official runtime saves and reads it (speaker_inversion.py): the
 * tensor "speaker_embedding", float32 [tokens, dim] or [1, tokens, dim], after the safetensors header, a little-endian
 * u64 length and that much JSON.
 */
Embedding read_embedding(const std::string & path) {
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
    if (!f) throw Failure("io", "embedding", "cannot open " + path + "; check the path and that the file can be read");
    const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto refuse = [&](const std::string & what) {
        return Failure("invalid_argument", "embedding", path + " is not a speaker-inversion embedding as the official runtime saves one: " + what);
    };
    uint64_t header = 0;
    if (bytes.size() < 8) throw refuse("it is shorter than a safetensors header");
    for (int i = 7; i >= 0; i--) header = header << 8 | (unsigned char) bytes[i];
    if (header > bytes.size() - 8) throw refuse("its header runs past the file");
    JsonValue json;
    try {
        json = parse_json(bytes.substr(8, header));
    } catch (const std::invalid_argument & e) {
        throw refuse(std::string("its header is not JSON (") + e.what() + ")");
    }
    const JsonValue * tensor = json.member("speaker_embedding");
    if (!tensor) throw refuse("it holds no tensor speaker_embedding");
    const JsonValue * dtype = tensor->member("dtype"), * shape = tensor->member("shape"), * offsets = tensor->member("data_offsets");
    if (!dtype || dtype->text != "F32") throw refuse("speaker_embedding is not float32; save it as float32");
    if (!shape || !offsets || offsets->items.size() != 2) throw refuse("speaker_embedding has no shape or no offsets");
    std::vector<size_t> axes;
    for (const JsonValue & a : shape->items) axes.push_back((size_t) std::stoull(a.text));
    if (axes.size() == 3 && axes[0] == 1) axes.erase(axes.begin());
    if (axes.size() != 2 || axes[0] == 0 || axes[1] == 0) throw refuse("speaker_embedding is not [tokens, dim] or [1, tokens, dim]");
    const size_t begin = (size_t) std::stoull(offsets->items[0].text), end = (size_t) std::stoull(offsets->items[1].text);
    if (end - begin != axes[0] * axes[1] * sizeof(float) || 8 + header + end > bytes.size()) throw refuse("speaker_embedding's data does not fit its shape");
    Embedding e;
    e.tokens = axes[0];
    e.dim = axes[1];
    e.values.resize(e.tokens * e.dim);
    std::memcpy(e.values.data(), bytes.data() + 8 + header + begin, end - begin);
    return e;
}

int run_voice(const CommandLine & line, FILE *) {
    const Loading loading = line.loading(false, std::string("cpu"));
    const LoadParams params = load_params(loading);
    const std::optional<std::string> lufs = line.value("--lufs");
    if (lufs && line.has("--keep-loudness")) throw UsageError("give --lufs or --keep-loudness, not both");
    speech_voice_params * raw = nullptr;
    check(speech_voice_params_new(&raw));
    const std::unique_ptr<speech_voice_params, decltype(&speech_voice_params_free)> voice(raw, speech_voice_params_free);
    for (size_t i = 1; i + 1 < line.args.size(); i++) {
        if (!is_embedding(line.args[i])) {
            check(speech_voice_params_add_reference(raw, line.args[i].c_str()));
            continue;
        }
        const Embedding e = read_embedding(line.args[i]);
        check(speech_voice_params_set_embedding(raw, e.values.data(), e.tokens, e.dim));
    }
    if (lufs) {
        char * end = nullptr;
        const double value = std::strtod(lufs->c_str(), &end);
        if (lufs->empty() || *end != '\0' || !std::isfinite(value)) throw UsageError("--lufs takes a number, not \"" + *lufs + "\"");
        check(speech_voice_params_set_loudness(raw, value));
    }
    if (line.has("--keep-loudness")) check(speech_voice_params_keep_loudness(raw));
    const std::string & out = line.args.back();
    const auto t0 = std::chrono::steady_clock::now();
    check(speech_voice_make_from(line.args[0].c_str(), raw, out.c_str(), params.get()));
    const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "wrote %s in %.2f s\n", out.c_str(), took);
    if (line.has("-v")) std::fprintf(stderr, "speech.cpp %s, device %s\n", speech_version(), loading.device->c_str());
    return 0;
}

/** An option's declaration in one line: its type, whether it is required or its default, its range or choices. */
std::string option_line(const speech_model_info * m, speech_option o) {
    static const char * types[] = {"string", "int", "float", "bool"};
    const speech_type type = speech_option_type(o);
    std::string out = types[type];
    if (speech_model_info_option_required(m, o)) out += ", required";
    if (speech_model_info_option_has_default(m, o)) {
        std::string value;
        if (type == SPEECH_TYPE_STRING) {
            const char * v = nullptr;
            check(speech_model_info_option_default_string(m, o, &v));
            // An empty default, the prompt's, reads as nothing after "default" unless it is quoted.
            value = *v ? v : "\"\"";
        } else if (type == SPEECH_TYPE_INT) {
            int64_t v = 0;
            check(speech_model_info_option_default_int(m, o, &v));
            value = std::to_string(v);
        } else if (type == SPEECH_TYPE_FLOAT) {
            double v = 0;
            check(speech_model_info_option_default_float(m, o, &v));
            value = json_number(v);
        } else {
            int v = 0;
            check(speech_model_info_option_default_bool(m, o, &v));
            value = v ? "true" : "false";
        }
        out += ", default " + value;
    }
    if (type == SPEECH_TYPE_INT || type == SPEECH_TYPE_FLOAT) {
        double lo = 0, hi = 0;
        int exclusive = 0;
        check(speech_model_info_option_range(m, o, &lo, &hi, &exclusive));
        const auto bound = [&](double v) { return type == SPEECH_TYPE_INT ? std::to_string((long long) v) : json_number(v); };
        std::string range;
        if (std::isfinite(lo)) range = (exclusive ? "above " : "from ") + bound(lo);
        if (std::isfinite(hi)) range += (range.empty() ? "" : " ") + std::string("to ") + bound(hi);
        if (!range.empty()) out += ", " + range;
    }
    const size_t choices = speech_model_info_option_choice_count(m, o);
    if (choices) out += ", one of " + list(choices, [&](size_t i) { return std::string(speech_model_info_option_choice(m, o, i)); });
    if (o == SPEECH_OPT_LANGUAGE) out += " (or auto, or a region or script of one)";
    if (type == SPEECH_TYPE_STRING && !choices && o == SPEECH_OPT_VOICE) out += ", none until a voice is added";
    out += speech_model_info_option_steers(m, o) ? "; steers" : "; checked, not used";
    return out;
}

/** A metadata value for the text form: arrays of more than eight items shortened, with their length. */
std::string meta_text(const std::string & value) {
    const JsonValue v = parse_json(value);
    if (v.kind != JsonValue::Kind::Array || v.items.size() <= 8) return value;
    std::string out = "[";
    for (size_t i = 0; i < 8; i++) out += (i ? "," : "") + to_json(v.items[i]);
    return out + ",...] (" + std::to_string(v.items.size()) + " items)";
}

void print_info(const speech_model_info * m, bool meta, FILE * out) {
    const bool synthesis = speech_model_info_task(m) == SPEECH_TASK_SYNTHESIS;
    std::string text = std::string(speech_model_info_name(m)) + "\n";
    const auto row = [&](const std::string & label, const std::string & value) {
        text += label + std::string(label.size() < 15 ? 15 - label.size() : 1, ' ') + value + "\n";
    };
    const char * finetune = speech_model_info_finetune(m), * version = speech_model_info_version(m);
    const char * repository = nullptr, * revision = nullptr;
    check(speech_model_info_source(m, &repository, &revision));
    row("organization", speech_model_info_organization(m));
    row("basename", speech_model_info_basename(m));
    row("size label", speech_model_info_size_label(m));
    if (finetune) row("finetune", finetune);
    if (version) row("version", version);
    row("license", speech_model_info_license(m));
    row("source", std::string(repository) + " at " + revision);
    row("weight type", speech_model_info_weight_type(m));
    row("architecture", std::string(speech_model_info_architecture(m)) + ", layout " + std::to_string(speech_model_info_layout(m)));
    row("task", std::string(task_name(speech_model_info_task(m))) + " at " + std::to_string(speech_model_info_sample_rate(m)) + " Hz" +
                    (synthesis ? speech_model_info_incremental(m) ? ", passing audio while it makes the rest" : ", making the whole speech before its audio" : ""));
    row("languages", list(speech_model_info_language_count(m), [&](size_t i) { return std::string(speech_model_info_language(m, i)); }));
    if (synthesis) {
        const size_t n = speech_model_info_voice_count(m);
        if (n == 0) row("voices", "none of its own; add them with --add-voice");
        for (size_t i = 0; i < n; i++) {
            std::string about = speech_model_info_voice_language(m, i);
            const std::string gender = speech_model_info_voice_gender(m, i), description = speech_model_info_voice_description(m, i);
            if (!gender.empty()) about += (about.empty() ? "" : ", ") + gender;
            if (!description.empty()) about += (about.empty() ? "" : ": ") + description;
            std::string name = speech_model_info_voice_name(m, i);
            row(i == 0 ? "voices" : "", name + std::string(name.size() < 14 ? 14 - name.size() : 1, ' ') + about);
        }
        row("voice files", speech_model_info_voice_files(m) ? std::string("yes, of the codec ") + speech_model_info_voice_codec(m) : "no");
        row("text", "at most " + std::to_string(speech_model_info_max_text_tokens(m)) + " tokens");
    }
    size_t width = 16;
    for (size_t i = 0; i < speech_model_info_option_count(m); i++) {
        width = std::max(width, std::string(speech_option_name(speech_model_info_option(m, i))).size() + 2);
    }
    for (size_t i = 0; i < speech_model_info_option_count(m); i++) {
        const speech_option o = speech_model_info_option(m, i);
        const std::string name = speech_option_name(o);
        row(i == 0 ? "options" : "", name + std::string(width - name.size(), ' ') + option_line(m, o));
    }
    row("size", gigabytes(speech_model_info_file_bytes(m)) + " file, " + gigabytes(speech_model_info_weight_bytes(m)) + " of weights");
    if (meta) {
        text += "\n";
        for (size_t i = 0; i < speech_model_info_meta_count(m); i++) {
            text += std::string(speech_model_info_meta_key(m, i)) + " = " + meta_text(speech_model_info_meta_value(m, i)) + "\n";
        }
    }
    std::fputs(text.c_str(), out);
}

int run_info(const CommandLine & line, FILE * out) {
    const ModelInfo info = file_info(line.args[0]);
    const speech_model_info * m = info.get();
    const bool meta = line.has("--meta");
    if (!line.has("--json")) {
        print_info(m, meta, out);
    } else {
        std::string json = speech_model_info_json(m);
        if (meta) {
            json.pop_back();
            json += ",\"meta\":{";
            for (size_t i = 0; i < speech_model_info_meta_count(m); i++) {
                json += (i ? "," : "") + json_string(speech_model_info_meta_key(m, i)) + ":" + speech_model_info_meta_value(m, i);
            }
            json += "}}";
        }
        std::fprintf(out, "%s\n", json.c_str());
    }
    std::fflush(out);
    return 0;
}

int run_devices(const CommandLine & line, FILE * out) {
    const bool json = line.has("--json");
    std::string text = json ? "{\"devices\":[" : "";
    for (size_t i = 0; i < speech_device_count(); i++) {
        uint64_t total = 0, free = 0;
        check(speech_device_memory(i, &total, &free));
        if (json) {
            text += std::string(i ? "," : "") + "{\"name\":" + json_string(speech_device_name(i)) + ",\"description\":" +
                    json_string(speech_device_description(i)) + ",\"kind\":\"" + device_kind_name(i) + "\",\"memory_total\":" +
                    std::to_string(total) + ",\"memory_free\":" + std::to_string(free) + "}";
        } else {
            char row[512];
            std::snprintf(row, sizeof row, "%-10s %-5s %s, %.1f GB, %.1f GB free\n", speech_device_name(i), device_kind_name(i),
                          speech_device_description(i), total / 1e9, free / 1e9);
            text += row;
        }
    }
    if (json) text += "]}\n";
    std::fputs(text.c_str(), out);
    std::fflush(out);
    return 0;
}

}  // namespace

Command voice_command() {
    Command c;
    c.name = "voice";
    c.usage = "voice MODEL (REFERENCE.wav... | EMBEDDING.speaker.safetensors) VOICE.gguf [options]";
    c.summary = "make a voice file from reference recordings or a speaker-inversion embedding";
    c.description =
        "Makes a voice file from one or more reference recordings in WAVE files at any rate, each encoded on its own and\n"
        "joined in order, reading only the codec's encoder from MODEL, for a model whose information says it takes voice\n"
        "files (Irodori-TTS). Each recording is brought to the model's loudness first, or to --lufs, or kept as it is with\n"
        "--keep-loudness. On the CPU, the default here, the voice's latent is the official encoder's to 99 dB SNR; a GPU\n"
        "computes it in less precision. A speaker-inversion embedding in a .speaker.safetensors file, as the official\n"
        "runtime saves one, makes a voice of its own for MODEL, which no other model takes.";
    c.flags = {
        {"--lufs", "LUFS", false, "the loudness each recording is brought to, instead of the model's (-16)"},
        {"--keep-loudness", "", false, "keep each recording's loudness, scaling down a peak above 1"},
        device_flag("cpu (the default here), auto, gpu or a name `speech devices` lists"),
        threads_flag(),
        verbose_flag("also report the release and the device"),
    };
    c.min_args = 3;
    c.max_args = SIZE_MAX;
    c.run = run_voice;
    return c;
}

Command info_command() {
    Command c;
    c.name = "info";
    c.usage = "info MODEL [--json] [--meta]";
    c.summary = "print a model file's information without loading it";
    c.description =
        "Prints what the model file says of its model, read from its metadata without its weights or a device: its name,\n"
        "organization, basename, size label, finetune and version, license, source and weight type, its architecture\n"
        "(the family that runs it) and layout, task and rate, languages, voices, each option with its type, default,\n"
        "range or choices, the longest text and the sizes.";
    c.flags = {
        {"--json", "", false, "print the model information as the JSON object that the worker's ready message carries"},
        {"--meta", "", false, "add every metadata entry of the GGUF file, whole as a \"meta\" object in JSON"},
    };
    c.min_args = 1;
    c.max_args = 1;
    c.run = run_info;
    return c;
}

Command devices_command() {
    Command c;
    c.name = "devices";
    c.usage = "devices [--json]";
    c.summary = "list the devices a model can run on";
    c.description = "Lists the CPU and the GPUs a model can run on, in ggml's order, with their kind and memory.";
    c.flags = {
        {"--json", "", false, "print one JSON object, {\"devices\": [{\"name\", \"description\", \"kind\", \"memory_total\", \"memory_free\"}]}"},
    };
    c.run = run_devices;
    return c;
}
