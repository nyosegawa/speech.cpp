#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
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

int run_voice(const CommandLine & line, FILE *) {
    const Loading loading = line.loading(false, std::string("cpu"));
    const LoadParams params = load_params(loading);
    const auto t0 = std::chrono::steady_clock::now();
    check(speech_voice_make(line.args[0].c_str(), line.args[1].c_str(), line.args[2].c_str(), params.get()));
    const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "wrote %s in %.2f s\n", line.args[2].c_str(), took);
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
            value = v;
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
    for (size_t i = 0; i < speech_model_info_option_count(m); i++) {
        const speech_option o = speech_model_info_option(m, i);
        const std::string name = speech_option_name(o);
        row(i == 0 ? "options" : "", name + std::string(name.size() < 16 ? 16 - name.size() : 1, ' ') + option_line(m, o));
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
    c.usage = "voice MODEL REFERENCE.wav VOICE.gguf [options]";
    c.summary = "make a voice file from a reference recording";
    c.description =
        "Makes a voice file from a reference recording in a WAVE file at any rate, reading only the codec's encoder from\n"
        "MODEL, for a model whose information says it takes voice files (Irodori-TTS). On the CPU, the default here, the\n"
        "voice's latent is the official encoder's to 99 dB SNR; a GPU computes it in less precision.";
    c.flags = {
        device_flag("cpu (the default here), auto, gpu or a name `speech devices` lists"),
        threads_flag(),
        verbose_flag("also report the release and the device"),
    };
    c.min_args = 3;
    c.max_args = 3;
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
