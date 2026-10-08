// Checks Silero VAD against the official silero-vad package on each dump of reference/silero-vad/dump.py, in the order
// data flows, each stage from the dump's own input: each chunk's input with its context, cut from the dump's audio; the
// STFT's magnitude; each encoder block; the LSTM cell's h and c over every chunk from the dump's encoder output; the
// decoder's probabilities from the dump's h; then the probabilities from the audio alone, which graphs of any number of
// chunks must give bit for bit, as audio given a piece at a time computes them, and the regions of every set of options
// in the dump's regions.json, which must equal the official's sample for sample, both from the official's probabilities
// and from the ones computed here. The dumps are those in <reference out dir>/<the file's general.name>/.
//
// usage: silero-vad-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "ggml-cpu.h"
#include "json-reader.h"
#include "npy.h"
#include "reference-dumps.h"
#include "silero-vad/detector.h"
#include "silero-vad/regions.h"

using namespace silero_vad;

namespace {

/** The folders under <root>/<general.name>/ that hold probs.npy, sorted. */
std::vector<std::filesystem::path> dumps(const std::string & root, const ModelFile & model) {
    const std::filesystem::path dir = std::filesystem::u8path(root) / std::filesystem::u8path(model.str("general.name"));
    std::vector<std::filesystem::path> out;
    if (std::filesystem::is_directory(dir)) {
        for (const auto & e : std::filesystem::directory_iterator(dir, std::filesystem::directory_options::follow_directory_symlink)) {
            if (std::filesystem::is_regular_file(e.path() / "probs.npy")) out.push_back(e.path());
        }
    }
    std::sort(out.begin(), out.end());
    if (out.empty()) throw std::runtime_error("no dump of reference/silero-vad/dump.py is under " + dir.u8string());
    return out;
}

Npy load(const std::filesystem::path & dump, const std::string & name) {
    return read_npy((dump / (name + ".npy")).u8string());
}

/** A dump of PyTorch's [n, channels, frames] in ggml's order, [channels, frames, n]. */
std::vector<float> by_channel(const Npy & a) {
    const int64_t n = a.shape[0], channels = a.shape[1], frames = a.shape[2];
    std::vector<float> out(a.f32.size());
    for (int64_t k = 0; k < n; k++) {
        for (int64_t c = 0; c < channels; c++) {
            for (int64_t f = 0; f < frames; f++) out[(size_t) ((k * frames + f) * channels + c)] = a.f32[(size_t) ((k * channels + c) * frames + f)];
        }
    }
    return out;
}

/** The rule of a set of options in regions.json: the file's defaults with the set's options. */
RegionRule rule_of(const ModelFile & m, const JsonValue & options) {
    RegionRule r = default_rule(m);
    for (const auto & [name, value] : options.members) {
        const double v = std::stod(value.text);
        if (name == "threshold") r.threshold = v;
        else if (name == "min_speech_duration_ms") r.min_speech_duration_ms = (int64_t) v;
        else if (name == "min_silence_duration_ms") r.min_silence_duration_ms = (int64_t) v;
        else if (name == "speech_pad_ms") r.speech_pad_ms = (int64_t) v;
        else if (name == "max_speech_duration_s") r.max_speech_duration_s = v;
        else throw std::runtime_error("regions.json sets " + name + ", which the check does not know");
    }
    return r;
}

bool same_region(const Region & a, const Region & b) {
    return a.start == b.start && a.end == b.end;
}

/**
 * What is wrong with the regions that RegionStream gives as the chunks come, settled after each whole chunk: a region
 * must not be given before the rule decided at the end gives it alike whether 1000 chunks of silence (probability 0) or
 * of speech (1) follow, and without max_speech_duration_s it must be given at the first chunk after which they do, which
 * then makes it certain; a limit cuts a region at its longest silence, which audio between the two can lengthen. Once the
 * audio has ended the regions must be the whole audio's. Empty when nothing is.
 */
std::string as_chunks_come(const std::vector<float> & probs, int64_t samples, int rate, int chunk, const RegionRule & rule) {
    RegionStream settled(rule, rate, chunk), walked(rule, rate, chunk);
    const int64_t n = (int64_t) probs.size();
    // The last chunk may be partial and the audio ends after it, which no continuation follows.
    for (int64_t k = 0; k + 1 < n; k++) {
        settled.add(probs[(size_t) k]);
        walked.add(probs[(size_t) k]);
        settled.settle((k + 1) * chunk);
        const auto followed = [&](float p) {
            RegionStream s = walked;
            for (int i = 0; i < 1000; i++) s.add(p);
            s.end((k + 1001) * chunk);
            return s.regions();
        };
        const std::vector<Region> silence = followed(0.0f), speech = followed(1.0f);
        const auto certain = [&](size_t i) { return i < silence.size() && i < speech.size() && same_region(silence[i], speech[i]); };
        const std::vector<Region> & given = settled.regions();
        for (size_t i = 0; i < given.size(); i++) {
            if (!certain(i) || !same_region(given[i], silence[i])) return "region " + std::to_string(i) + " is given at chunk " + std::to_string(k) + " before it is certain";
        }
        if (std::isinf(rule.max_speech_duration_s) && certain(given.size())) {
            return "region " + std::to_string(given.size()) + " is certain at chunk " + std::to_string(k) + " but not given";
        }
    }
    if (n > 0) {
        settled.add(probs.back());
        walked.add(probs.back());
    }
    settled.end(samples);
    walked.end(samples);
    const std::vector<Region> & a = settled.regions(), & b = walked.regions();
    if (a.size() != b.size() || !std::equal(a.begin(), a.end(), b.begin(), same_region)) return "the regions given as the chunks came differ from the whole's";
    return "";
}

std::string regions_text(const std::vector<Region> & regions) {
    std::string out;
    for (const Region & r : regions) out += (out.empty() ? "" : " ") + std::to_string(r.start) + "-" + std::to_string(r.end);
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
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
        std::printf("backend: %s\n", ggml_backend_name(backend));
        // Measured on an Apple M5 on 2026-10-08 over the six dumps, each product of the network multiplying one column:
        // every stage lies 125 to 164 dB from the official on the CPU in F32 and 119 to 155 dB on Metal, and the
        // probabilities from the audio within 3.6e-6 and 3.8e-6 of it. A GPU whose kernels round to half precision lies
        // further: Metal's matrix kernel, which products of more columns took, gave 62 to 113 dB and 6.2e-3. A wrong window,
        // padding, stride or gate gives a few dB.
        const bool cpu = ggml_backend_is_cpu(backend), single = cpu || std::string(ggml_backend_name(backend)).rfind("MTL", 0) == 0;
        const double stage_db = single ? 100 : 50, probs_abs = single ? 1e-5 : 2e-2;
        bool ok = true;
        {
            Detector detector(args[1], backend);
            const Network & net = detector.network();
            const ModelFile & model = detector.model();
            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            for (const auto & d : dumps(args[2], model)) {
                const Npy audio = load(d, "audio"), input = load(d, "input"), probs = load(d, "probs");
                const int64_t n = probs.shape[0];
                std::printf("%s (%.2f s, %lld chunks)\n", d.filename().u8string().c_str(), (double) audio.f32.size() / detector.sample_rate(), (long long) n);
                const std::vector<float> inputs = detector.inputs(audio.f32, 0, detector.chunks(audio.f32));
                const bool same_inputs = inputs == input.f32;
                std::printf("  %-32s %s\n", "inputs", same_inputs ? "equal" : "DIFFER");
                ok = ok && same_inputs;

                // The stages whose chunks are independent, all chunks in one graph.
                {
                    Graph g;
                    ggml_context * ctx = g.ctx();
                    ggml_tensor * magnitude = net.stft(ctx, g.input(input.f32, net.context() + net.chunk(), n));
                    g.output(magnitude);
                    std::vector<ggml_tensor *> blocks;
                    std::vector<Npy> wanted;
                    for (int i = 0; i < net.blocks(); i++) {
                        const Npy before = load(d, i == 0 ? std::string("stft") : "block" + std::to_string(i - 1));
                        ggml_tensor * x = g.input(by_channel(before), before.shape[1], before.shape[2], n);
                        blocks.push_back(net.block(ctx, i, x));
                        g.output(blocks.back());
                        wanted.push_back(load(d, "block" + std::to_string(i)));
                    }
                    g.compute(backend, allocr);
                    const Diff ds = compare(Graph::read(magnitude), by_channel(load(d, "stft")));
                    print_diff("  stft", ds);
                    ok = ok && ds.snr_db > stage_db;
                    for (int i = 0; i < net.blocks(); i++) {
                        const Diff db = compare(Graph::read(blocks[i]), by_channel(wanted[i]));
                        print_diff("  block " + std::to_string(i), db);
                        ok = ok && db.snr_db > stage_db;
                    }
                }

                // The cell from the dump's encoder output, in the detector's blocks of chunks, and the decoder from the
                // dump's h.
                {
                    const Npy encoded = load(d, "block" + std::to_string(net.blocks() - 1));
                    const Npy want_h = load(d, "lstm_h"), want_c = load(d, "lstm_c");
                    const int64_t channels = encoded.shape[1];
                    std::vector<float> got_h, got_c, h((size_t) net.hidden(), 0.0f), c((size_t) net.hidden(), 0.0f), part;
                    for (int64_t first = 0; first < n; first += Detector::kBlock) {
                        const int64_t count = std::min(Detector::kBlock, n - first);
                        Graph g;
                        ggml_tensor * x = g.input(std::vector<float>(encoded.f32.begin() + first * channels, encoded.f32.begin() + (first + count) * channels),
                                                  channels, count);
                        const Network::Steps steps = net.lstm(g.ctx(), x, g.input(h, net.hidden()), g.input(c, net.hidden()), true);
                        g.output(steps.h);
                        g.output(steps.c);
                        g.output(steps.last_h);
                        g.output(steps.last_c);
                        g.compute(backend, allocr);
                        Graph::read(steps.h, part);
                        got_h.insert(got_h.end(), part.begin(), part.end());
                        Graph::read(steps.c, part);
                        got_c.insert(got_c.end(), part.begin(), part.end());
                        Graph::read(steps.last_h, h);
                        Graph::read(steps.last_c, c);
                    }
                    const Diff dh = compare(got_h, want_h.f32), dc = compare(got_c, want_c.f32);
                    print_diff("  lstm h", dh);
                    print_diff("  lstm c", dc);
                    ok = ok && got_h.size() == want_h.f32.size() && dh.snr_db > stage_db && dc.snr_db > stage_db;

                    Graph g;
                    ggml_tensor * p = net.decode(g.ctx(), g.input(want_h.f32, net.hidden(), n));
                    g.output(p);
                    g.compute(backend, allocr);
                    const Diff dp = compare(Graph::read(p), probs.f32);
                    print_diff("  decoder", dp);
                    ok = ok && dp.max_abs < probs_abs;
                }

                const std::vector<float> got = detector.probabilities(audio.f32);
                const Diff dp = compare(got, probs.f32);
                print_diff("  probabilities from the audio", dp);
                ok = ok && got.size() == probs.f32.size() && dp.max_abs < probs_abs;

                // The probabilities in graphs of other numbers of chunks than the detector's blocks, bit for bit.
                bool independent = true;
                for (const int64_t size : {1, 2, 3, 5, 8, 13, 100}) {
                    std::vector<float> in_blocks;
                    CellState state = detector.start();
                    for (int64_t first = 0; first < n; first += size) {
                        const int64_t count = std::min(size, n - first);
                        detector.compute(detector.inputs(audio.f32, first, count), count, state, in_blocks);
                    }
                    const bool same = in_blocks.size() == got.size() && std::memcmp(in_blocks.data(), got.data(), got.size() * sizeof(float)) == 0;
                    if (!same) {
                        std::printf("  %-32s DIFFER, by up to %.2e\n", ("in graphs of " + std::to_string(size) + " chunks").c_str(),
                                    compare(in_blocks, got).max_abs);
                    }
                    independent = independent && same;
                }
                std::printf("  %-32s %s\n", "in graphs of 1 to 100 chunks", independent ? "equal" : "DIFFER");
                ok = ok && independent;

                const JsonValue sets = parse_json(dump_text(d / "regions.json"));
                for (const auto & [name, set] : sets.members) {
                    std::vector<Region> want;
                    for (const JsonValue & r : set.member("regions")->items) want.push_back({std::stoll(r.items[0].text), std::stoll(r.items[1].text)});
                    const RegionRule rule = rule_of(model, *set.member("options"));
                    const auto same = [&](const std::vector<Region> & a) {
                        return a.size() == want.size() && std::equal(a.begin(), a.end(), want.begin(), [](const Region & x, const Region & y) {
                                   return x.start == y.start && x.end == y.end;
                               });
                    };
                    const std::vector<Region> official = speech_regions(probs.f32, (int64_t) audio.f32.size(), detector.sample_rate(), net.chunk(), rule);
                    const std::vector<Region> own = speech_regions(got, (int64_t) audio.f32.size(), detector.sample_rate(), net.chunk(), rule);
                    // A probability within the backend's error of a threshold the rule compares it with may fall on the other
                    // side, which moves a region by a chunk or more; on the CPU none lies so near on these dumps.
                    const double neg = std::max(rule.threshold - rule.neg_threshold_offset, rule.neg_threshold_floor);
                    const bool near = !cpu && std::any_of(probs.f32.begin(), probs.f32.end(), [&](float p) {
                        return std::fabs(p - rule.threshold) < probs_abs || std::fabs(p - neg) < probs_abs;
                    });
                    std::printf("  regions %-24s %zu, %s from the official probabilities, %s from these\n", name.c_str(), want.size(),
                                same(official) ? "equal" : "DIFFER", same(own) ? "equal" : near ? "other, a probability lying near a threshold," : "DIFFER");
                    if (!same(official) || !same(own)) {
                        std::printf("    want %s\n    from the official %s\n    from these %s\n", regions_text(want).c_str(), regions_text(official).c_str(),
                                    regions_text(own).c_str());
                    }
                    ok = ok && same(official) && (same(own) || near);
                }

                // The regions of the official probabilities as the chunks come, with each set of options of the dump and
                // with sets whose regions' ends wait on what follows them.
                std::vector<std::pair<std::string, RegionRule>> rules;
                for (const auto & [name, set] : sets.members) rules.push_back({name, rule_of(model, *set.member("options"))});
                for (const auto & [name, pad, silence, speech, max] : std::vector<std::tuple<std::string, int64_t, int64_t, int64_t, double>>{
                         {"pad100-silence0", 100, 0, -1, INFINITY}, {"pad200-silence50-speech0", 200, 50, 0, INFINITY}, {"pad100-silence0-max2", 100, 0, -1, 2.0}}) {
                    RegionRule r = default_rule(model);
                    r.speech_pad_ms = pad;
                    r.min_silence_duration_ms = silence;
                    if (speech >= 0) r.min_speech_duration_ms = speech;
                    r.max_speech_duration_s = max;
                    rules.push_back({name, r});
                }
                std::string wrong;
                for (const auto & [name, rule] : rules) {
                    if (!wrong.empty()) break;
                    wrong = as_chunks_come(probs.f32, (int64_t) audio.f32.size(), detector.sample_rate(), net.chunk(), rule);
                    if (!wrong.empty()) wrong = name + ": " + wrong;
                }
                std::printf("  %-32s %s\n", "regions as the chunks come", wrong.empty() ? ("each given once certain, " + std::to_string(rules.size()) + " sets").c_str()
                                                                                       : ("WRONG: " + wrong).c_str());
                ok = ok && wrong.empty();
            }
            ggml_gallocr_free(allocr);
        }
        ggml_backend_free(backend);
        std::printf("%s\n", ok ? "ok" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
