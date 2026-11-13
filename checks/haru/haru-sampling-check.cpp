// Checks Haru's text sampling against the official haru.sampling, on the cases of reference/haru/sampling_cases.py and
// a dump of reference/haru/dump.py: the tokens each draw takes from and their probabilities for each setting, the
// file's settings against the official defaults, DRY's penalties after each token of each sequence, the logits a
// conversation's frames draw from, formed from the dump's text logits, and the text of the dump's tokens as the
// conversation streams it.
//
// usage: haru-sampling-check <model.gguf> <sampling dir> <dump dir>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "args.h"
#include "haru/detokenizer.h"
#include "haru/layout.h"
#include "haru/sampler.h"
#include "json-reader.h"
#include "npy.h"
#include "qwen2-tokenizer.h"

using namespace haru;

namespace {

/**
 * The largest difference of a probability that a softmax in float32 of `k` tokens explains, done with other
 * exponentials and another order of the sum: torch's vectorized exp and its sum against the C library's and a sum in
 * order, whose sums of k terms of at most 1 differ by k units of float32's rounding at most, and the exponentials by a
 * few.
 */
double probability_tolerance(size_t k) { return (double) (k + 64) * 0x1p-24; }

JsonValue read_json(const std::filesystem::path & path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path.string());
    return parse_json(std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()));
}

double number(const JsonValue & v, const char * name) { return std::stod(v.member(name)->text); }

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <sampling dir> <dump dir>\n", args[0].c_str());
        return 2;
    }
    const ModelFile model(args[1], model_layout);
    const std::filesystem::path cases = std::filesystem::u8path(args[2]), dump = std::filesystem::u8path(args[3]);
    bool ok = true;

    const Npy logits = read_npy((cases / "logits.npy").u8string());
    const size_t rows = (size_t) logits.shape[0], vocab = (size_t) logits.shape[1];
    for (const char * name : {"file", "top-p", "cold-top-k-1", "wide"}) {
        const JsonValue meta = read_json(cases / name / "meta.json");
        Sampling s;
        s.temperature = (float) number(meta, "temperature");
        s.top_k = std::stoi(meta.member("top_k")->text);
        s.top_p = (float) number(meta, "top_p");
        if (std::string(name) == "file") {
            const Sampling f = file_sampling(model);
            const bool same = f.temperature == s.temperature && f.top_k == s.top_k && f.top_p == s.top_p;
            std::printf("%-34s %s\n", "the file's settings", same ? "the official defaults" : "differ from the official defaults");
            ok = ok && same;
        }
        const Npy ids = read_npy((cases / name / "ids.npy").u8string());
        const Npy probabilities = read_npy((cases / name / "probabilities.npy").u8string());
        const size_t k = (size_t) ids.shape[1];
        size_t same_ids = 0, same_sets = 0;
        double worst = 0;
        for (size_t r = 0; r < rows; r++) {
            const std::vector<float> row(logits.f32.begin() + (std::ptrdiff_t) (r * vocab), logits.f32.begin() + (std::ptrdiff_t) ((r + 1) * vocab));
            const Candidates c = candidates(row, s);
            if (c.tokens.size() != k) continue;
            // torch.topk() orders tokens of equal logits, after the division by the temperature, as it pleases, which
            // changes no token's chance: the candidates match where they are the same tokens and each position holds
            // the same logit divided by the temperature.
            std::vector<int32_t> got(c.tokens), want(ids.i32.begin() + (std::ptrdiff_t) (r * k), ids.i32.begin() + (std::ptrdiff_t) ((r + 1) * k));
            const auto scaled = [&](int32_t t) { return s.temperature > 0.0f ? row[(size_t) t] / s.temperature : row[(size_t) t]; };
            bool same = true;
            for (size_t i = 0; same && i < k; i++) same = scaled(got[i]) == scaled(want[i]);
            same_ids += same && got == want;
            std::sort(got.begin(), got.end());
            std::sort(want.begin(), want.end());
            same = same && got == want;
            same_sets += same;
            for (size_t i = 0; same && i < k; i++) worst = std::max(worst, std::abs(c.weights[i] - (double) probabilities.f32[r * k + i]));
        }
        std::printf("%-34s %zu of %zu rows' candidates (%zu in the same order), probabilities within %.1e\n", name, same_sets, rows, same_ids, worst);
        ok = ok && same_sets == rows && worst <= probability_tolerance(k);
    }

    const DryConfig dry_config = file_dry(model);
    const TextIds text_ids = file_text_ids(model);
    const std::vector<int32_t> silent = {text_ids.pad, text_ids.eop, text_ids.eos};
    std::ifstream lines(cases / "dry.jsonl");
    std::string line;
    int sequences = 0;
    size_t steps = 0, same_steps = 0;
    while (std::getline(lines, line)) {
        const JsonValue v = parse_json(line);
        const JsonValue & tokens = *v.member("tokens");
        const JsonValue & want = *v.member("penalties");
        Dry dry(dry_config, silent);
        for (size_t i = 0; i < tokens.items.size(); i++) {
            dry.update(std::stoi(tokens.items[i].text));
            const auto & got = dry.penalties();
            const JsonValue & w = want.items[i];
            bool same = got.size() == w.items.size();
            for (size_t j = 0; same && j < got.size(); j++) {
                same = got[j].first == std::stoi(w.items[j].items[0].text) && got[j].second == std::stod(w.items[j].items[1].text);
            }
            same_steps += same;
            steps++;
        }
        sequences++;
    }
    std::printf("%-34s %zu of %zu steps of %d sequences equal\n", "DRY's penalties", same_steps, steps, sequences);
    ok = ok && steps > 0 && same_steps == steps;

    // The dump's frames: its text logits shaped as the session shapes them, against the logits it drew from.
    const Npy raw = read_npy((dump / "logits.npy").u8string());
    const Npy silent_logits = read_npy((dump / "silent_logits.npy").u8string());
    const Npy sampled = read_npy((dump / "sampled_logits.npy").u8string());
    const Npy tokens = read_npy((dump / "tokens.npy").u8string());
    const size_t frames = (size_t) raw.shape[0];
    Dry dry(dry_config, silent);
    size_t same_frames = 0;
    for (size_t f = 0; f < frames; f++) {
        std::vector<float> row(raw.f32.begin() + (std::ptrdiff_t) (f * vocab), raw.f32.begin() + (std::ptrdiff_t) ((f + 1) * vocab));
        shape_logits(row, dry, &silent_logits.f32[f * 2], text_ids, 0.0f, false);
        same_frames += std::equal(row.begin(), row.end(), sampled.f32.begin() + (std::ptrdiff_t) (f * vocab));
        dry.update(tokens.i32[f]);
    }
    std::printf("%-34s %zu of %zu frames equal bit for bit\n", "logits drawn from", same_frames, frames);
    ok = ok && same_frames == frames;

    Qwen2Pattern pattern;
    qwen2_pattern(model.str("haru.tokenizer.pre_tokenizer"), &pattern);
    const Qwen2Tokenizer tokenizer(model, "haru.tokenizer", pattern);
    StreamDetokenizer detokenizer(tokenizer, {text_ids.pad, text_ids.eop});
    const JsonValue meta = read_json(dump / "meta.json");
    const JsonValue & text = *meta.member("text");
    size_t same_text = 0;
    for (size_t f = 0; f < frames; f++) {
        const std::string got = detokenizer.push(tokens.i32[f]);
        if (got == text.items[f].text) {
            same_text++;
        } else {
            std::printf("  frame %zu: \"%s\" where the official text is \"%s\"\n", f, got.c_str(), text.items[f].text.c_str());
        }
    }
    std::printf("%-34s %zu of %zu frames' text equal\n", "streamed text", same_text, frames);
    ok = ok && same_text == frames;
    return ok ? 0 : 1;
}
