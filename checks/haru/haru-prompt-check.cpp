// Checks Haru's prompt against the official haru.prompt, on the cases of reference/haru/prompt_cases.py and dumps of
// reference/haru/dump.py: each card's text and ids in each of the nine languages, the references prepare_reference()
// trims, and each dump's prompt rows, its tokens, its latents on either side and its masks, laid out from the dump's own
// card, reference latents and silence latent.
//
// usage: haru-prompt-check <model.gguf> <prompt cases dir> <dump dir>...

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "args.h"
#include "haru/layout.h"
#include "haru/prompt.h"
#include "json-reader.h"
#include "npy.h"

using namespace haru;

namespace {

JsonValue read_json(const std::filesystem::path & path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path.string());
    return parse_json(std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()));
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <prompt cases dir> <dump dir>...\n", args[0].c_str());
        return 2;
    }
    const ModelFile model(args[1], model_layout);
    const std::filesystem::path cases = std::filesystem::u8path(args[2]);
    Qwen2Pattern pattern;
    qwen2_pattern(model.str("haru.tokenizer.pre_tokenizer"), &pattern);
    const Qwen2Tokenizer tokenizer(model, "haru.tokenizer", pattern);
    const AddedVocabulary added(model, "haru.tokenizer");
    bool ok = true;

    std::ifstream lines(cases / "prompt-cases.jsonl");
    std::string line;
    int count = 0, same_text = 0, same_ids = 0;
    while (std::getline(lines, line)) {
        const JsonValue v = parse_json(line);
        std::map<std::string, std::string> fields;
        for (const auto & [name, value] : v.member("card")->members) fields[name] = value.text;
        const std::string text = render_card(model, v.member("language")->text, fields);
        const bool text_ok = text == v.member("text")->text;
        std::vector<int32_t> want;
        for (const JsonValue & id : v.member("ids")->items) want.push_back(std::stoi(id.text));
        const bool ids_ok = card_ids(model, tokenizer, added, text) == want;
        if (!text_ok || !ids_ok) std::printf("  %s card %d: %s\n", v.member("language")->text.c_str(), count, text_ok ? "ids differ" : "text differs");
        same_text += text_ok;
        same_ids += ids_ok;
        count++;
    }
    std::printf("%-34s %d of %d cards' text, %d of %d cards' ids\n", "cards", same_text, count, same_ids, count);
    ok = ok && count > 0 && same_text == count && same_ids == count;

    int references = 0, same_references = 0;
    for (int n = 0;; n++) {
        const std::filesystem::path in = cases / ("reference-" + std::to_string(n) + "-in.npy");
        if (!std::filesystem::exists(in)) break;
        const std::vector<float> got = prepare_reference(model, read_npy(in.u8string()).f32);
        const Npy want = read_npy((cases / ("reference-" + std::to_string(n) + "-out.npy")).u8string());
        const bool equal = got == want.f32;
        if (!equal) std::printf("  reference %d: %zu samples where the official keeps %zu\n", n, got.size(), want.f32.size());
        same_references += equal;
        references++;
    }
    std::printf("%-34s %d of %d equal\n", "references prepared", same_references, references);
    ok = ok && references > 0 && same_references == references;

    const TextIds ids = file_text_ids(model);
    const int64_t latent = model.size("haru.latent_dim"), frame = model.size("haru.frame_samples");
    for (size_t d = 3; d < args.size(); d++) {
        const std::filesystem::path dir = std::filesystem::u8path(args[d]);
        const JsonValue meta = read_json(dir / "meta.json");
        const std::vector<float> reference = read_npy((dir / "reference_latents.npy").u8string()).f32;
        const std::vector<float> silence = read_npy((dir / "silence_latent.npy").u8string()).f32;
        const bool speak_first = meta.member("speak_first")->boolean;
        const int64_t quietest = quietest_frame(read_npy((dir / "reference_audio.npy").u8string()).f32, frame);
        const std::vector<float> room = quietest >= 0 ? std::vector<float>(reference.begin() + quietest * latent, reference.begin() + (quietest + 1) * latent) : silence;
        const Prompt p = build_prompt(card_ids(model, tokenizer, added, meta.member("card")->text), reference, speak_first, room, silence, ids, latent);
        std::vector<float> model_mask, user_mask;
        for (int32_t m : read_npy((dir / "prompt_model_mask.npy").u8string()).i32) model_mask.push_back((float) m);
        for (int32_t m : read_npy((dir / "prompt_user_mask.npy").u8string()).i32) user_mask.push_back((float) m);
        const bool tokens_ok = p.tokens == read_npy((dir / "prompt_tokens.npy").u8string()).i32;
        const bool latents_ok = p.model_latents == read_npy((dir / "prompt_model.npy").u8string()).f32 &&
                                p.user_latents == read_npy((dir / "prompt_user.npy").u8string()).f32;
        const bool masks_ok = p.model_mask == model_mask && p.user_mask == user_mask;
        std::printf("%-34s %lld rows: tokens %s, latents %s, masks %s\n", ("prompt of " + dir.filename().u8string()).c_str(), (long long) p.rows(),
                    tokens_ok ? "equal" : "differ", latents_ok ? "equal" : "differ", masks_ok ? "equal" : "differ");
        ok = ok && tokens_ok && latents_ok && masks_ok;
    }
    return ok ? 0 : 1;
}
