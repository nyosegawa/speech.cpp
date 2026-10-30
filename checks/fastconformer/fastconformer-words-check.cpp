// Checks Haas-ASR's custom vocabulary against each dump of reference/fastconformer/dump.py --words, in the order data
// flows: the spellings of each vocabulary of a dump's words/ and their token sequences, the boosting tree compiled
// from them, the decodings without and with the words from the dump's encoder output, on the CPU's copy of the
// decoding's tensors and on the model's device, and what Haas-ASR makes of their texts (whether the audio has speech,
// the guarded text and the text with the words in their written forms); then the whole recognition from the dump's
// audio. Token ids, the tree and the texts must be NeMo's and Haas-ASR's exactly.
//
// usage: fastconformer-words-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "args.h"
#include "backend.h"
#include "fastconformer/guard.h"
#include "fastconformer/layout.h"
#include "fastconformer/recognizer.h"
#include "json-reader.h"
#include "npy.h"
#include "reference-dumps.h"
#include "weight-types.h"

using namespace fastconformer;

namespace {

std::string read_file(const std::filesystem::path & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path.u8string());
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

std::vector<std::string> lines(const std::string & text) {
    std::vector<std::string> out;
    std::stringstream s(text);
    for (std::string line; std::getline(s, line);) out.push_back(line);
    return out;
}

/** The member `name` of a JSON object of the dump, which it must hold. */
const JsonValue & at(const JsonValue & object, const std::string & name) {
    const JsonValue * v = object.member(name);
    if (!v) throw std::runtime_error("a meta.json of dump.py --words lacks " + name);
    return *v;
}

std::string ids_text(const std::vector<int32_t> & ids) {
    std::string s;
    for (int32_t id : ids) s += " " + std::to_string(id);
    return s;
}

/** Whether the tree's arcs and states are the dump's, each value exactly. */
bool same_tree(const BoostingTree & tree, const std::filesystem::path & w) {
    const Npy arcs = read_npy((w / "tree_arcs.npy").u8string()), weights = read_npy((w / "tree_weights.npy").u8string());
    const Npy states = read_npy((w / "tree_states.npy").u8string()), backoffs = read_npy((w / "tree_backoffs.npy").u8string());
    if (arcs.i32.size() != tree.arcs().size() * 3 || states.i32.size() != tree.states().size() * 4) return false;
    for (size_t a = 0; a < tree.arcs().size(); a++) {
        const BoostingTree::Arc & arc = tree.arcs()[a];
        if (arc.from != arcs.i32[3 * a] || arc.to != arcs.i32[3 * a + 1] || arc.token != arcs.i32[3 * a + 2] || arc.weight != weights.f32[a]) return false;
    }
    for (size_t s = 0; s < tree.states().size(); s++) {
        const BoostingTree::State & st = tree.states()[s];
        const int32_t * want = &states.i32[4 * s];
        if (st.arcs_start != want[0] || st.arcs_end != want[1] || st.order != want[2] || st.backoff_to != want[3] || st.backoff_weight != backoffs.f32[s]) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    try {
        if (!takes_weights(args[1], layout, {"F32", "F16"}, "it wants NeMo's tokens, which other weights can change")) return 2;
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
        std::printf("backend: %s\n", ggml_backend_name(backend));
        bool ok = true;
        size_t cases = 0;
        {
            Recognizer recognizer(args[1], backend, default_threads());
            const WordsModel * words = recognizer.words();
            if (!words) throw std::runtime_error(args[1] + " offers no custom vocabulary");
            const std::string name = recognizer.decodings().front();
            std::vector<DecodeOn> places{DecodeOn::Cpu};
            if (!recognizer.on_cpu()) places.push_back(DecodeOn::Device);
            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            for (const auto & d : reference_dumps(args[2], recognizer.model())) {
                if (!std::filesystem::is_directory(d / "words")) continue;
                const Npy encoded = read_npy((d / "encoded.npy").u8string());
                const std::vector<float> audio = read_npy((d / "audio.npy").u8string()).f32;
                Graph e;
                ggml_tensor * projected = recognizer.joint().project_encoder(e.ctx(), e.input(encoded.f32, encoded.shape[1], encoded.shape[0]));
                e.output(projected);
                e.compute(backend, allocr);
                const std::vector<float> dump_projected = Graph::read(projected);
                for (const auto & entry : std::filesystem::directory_iterator(d / "words")) {
                    const std::filesystem::path w = entry.path();
                    const JsonValue meta = parse_json(read_file(w / "meta.json"));
                    const JsonValue & v = at(meta, "vocabulary");
                    std::vector<std::string> entries;
                    for (const JsonValue & word : at(v, "words").items) entries.push_back(word.text);
                    const Vocabulary vocabulary(entries, std::stof(at(v, "alpha").text));
                    const bool guard = at(v, "guard").boolean;
                    cases++;
                    std::printf("%s %s\n", d.filename().u8string().c_str(), w.filename().u8string().c_str());
                    // The spellings and their token sequences.
                    const bool spellings = vocabulary.spellings() == lines(read_file(w / "spellings.txt"));
                    const std::vector<int32_t> want_tokens = read_npy((w / "word_tokens.npy").u8string()).i32;
                    const std::vector<int32_t> offsets = read_npy((w / "word_offsets.npy").u8string()).i32;
                    const std::vector<int32_t> of_spelling = read_npy((w / "word_spellings.npy").u8string()).i32;
                    std::vector<int32_t> tokens, got_offsets = {0}, got_spellings;
                    for (size_t s = 0; s < vocabulary.spellings().size(); s++) {
                        for (const std::vector<int32_t> & form : words->word_forms(vocabulary.spellings()[s])) {
                            tokens.insert(tokens.end(), form.begin(), form.end());
                            got_offsets.push_back((int32_t) tokens.size());
                            got_spellings.push_back((int32_t) s);
                        }
                    }
                    const bool forms = tokens == want_tokens && got_offsets == offsets && got_spellings == of_spelling;
                    const BoostingTree tree = words->tree(vocabulary);
                    const bool same = same_tree(tree, w);
                    std::printf("  spellings %s, word forms %s, tree %s (%zu states, %zu arcs)\n", spellings ? "equal" : "DIFFER", forms ? "equal" : "DIFFER",
                                same ? "equal" : "DIFFERS", tree.states().size(), tree.arcs().size());
                    ok = ok && spellings && forms && same;
                    // The two decodings from the dump's encoder output.
                    const std::vector<int32_t> want_plain = read_npy((w / "plain_ids.npy").u8string()).i32;
                    const std::vector<int32_t> want_boosted = read_npy((w / "boosted_ids.npy").u8string()).i32;
                    for (const DecodeOn on : places) {
                        const Decoding plain = recognizer.decoding(dump_projected, name, on);
                        const Decoding boosted = recognizer.decoding(dump_projected, name, on, {}, Boost{tree, vocabulary.alpha()});
                        const bool eq = plain.ids == want_plain && boosted.ids == want_boosted;
                        std::printf("  decodings on the %s: plain %s, boosted %s\n", on == DecodeOn::Cpu ? "cpu" : "device",
                                    plain.ids == want_plain ? "equal" : "DIFFER", boosted.ids == want_boosted ? "equal" : "DIFFER");
                        if (boosted.ids != want_boosted) std::printf("    got%s\n    want%s\n", ids_text(boosted.ids).c_str(), ids_text(want_boosted).c_str());
                        ok = ok && eq;
                    }
                    // What Haas-ASR makes of the texts, from the dump's encoder output, then from its audio.
                    const bool speech = has_speech(audio, words->speech_rms(), (size_t) words->speech_frame());
                    const std::string plain = at(meta, "plain").text, boosted = at(meta, "boosted").text, want = at(meta, "text").text;
                    bool texts = speech == at(meta, "has_speech").boolean;
                    if (guard) texts = texts && guarded(plain, boosted, vocabulary) == at(meta, "guarded").text;
                    for (const DecodeOn on : places) {
                        const WordsText found = words->recognize(recognizer, vocabulary, guard, audio, dump_projected, name, on, {});
                        texts = texts && found.plain == plain && (!speech || found.boosted == boosted) && found.text == want;
                        if (found.text != want) std::printf("    got  %s\n    want %s\n", found.text.c_str(), want.c_str());
                    }
                    const std::vector<float> from_audio = recognizer.encode(recognizer.frontend().features(audio), recognizer.frontend().frames(audio.size()));
                    const WordsText heard = words->recognize(recognizer, vocabulary, guard, audio, from_audio, name, DecodeOn::Cpu, {});
                    std::printf("  speech %s, guarded and written %s; from the audio: %s\n", speech ? "yes" : "no", texts ? "equal" : "DIFFER",
                                heard.text == want ? "equal" : "DIFFERS");
                    ok = ok && texts && heard.text == want;
                }
            }
            ggml_gallocr_free(allocr);
        }
        ggml_backend_free(backend);
        if (cases == 0) throw std::runtime_error("no dump of dump.py --words is under " + args[2]);
        std::printf("%zu vocabularies of dumps: %s\n", cases, ok ? "pass" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
