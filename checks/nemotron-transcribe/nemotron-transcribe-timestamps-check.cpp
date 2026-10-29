// Checks Nemotron 3.5 Transcribe's timestamps against NeMo's on the dumps of reference/nemotron-transcribe/align.py from
// the audio on: the perception module's frames, the CTC head and the alignment, each computed here from the stage
// before, of the transcript the official model wrote, or of the one align.py put in its place. It compares the public
// words, every speaker's by start, rounded to the millisecond, and counts those whose start or end differs; and it checks
// that the alignment fails where NeMo's fails, and that the tokens' texts and the segments' texts each join into the
// text, which holds no speaker tag where speakers are asked, as they are of a transcript with tags. The stages' own
// checks compare each from the dump's inputs; here the error of each reaches the next, and a word may move by a frame
// where two paths score alike. The dumps are those under <reference out dir>/<the file's general.name>/.
//
// usage: nemotron-transcribe-timestamps-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>

#include "args.h"
#include "backend.h"
#include "json-reader.h"
#include "nemotron-transcribe/layout.h"
#include "nemotron-transcribe/perception.h"
#include "nemotron-transcribe/timing.h"
#include "npy.h"
#include "reference-dumps.h"
#include "weight-types.h"

using namespace nemotron_transcribe;

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    try {
        if (!takes_weights(args[1], layout, {"F32"}, "the bounds are set from F32's arithmetic")) return 2;
        start_ggml();
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "auto");
        const bool cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        std::printf("device: %s\n", ggml_backend_name(backend));
        bool ok = true;
        int64_t words_all = 0, moved_all = 0;
        {
            const ModelFile model(args[1], backend, layout, [](const std::string & name) {
                for (const char * prefix : {"frontend.", "asr.", "spk.", "fuse.", "proj.", "ctc."}) {
                    if (name.rfind(prefix, 0) == 0) return true;
                }
                return false;
            });
            Perception perception(model, backend);
            Timing timing(model, backend);
            const std::filesystem::path root = std::filesystem::u8path(args[2]) / std::filesystem::u8path(model.str("general.name"));
            for (const auto & dump : reference_dumps(args[2], model)) {
                const Npy audio = read_npy((dump / "audio.npy").u8string());
                AlignmentFrames frames;
                perception.encode(audio.f32, {}, nullptr, &frames);
                const JsonValue dumped = parse_json(dump_text(dump / "meta.json"));
                for (const auto & e : std::filesystem::directory_iterator(dump)) {
                    if (!e.is_directory() || e.path().filename().u8string().rfind("ctc", 0) != 0) continue;
                    const JsonValue meta = parse_json(dump_text(e.path() / "meta.json"));
                    // The model's own text, special tokens kept, as the recognizer decodes it, or align.py's in its place.
                    const bool changed = meta.member("transcript_change")->kind != JsonValue::Kind::Null;
                    const std::string transcript = changed ? meta.member("transcript")->text : dumped.member("text")->text;
                    const bool speakers = transcript.find("<spk:") != std::string::npos;
                    Alignment al;
                    const TimedTranscript timed = timing.time(transcript, frames, audio.f32.size(), speakers, &al);
                    const std::string name = dump.filename().u8string() + "/" + e.path().filename().u8string();
                    const bool refused = meta.member("error") != nullptr;
                    const bool nemo_fails = refused || !meta.member("result")->member("alignment_diagnostics")->member("failed_streams")->items.empty();
                    std::string joined_tokens, joined_segments;
                    for (const TimedText & t : timed.tokens) joined_tokens += t.text;
                    for (const TimedText & s : timed.segments) joined_segments += s.text;
                    if (!timed.failure.empty()) {
                        std::printf("%s: fails: %s\n", name.c_str(), timed.failure.c_str());
                        if (!nemo_fails) ok = false, std::printf("  FAIL: NeMo aligns it\n");
                        continue;
                    }
                    if (nemo_fails) {
                        ok = false;
                        std::printf("%s: FAIL: aligned, where NeMo fails\n", name.c_str());
                        continue;
                    }
                    if (joined_tokens != timed.text || joined_segments != timed.text || (speakers && timed.text.find("<spk:") != std::string::npos)) {
                        ok = false;
                        std::printf("%s: FAIL: the tokens or the segments do not join into the text\n", name.c_str());
                    }
                    // The public words: every stream's by start, then all by start and end.
                    struct Public {
                        std::string word;
                        double start, end;
                    };
                    std::vector<Public> words;
                    const AlignmentText at = CtcText(model).prepare(transcript);
                    for (const Stream & s : at.streams) {
                        std::vector<size_t> order = s.units;
                        std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) { return al.rows[x].start < al.rows[y].start; });
                        for (size_t u : order) words.push_back({at.units[u].text, al.times[u].start, al.times[u].end});
                    }
                    std::stable_sort(words.begin(), words.end(), [](const Public & a, const Public & b) { return a.start < b.start || (a.start == b.start && a.end < b.end); });
                    const JsonValue & expected = *meta.member("public")->member("words");
                    int64_t moved = 0;
                    if (expected.items.size() != words.size()) {
                        ok = false;
                        std::printf("%s: FAIL: %zu words, NeMo %zu\n", name.c_str(), words.size(), expected.items.size());
                        continue;
                    }
                    for (size_t i = 0; i < words.size(); i++) {
                        const JsonValue & w = expected.items[i];
                        moved += words[i].word != w.member("word")->text || words[i].start != std::stod(w.member("start")->text) ||
                                 words[i].end != std::stod(w.member("end")->text);
                    }
                    std::printf("%s: %zu words, %lld differ, %zu segments\n", name.c_str(), words.size(), (long long) moved, timed.segments.size());
                    words_all += (int64_t) words.size();
                    moved_all += moved;
                }
            }
        }
        // The share of words that may move: the stages' rounding on the CPU, and a GPU's half-precision matrix inputs.
        const double share = words_all ? (double) moved_all / (double) words_all : 0, bound = cpu ? 0.01 : 0.05;
        ok = ok && share <= bound;
        std::printf("%s: %lld of %lld words differ, %.4f against a bound of %.2f\n", ok ? "OK" : "FAIL", (long long) moved_all, (long long) words_all, share, bound);
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
