// Checks Nemotron 3.5 Transcribe's CTC alignment against NeMo's on the dumps of reference/nemotron-transcribe/align.py,
// each stage from the dump's own inputs, which must agree exactly: the CTC head's SentencePiece encoding of
// ctc-tokenizer.json's texts and its ids of their characters; the words, units, token ids, streams and columns of
// ctc-text.json's texts, or that the alignment refuses the text; and for each ctc*/ folder of a case, from its transcript,
// its columns' log probabilities, its speaker probabilities and its 10 ms labels: the same text stages, the speaker
// probabilities on the CTC frames, the speaker column of each stream, each stream's searched frames, band and path, its
// words' spans from the path and after the refinement, and the public words rounded to the millisecond; or that the
// alignment fails where NeMo's raises or drops a stream. It reads the model file's keys alone, of any weight type. The
// dumps are those under <reference out dir>/<the file's general.name>/.
//
// usage: nemotron-transcribe-alignment-check <model.gguf> <reference out dir>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>

#include "args.h"
#include "json-reader.h"
#include "nemotron-transcribe/alignment.h"
#include "nemotron-transcribe/layout.h"
#include "npy.h"
#include "reference-dumps.h"

using namespace nemotron_transcribe;

namespace {

int64_t integer(const JsonValue & v) { return std::stoll(v.text); }
double number(const JsonValue & v) { return std::stod(v.text); }
/** A speaker tag of a dump: its number, -1 for None. */
int tag(const JsonValue & v) { return v.kind == JsonValue::Kind::Null ? -1 : (int) integer(v); }
std::vector<int64_t> integers(const JsonValue & v) {
    std::vector<int64_t> out;
    for (const JsonValue & x : v.items) out.push_back(integer(x));
    return out;
}

/** Counts the mismatches of one dump and prints the first few. */
struct Tally {
    int64_t mismatches = 0;
    void fail(const std::string & what) {
        if (mismatches++ < 12) std::printf("  mismatch: %s\n", what.c_str());
    }
};

/** Compares the text stages of `got` with NeMo's records of them: units, their ids, the streams and the columns. */
void compare_text(const AlignmentText & got, const JsonValue & units, const JsonValue * streams, const JsonValue * columns, Tally & t) {
    if (got.units.size() != units.items.size()) {
        t.fail(std::to_string(got.units.size()) + " units, NeMo " + std::to_string(units.items.size()));
        return;
    }
    for (size_t i = 0; i < got.units.size(); i++) {
        const Unit & u = got.units[i];
        const JsonValue & w = units.items[i];
        const JsonValue * unit = w.member("alignment_unit");
        const std::vector<int64_t> ids = integers(*w.member("token_ids"));
        if (u.text != w.member("word")->text || u.speaker != tag(*w.member("speaker_tag")) || u.character != (unit && unit->text == "character") ||
            std::vector<int64_t>(u.ids.begin(), u.ids.end()) != ids) {
            t.fail("unit " + std::to_string(i) + " " + u.text + ", NeMo " + w.member("word")->text);
        }
    }
    if (streams) {
        if (got.streams.size() != streams->items.size()) t.fail(std::to_string(got.streams.size()) + " streams, NeMo " + std::to_string(streams->items.size()));
        for (size_t s = 0; s < std::min(got.streams.size(), streams->items.size()); s++) {
            const JsonValue & n = streams->items[s];
            if (got.streams[s].speaker != tag(*n.member("speaker_tag")) ||
                std::vector<int64_t>(got.streams[s].labels.begin(), got.streams[s].labels.end()) != integers(*n.member("labels"))) {
                t.fail("stream " + std::to_string(s));
            }
        }
    }
    if (columns && std::vector<int64_t>(got.columns.begin(), got.columns.end()) != integers(*columns)) t.fail("columns");
}

/** The folders of the cases under `root` named ctc or ctc-<text>, sorted. */
std::vector<std::filesystem::path> alignment_dumps(const std::filesystem::path & root) {
    std::vector<std::filesystem::path> out;
    for (const auto & c : std::filesystem::directory_iterator(root)) {
        if (!c.is_directory()) continue;
        for (const auto & e : std::filesystem::directory_iterator(c.path())) {
            if (e.is_directory() && e.path().filename().u8string().rfind("ctc", 0) == 0 && std::filesystem::is_regular_file(e.path() / "meta.json")) out.push_back(e.path());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir>\n", args[0].c_str());
        return 2;
    }
    try {
        const ModelFile model(args[1], layout);
        const CtcText text(model);
        const Aligner aligner(model);
        const std::filesystem::path root = std::filesystem::u8path(args[2]) / std::filesystem::u8path(model.str("general.name"));
        int64_t mismatches = 0, cases = 0;

        if (std::filesystem::is_regular_file(root / "ctc-tokenizer.json")) {
            const JsonValue cases_json = parse_json(dump_text(root / "ctc-tokenizer.json"));
            Tally t;
            for (const JsonValue & c : cases_json.member("cases")->items) {
                const std::vector<int32_t> ids = text.tokenizer().encode(c.member("text")->text);
                if (std::vector<int64_t>(ids.begin(), ids.end()) != integers(*c.member("ids"))) t.fail("encoding of " + c.member("text")->text);
            }
            for (const JsonValue & c : cases_json.member("pieces")->items) {
                if (text.tokenizer().piece_to_id(c.member("text")->text) != integer(*c.member("id"))) t.fail("id of the piece " + c.member("text")->text);
            }
            const size_t n = cases_json.member("cases")->items.size() + cases_json.member("pieces")->items.size();
            std::printf("tokenizer: %zu cases, %lld mismatches\n", n, (long long) t.mismatches);
            mismatches += t.mismatches;
            cases += (int64_t) n;
        }

        const JsonValue texts = parse_json(dump_text(root / "ctc-text.json"));
        {
            Tally t;
            for (const JsonValue & c : texts.member("cases")->items) {
                const bool refused = c.member("error") != nullptr;
                try {
                    const AlignmentText got = text.prepare(c.member("text")->text);
                    if (refused) t.fail("the text " + c.member("text")->text + " is taken, which NeMo refuses: " + c.member("error")->member("message")->text);
                    else compare_text(got, *c.member("tokenized"), c.member("streams"), c.member("columns"), t);
                } catch (const AlignmentFailure & e) {
                    if (!refused) t.fail("the text " + c.member("text")->text + " is refused: " + e.what());
                }
            }
            std::printf("texts: %zu cases, %lld mismatches\n", texts.member("cases")->items.size(), (long long) t.mismatches);
            mismatches += t.mismatches;
            cases += (int64_t) texts.member("cases")->items.size();
        }

        for (const auto & dump : alignment_dumps(root)) {
            const std::string name = dump.parent_path().filename().u8string() + "/" + dump.filename().u8string();
            const JsonValue meta = parse_json(dump_text(dump / "meta.json"));
            const bool refused = meta.member("error") != nullptr;
            const JsonValue * failed = refused ? nullptr : meta.member("result")->member("alignment_diagnostics")->member("failed_streams");
            const bool nemo_fails = refused || !failed->items.empty();
            Tally t;
            std::string outcome;
            try {
                const AlignmentText got = text.prepare(meta.member("transcript")->text);
                if (!refused) compare_text(got, *meta.member("prepared")->member("words"), meta.member("text_stages")->member("streams"), meta.member("prepared")->member("columns"), t);
                AlignmentInputs in;
                in.frames = 2 * integer(*meta.member("valid_frames"));
                in.duration = number(*meta.member("duration"));
                if (std::filesystem::is_regular_file(dump / "columns.npy")) {
                    const Npy columns = read_npy((dump / "columns.npy").u8string());
                    in.frames = columns.shape[0];
                    in.log_probs = columns.f32;
                } else {
                    in.log_probs.assign((size_t) in.frames * got.columns.size(), 0.0f);
                }
                const std::filesystem::path probs = std::filesystem::is_regular_file(dump / "speaker_probs.npy") ? dump / "speaker_probs.npy" : dump.parent_path() / "ctc" / "speaker_probs.npy";
                const Npy speakers = read_npy(probs.u8string());
                in.speaker_probs = speakers.f32;
                in.speaker_frames = speakers.shape[0];
                in.speakers = (int) speakers.shape[1];
                const std::filesystem::path labels_path = std::filesystem::is_regular_file(dump / "diarization_labels.npy") ? dump / "diarization_labels.npy" : dump.parent_path() / "ctc" / "diarization_labels.npy";
                const Npy labels = read_npy(labels_path.u8string());
                in.labels = labels.u8;
                in.label_speakers = (int) labels.shape[0];
                in.label_frames = labels.shape[1];
                const Alignment al = *aligner.align(got, in);
                outcome = std::to_string(got.units.size()) + " units";
                if (nemo_fails) {
                    t.fail("aligned, where NeMo fails");
                } else {
                    // The speaker probabilities on the CTC frames.
                    const Npy speaker = read_npy((dump / "speaker.npy").u8string());
                    int64_t differ = 0;
                    for (size_t i = 0; i < speaker.f32.size() && i < al.speaker.size(); i++) differ += speaker.f32[i] != al.speaker[i];
                    if (differ || speaker.f32.size() != al.speaker.size()) t.fail(std::to_string(differ) + " resampled speaker probabilities");
                    // The mapping, each search, and the words' spans.
                    const JsonValue & mapping = *meta.member("mapping")->items[0].member("mapping");
                    for (size_t s = 0; s < got.streams.size(); s++) {
                        const int speaker_tag = got.streams[s].speaker;
                        const JsonValue * c = mapping.member(speaker_tag < 0 ? "null" : std::to_string(speaker_tag));
                        if (!c || tag(*c) != al.columns[s]) t.fail("the column of stream " + std::to_string(s));
                        const JsonValue & n = meta.member("searches")->items[s];
                        const StreamSearch & f = al.searches[s];
                        if (f.positions != integers(*n.member("positions"))) t.fail("the frames of stream " + std::to_string(s));
                        if (f.starts != integers(*n.member("starts"))) t.fail("the band of stream " + std::to_string(s));
                        if (f.path != integers(*n.member("path"))) t.fail("the path of stream " + std::to_string(s));
                        if ((double) f.score != number(*n.member("score"))) t.fail("the score of stream " + std::to_string(s));
                    }
                    auto compare_spans = [&](const JsonValue & by_tag, const std::vector<Span> & spans, bool sorted, const std::string & what) {
                        for (size_t s = 0; s < got.streams.size(); s++) {
                            const int speaker_tag = got.streams[s].speaker;
                            const JsonValue * words = by_tag.member(speaker_tag < 0 ? "null" : std::to_string(speaker_tag));
                            std::vector<size_t> order = got.streams[s].units;
                            if (sorted) std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) { return al.rows[x].start < al.rows[y].start; });
                            if (!words || words->items.size() != order.size()) {
                                t.fail(what + " of stream " + std::to_string(s));
                                continue;
                            }
                            for (size_t k = 0; k < order.size(); k++) {
                                const JsonValue & w = words->items[k];
                                if (spans[order[k]].start != number(*w.member("start")) || spans[order[k]].end != number(*w.member("end"))) {
                                    t.fail(what + " of " + got.units[order[k]].text + " of stream " + std::to_string(s));
                                }
                            }
                        }
                    };
                    compare_spans(*meta.member("refined")->items[0].member("before"), al.rows, false, "the span from the path");
                    compare_spans(*meta.member("result")->member("speaker_word_timestamps"), al.refined, true, "the refined span");
                    // The public words, every stream's in the order of their starts and ends.
                    struct Public {
                        std::string word, speaker;
                        double start, end;
                    };
                    std::vector<Public> words;
                    for (const Stream & s : got.streams) {
                        std::vector<size_t> order = s.units;
                        std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) { return al.rows[x].start < al.rows[y].start; });
                        for (size_t u : order) words.push_back({got.units[u].text, s.speaker < 0 ? "None" : std::to_string(s.speaker), al.times[u].start, al.times[u].end});
                    }
                    std::stable_sort(words.begin(), words.end(), [](const Public & a, const Public & b) { return a.start < b.start || (a.start == b.start && a.end < b.end); });
                    const JsonValue & expected = *meta.member("public")->member("words");
                    if (expected.items.size() != words.size()) t.fail(std::to_string(words.size()) + " public words, NeMo " + std::to_string(expected.items.size()));
                    for (size_t i = 0; i < std::min(words.size(), expected.items.size()); i++) {
                        const JsonValue & w = expected.items[i];
                        if (words[i].word != w.member("word")->text || words[i].speaker != w.member("speaker")->text || words[i].start != number(*w.member("start")) ||
                            words[i].end != number(*w.member("end"))) {
                            t.fail("public word " + std::to_string(i) + " " + words[i].word);
                        }
                    }
                }
            } catch (const AlignmentFailure & e) {
                outcome = std::string("fails: ") + e.what();
                if (!nemo_fails) t.fail(std::string("fails, where NeMo aligns: ") + e.what());
            }
            std::printf("%s: %s, %lld mismatches\n", name.c_str(), outcome.c_str(), (long long) t.mismatches);
            mismatches += t.mismatches;
            cases++;
        }
        const bool ok = mismatches == 0;
        std::printf("%s: %lld cases, %lld mismatches\n", ok ? "OK" : "FAIL", (long long) cases, (long long) mismatches);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
