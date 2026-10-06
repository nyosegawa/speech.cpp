// Checks FastConformer's recognition times against NeMo's transcribe(timestamps=True) on each dump of
// reference/fastconformer/dump.py: the frame the decoding emitted each token on and, for TDT, the duration it
// predicted, from the dump's encoder output and from the dump's audio; each token's span in frames and in seconds;
// and the segments the file's separators give, without breaks, their spans and their texts. For a model whose file
// has breaks (the Japanese models) it then prints the recognizer's segments, with them, and checks that each ends in
// a break, at a separator that ends a word, or with the last token. The dumps are those of the model, in
// <reference out dir>/<its general.name>/.
//
// usage: fastconformer-times-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "args.h"
#include "backend.h"
#include "reference-dumps.h"
#include "fastconformer/recognizer.h"
#include "fastconformer/times.h"
#include "npy.h"

using namespace fastconformer;

namespace {

bool ends_with(const std::string & text, const std::vector<std::string> & marks) {
    for (const std::string & m : marks) {
        if (text.size() >= m.size() && text.compare(text.size() - m.size(), m.size(), m) == 0) return true;
    }
    return false;
}

/** The lines of a dump's segments.txt, one segment's text each. */
std::vector<std::string> read_lines(const std::filesystem::path & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path.u8string());
    std::vector<std::string> lines;
    for (std::string line; std::getline(f, line);) lines.push_back(line);
    return lines;
}

/**
 * A text without its spaces. NeMo joins a segment's words with one space where a token's text, as it stands in the
 * text, carries the space it follows, so the two are compared by their other characters.
 */
std::string without_spaces(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c != ' ') out += c;
    }
    return out;
}

/**
 * The spans as NeMo writes them. Its greedy TDT decoding records the frame a token was emitted on, as this one does;
 * its beam search records the step of the search, the frame plus the tokens emitted before it.
 */
std::vector<Span> nemo_spans(std::vector<Span> spans, bool steps) {
    for (size_t i = 0; steps && i < spans.size(); i++) {
        spans[i].start += (int64_t) i;
        spans[i].end += (int64_t) i;
    }
    return spans;
}

/**
 * The number of tokens of `d`, whose ids are NeMo's, emitted on another frame than NeMo's or, for TDT, with another
 * duration, printed with the largest difference of frame.
 */
size_t frames_off(const char * what, const Decoding & d, const Npy & timestep, const Npy * duration, bool steps) {
    size_t off = 0;
    int64_t largest = 0;
    for (size_t i = 0; i < d.frames.size(); i++) {
        const int64_t frame = steps ? timestep.i32[i] - (int64_t) i : timestep.i32[i];
        off += d.frames[i] != frame || (duration && d.durations[i] != duration->i32[i]);
        largest = std::max(largest, std::abs(d.frames[i] - frame));
    }
    const char * nemo = steps ? "NeMo's steps less the tokens before each" : "NeMo's";
    if (off == 0) {
        std::printf("  %s: frames%s equal to %s\n", what, duration ? " and durations" : "", nemo);
    } else {
        std::printf("  %s: %zu of %zu tokens differ from %s, the farthest by %lld frame%s\n", what, off, d.frames.size(), nemo,
                    (long long) largest, largest == 1 ? "" : "s");
    }
    return off;
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
        bool ok = true;
        {
            Recognizer recognizer(args[1], backend);
            const Detokenizer & detokenizer = recognizer.detokenizer();
            const bool tdt = recognizer.model().str("fastconformer.decoder.kind") == "tdt";
            // The file's separators are NeMo's, whose segments the dumps hold; its breaks are speech.cpp's own.
            const std::vector<std::string> separators = recognizer.model().str_array("fastconformer.segment.separators");
            const std::vector<std::string> breaks = recognizer.model().str_array("fastconformer.segment.breaks");
            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            for (const auto & d : reference_dumps(args[2], recognizer.model())) {
                const Npy encoded = read_npy((d / "encoded.npy").u8string());
                const Npy audio = read_npy((d / "audio.npy").u8string());
                const Npy want_ids = read_npy((d / "ids.npy").u8string());
                const Npy timestep = read_npy((d / "token_timestep.npy").u8string());
                const Npy token_offsets = read_npy((d / "token_offsets.npy").u8string());
                const Npy token_seconds = read_npy((d / "token_seconds.npy").u8string());
                const Npy segment_offsets = read_npy((d / "segment_offsets.npy").u8string());
                const Npy segment_seconds = read_npy((d / "segment_seconds.npy").u8string());
                const std::vector<std::string> want_segments = read_lines(d / "segments.txt");
                Npy duration;
                if (tdt) duration = read_npy((d / "token_duration.npy").u8string());
                const std::string want_text = dump_text(d / "text.txt");
                std::printf("%s (%.2f s, %zu tokens, %zu segments)\n", d.filename().u8string().c_str(),
                            (double) audio.f32.size() / recognizer.sample_rate(), want_ids.i32.size(), want_segments.size());

                // The decoding from the dump's encoder output.
                Graph e;
                ggml_tensor * projected = recognizer.joint().project_encoder(e.ctx(), e.input(encoded.f32, encoded.shape[1], encoded.shape[0]));
                e.output(projected);
                e.compute(backend, allocr);
                const Decoding decoding = recognizer.decoding(Graph::read(projected));
                if (decoding.ids != want_ids.i32) {
                    std::printf("  decoded ids DIFFER from NeMo's\n");
                    ok = false;
                    continue;
                }
                bool same = frames_off("from the encoder output", decoding, timestep, tdt ? &duration : nullptr, !tdt) == 0;
                if (!tdt && !decoding.frames.empty()) {
                    const int64_t past = (int64_t) decoding.frames.size() - 1;
                    std::printf("    NeMo's last token is %lld frames (%.2f s) past the frame it was emitted on\n", (long long) past,
                                recognizer.seconds(past));
                }

                // The spans of the tokens, in frames and in seconds.
                const std::vector<Span> spans = token_spans(decoding, detokenizer);
                const std::vector<Span> nemo = nemo_spans(spans, !tdt);
                size_t spans_differ = 0, seconds_differ = 0;
                for (size_t i = 0; i < nemo.size(); i++) {
                    spans_differ += nemo[i].start != token_offsets.i32[2 * i] || nemo[i].end != token_offsets.i32[2 * i + 1];
                    seconds_differ += recognizer.seconds(nemo[i].start) != token_seconds.f64[2 * i] ||
                                      recognizer.seconds(nemo[i].end) != token_seconds.f64[2 * i + 1];
                }
                std::printf("  token spans: %zu of %zu differ in frames, %zu in seconds\n", spans_differ, nemo.size(), seconds_differ);
                same = same && spans_differ == 0 && seconds_differ == 0;

                // The tokens' texts and the segments the separators give, as NeMo's.
                const std::vector<std::string> texts = detokenizer.token_texts(decoding.ids);
                const std::vector<bool> word_starts = detokenizer.word_starts(decoding.ids);
                std::string joined;
                for (const std::string & t : texts) joined += t;
                const std::vector<Segment> segs = segments(texts, word_starts, nemo, separators, {});
                bool same_segments = joined == want_text && segs.size() == want_segments.size();
                for (size_t k = 0; same_segments && k < segs.size(); k++) {
                    same_segments = segs[k].span.start == segment_offsets.i32[2 * k] && segs[k].span.end == segment_offsets.i32[2 * k + 1] &&
                                    recognizer.seconds(segs[k].span.start) == segment_seconds.f64[2 * k] &&
                                    recognizer.seconds(segs[k].span.end) == segment_seconds.f64[2 * k + 1] &&
                                    without_spaces(segs[k].text) == without_spaces(want_segments[k]);
                }
                std::printf("  token texts %s; segments with the file's separators %s\n", joined == want_text ? "make the text" : "DO NOT make the text",
                            same_segments ? "equal" : "DIFFER");
                if (!same_segments) {
                    for (const Segment & s : segs) std::printf("    got  %lld-%lld %s\n", (long long) s.span.start, (long long) s.span.end, s.text.c_str());
                    for (size_t k = 0; k < want_segments.size(); k++) {
                        std::printf("    want %d-%d %s\n", segment_offsets.i32[2 * k], segment_offsets.i32[2 * k + 1], want_segments[k].c_str());
                    }
                }
                ok = ok && same && same_segments;

                // The whole path from the dump's audio, whose ids must be NeMo's. Its frames are printed, not required:
                // the beam search keeps the frames of the better of two alignments of the same labels, and two can be
                // as good. Measured on an Apple M5 on 2026-10-06, the 17th token of reazonspeech-nemo-v2's
                // 9518252661993015549 is on frame 58 with a log-probability of -1.4191 and on frame 57 with -1.4258 on
                // the CPU in float32; from Metal's encoder with F16 weights, -1.4225 and -1.4210, and it goes on 57.
                const Transcript path = recognizer.recognize(audio.f32);
                if (path.decoding.ids == want_ids.i32) {
                    frames_off("from the audio", path.decoding, timestep, tdt ? &duration : nullptr, !tdt);
                } else {
                    std::printf("  from the audio: ids DIFFER from NeMo's\n");
                    ok = false;
                }

                // The recognizer's segments, with the breaks, at the frames the tokens were emitted on.
                if (!breaks.empty()) {
                    const std::vector<Segment> broken = recognizer.segments(decoding);
                    bool ends = true;
                    for (size_t k = 0; k + 1 < broken.size(); k++) {
                        ends = ends && (ends_with(broken[k].text, breaks) || (ends_with(broken[k].text, separators) && word_starts[broken[k].end]));
                    }
                    std::printf("  segments with breaks: %zu, each %s\n", broken.size(),
                                ends ? "ending in a break, at a separator that ends a word or with the last token" : "NOT ending where it should");
                    for (const Segment & s : broken) {
                        std::printf("    %8.2f %8.2f  %s\n", recognizer.seconds(s.span.start), recognizer.seconds(s.span.end), s.text.c_str());
                    }
                    ok = ok && ends;
                }
            }
            ggml_gallocr_free(allocr);
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
