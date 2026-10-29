#include "timing.h"

#include <algorithm>

#include "sentence-segments.h"
#include "unicode.h"
#include "utf8.h"

namespace nemotron_transcribe {

namespace {

const std::string kKeys = "nemotron-transcribe.";

/** The speaker branch's 10 ms labels: its probabilities over 0.5, as ParallelExpertEncoder._build_ctc_timestamp_inputs() takes them. */
constexpr float kLabelThreshold = 0.5f;

}  // namespace

Timing::Timing(const ModelFile & m, ggml_backend_t backend)
    : text_(m),
      head_(m, backend),
      aligner_(m),
      separators_(m.str_array(kKeys + "segment.separators")),
      breaks_(m.str_array(kKeys + "segment.breaks")),
      sample_rate_((int) m.u32("speech.sample_rate")) {}

std::vector<bool> Timing::kept(const std::string & transcript, bool speakers) const {
    std::vector<bool> keep(transcript.size(), true);
    if (!speakers) return keep;
    const std::vector<uint32_t> cps = decode_utf8(transcript).value_or(std::vector<uint32_t>());
    std::vector<size_t> offset(cps.size() + 1, 0);
    for (size_t i = 0; i < cps.size(); i++) offset[i + 1] = offset[i] + encode_utf8({cps[i]}).size();
    for (const auto & [begin, end] : text_.speaker_tags(transcript)) {
        size_t stop = end;
        size_t i = (size_t) (std::lower_bound(offset.begin(), offset.end(), end) - offset.begin());
        while (i < cps.size() && python_space(cps[i])) stop = offset[++i];
        std::fill(keep.begin() + (std::ptrdiff_t) begin, keep.begin() + (std::ptrdiff_t) stop, false);
    }
    return keep;
}

std::string Timing::untagged(const std::string & transcript) const {
    std::string out;
    try {
        const std::vector<bool> keep = kept(transcript, true);
        for (size_t i = 0; i < transcript.size(); i++) {
            if (keep[i]) out += transcript[i];
        }
    } catch (const AlignmentFailure &) {
        return transcript;
    }
    return out;
}

TimedTranscript Timing::time(const std::string & transcript, const AlignmentFrames & frames, size_t samples, bool speakers, Alignment * alignment) {
    TimedTranscript out;
    std::vector<bool> keep;
    try {
        keep = kept(transcript, speakers);
    } catch (const AlignmentFailure & e) {
        out.text = transcript;
        out.failure = e.what();
        return out;
    }
    auto text_of = [&](size_t begin, size_t end) {
        std::string t;
        for (size_t i = begin; i < end; i++) {
            if (keep[i]) t += transcript[i];
        }
        return t;
    };
    out.text = text_of(0, transcript.size());
    try {
        const AlignmentText at = text_.prepare(transcript);
        if (at.units.empty()) {
            if (!out.text.empty()) throw AlignmentFailure("the text holds no word to time");
            return out;
        }
        AlignmentInputs in;
        in.frames = frames.frames * head_.upsampling();
        in.log_probs = head_.log_probs(frames.asr, frames.frames, at.columns);
        in.speaker_frames = frames.frames;
        in.speakers = frames.frames > 0 ? (int) (frames.speakers.size() / (size_t) frames.frames) : 0;
        in.speaker_probs = frames.speakers;
        in.label_speakers = in.speakers;
        in.label_frames = frames.fine_frames;
        in.labels.resize((size_t) (in.label_speakers * in.label_frames));
        for (int64_t t = 0; t < in.label_frames; t++) {
            for (int s = 0; s < in.label_speakers; s++) {
                in.labels[(size_t) (s * in.label_frames + t)] = frames.fine[(size_t) (t * in.label_speakers + s)] > kLabelThreshold;
            }
        }
        in.duration = (double) samples / (double) sample_rate_;
        Alignment al = aligner_.align(at, in);

        std::vector<std::string> texts;
        std::vector<bool> word_starts, turns;
        for (size_t i = 0; i < at.units.size(); i++) {
            const Unit & u = at.units[i];
            out.tokens.push_back({al.times[i].start, al.times[i].end, text_of(u.token_begin, u.token_end), u.speaker});
            texts.push_back(out.tokens.back().text);
            word_starts.push_back(i == 0 || at.units[i - 1].word != u.word);
            turns.push_back(i > 0 && at.units[i - 1].turn != u.turn);
        }
        size_t first = 0;
        for (size_t end : segment_ends(texts, word_starts, separators_, breaks_, turns)) {
            TimedText s{out.tokens[first].start, out.tokens[end - 1].end, "", speakers ? at.units[first].speaker : -1};
            for (size_t k = first; k < end; k++) s.text += texts[k];
            out.segments.push_back(std::move(s));
            first = end;
        }
        if (alignment) *alignment = std::move(al);
    } catch (const AlignmentFailure & e) {
        out.tokens.clear();
        out.segments.clear();
        out.failure = e.what();
    }
    return out;
}

}  // namespace nemotron_transcribe
