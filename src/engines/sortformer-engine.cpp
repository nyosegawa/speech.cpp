#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "engine.h"
#include "sortformer/diarizer.h"

namespace {

/**
 * The most frames a streaming setting takes, 2^20 (23 hours of 80 ms frames), so that every index into a step's
 * frames, their features and their probabilities stays within the range of an int64 by far, whatever the audio's length.
 */
constexpr double kLongestSetting = 1 << 20;

/** The streaming configuration of a request: the mode it sets, or `default_mode`, with each setting it gives. */
sortformer::Streaming streaming_of(const sortformer::Diarizer & diarizer, const RequestValues & values, const std::string & default_mode) {
    sortformer::Streaming s = diarizer.mode(values.has(SPEECH_OPT_STREAMING_MODE) ? values.string(SPEECH_OPT_STREAMING_MODE) : default_mode);
    const auto take = [&](speech_option option, int & setting) {
        if (values.has(option)) setting = (int) values.integer(option);
    };
    take(SPEECH_OPT_SPKCACHE_LEN, s.spkcache_len);
    take(SPEECH_OPT_FIFO_LEN, s.fifo_len);
    take(SPEECH_OPT_CHUNK_LEN, s.chunk_len);
    take(SPEECH_OPT_CHUNK_RIGHT_CONTEXT, s.chunk_right_context);
    take(SPEECH_OPT_SPKCACHE_UPDATE_PERIOD, s.spkcache_update_period);
    return s;
}

sortformer::Postprocessing postprocessing_of(const RequestValues & values) {
    return {values.number(SPEECH_OPT_ONSET),           values.number(SPEECH_OPT_OFFSET),          values.number(SPEECH_OPT_PAD_ONSET),
            values.number(SPEECH_OPT_PAD_OFFSET),      values.number(SPEECH_OPT_MIN_DURATION_ON), values.number(SPEECH_OPT_MIN_DURATION_OFF)};
}

SpeakerSegment speaker_segment(const sortformer::Segment & s) {
    return {s.start, s.end, s.speaker};
}

/** The chunks computed as their audio arrives and the segments of their probabilities given as each becomes final. */
class SortformerStream : public DiarizationStream {
public:
    SortformerStream(sortformer::Diarizer & diarizer, const sortformer::Streaming & streaming, const sortformer::Postprocessing & post)
        : chunks_(diarizer, streaming), segments_(diarizer.speakers(), diarizer.frame_seconds(), post), speakers_(diarizer.speakers()) {}

    void push(const float * samples, size_t n) override {
        probs_.clear();
        chunks_.push(samples, n, probs_);
        segments_.push(probs_.data(), (int64_t) probs_.size() / speakers_);
        collect();
    }

    void end() override {
        probs_.clear();
        chunks_.end(probs_);
        segments_.push(probs_.data(), (int64_t) probs_.size() / speakers_);
        segments_.end();
        collect();
    }

    const std::vector<SpeakerSegment> & segments() const override { return given_; }
    double settled() const override { return segments_.settled(); }
    std::optional<double> speaking(int speaker) const override { return segments_.speaking(speaker); }

private:
    void collect() {
        for (size_t i = given_.size(); i < segments_.given().size(); i++) given_.push_back(speaker_segment(segments_.given()[i]));
    }

    sortformer::ChunkStream chunks_;
    sortformer::SegmentStream segments_;
    int speakers_;
    std::vector<float> probs_;
    std::vector<SpeakerSegment> given_;
};

class SortformerEngine : public Engine {
public:
    SortformerEngine(const std::string & path, ggml_backend_t backend)
        : diarizer_(path, backend), stream_mode_(diarizer_.model().str("sortformer.streaming.stream_default")) {}

    /**
     * The audio goes to the chunks a step's audio at a time, which gives what the whole audio gives at once, so that the
     * steps report their progress and a cancellation stops the request between two of them.
     */
    std::vector<SpeakerSegment> diarize(const std::vector<float> & samples, const RequestValues & values, Run & run) override {
        const sortformer::Streaming streaming = streaming_of(diarizer_, values, "offline");
        const sortformer::Postprocessing post = postprocessing_of(values);
        const int64_t chunk = (int64_t) streaming.chunk_len * diarizer_.network().factor();
        const int64_t steps = (diarizer_.frontend().frames(samples.size()) + chunk - 1) / chunk;
        sortformer::ChunkStream stream(diarizer_, streaming, [&](int64_t step) { run.progress((double) (step + 1) / (double) steps); });
        const size_t piece = (size_t) streaming.chunk_len * diarizer_.network().factor() * diarizer_.frontend().hop_length();
        std::vector<float> probs;
        for (size_t at = 0; at < samples.size(); at += piece) {
            stream.push(samples.data() + at, std::min(piece, samples.size() - at), probs);
            if (run.stopped()) return {};
        }
        stream.end(probs);
        if (run.stopped()) return {};
        std::vector<sortformer::Segment> found = sortformer::segments(probs, diarizer_.speakers(), diarizer_.frame_seconds(), post);
        std::stable_sort(found.begin(), found.end(), [](const sortformer::Segment & a, const sortformer::Segment & b) {
            return a.start < b.start || (a.start == b.start && a.speaker < b.speaker);
        });
        std::vector<SpeakerSegment> out;
        for (const sortformer::Segment & s : found) out.push_back(speaker_segment(s));
        return out;
    }

    std::unique_ptr<DiarizationStream> start_diarization(const RequestValues & values) override {
        return std::make_unique<SortformerStream>(diarizer_, streaming_of(diarizer_, values, stream_mode_), postprocessing_of(values));
    }

    void warm_up() override {
        // The first graph builds Metal's pipelines and compiles Vulkan's shaders.
        sortformer::ChunkStream stream(diarizer_, diarizer_.mode(stream_mode_));
        const std::vector<float> silence((size_t) diarizer_.sample_rate(), 0.0f);
        std::vector<float> probs;
        stream.push(silence.data(), silence.size(), probs);
        stream.end(probs);
    }

private:
    sortformer::Diarizer diarizer_;
    std::string stream_mode_;
};

}  // namespace

/**
 * The table of the options Sortformer takes: the streaming configuration, by the name of one of the model card's or by
 * NeMo's settings, and the postprocessing of diarize(), under its names and with its defaults, which the file gives. The
 * streaming configuration has no default, since a request of a whole recording takes the offline one and a stream the
 * file's mode for streams. The model takes no language, whose neutral value every model accepts.
 */
FamilyInfo describe_sortformer(const std::shared_ptr<const ModelFile> & file) {
    const ModelFile & m = *file;
    const std::string p = "sortformer.";
    FamilyInfo info;
    info.speakers = (int) m.u32(p + "speakers");
    const double least_cache = (double) ((1 + m.u32(p + "cache.silence_frames")) * m.u32(p + "speakers"));
    const auto post = [&](const char * key) { return m.f64(p + "postprocessing." + key); };
    info.options = {
        {SPEECH_OPT_STREAMING_MODE, false, true, std::nullopt, -INFINITY, INFINITY, false, m.str_array(p + "streaming.modes")},
        {SPEECH_OPT_SPKCACHE_LEN, false, true, std::nullopt, least_cache, kLongestSetting},
        {SPEECH_OPT_FIFO_LEN, false, true, std::nullopt, 0, kLongestSetting},
        {SPEECH_OPT_CHUNK_LEN, false, true, std::nullopt, 1, kLongestSetting},
        {SPEECH_OPT_CHUNK_RIGHT_CONTEXT, false, true, std::nullopt, 0, kLongestSetting},
        {SPEECH_OPT_SPKCACHE_UPDATE_PERIOD, false, true, std::nullopt, 1, kLongestSetting},
        {SPEECH_OPT_ONSET, false, true, post("onset"), 0, 1},
        {SPEECH_OPT_OFFSET, false, true, post("offset"), 0, 1},
        {SPEECH_OPT_PAD_ONSET, false, true, post("pad_onset"), 0},
        {SPEECH_OPT_PAD_OFFSET, false, true, post("pad_offset"), 0},
        {SPEECH_OPT_MIN_DURATION_ON, false, true, post("min_duration_on"), 0},
        {SPEECH_OPT_MIN_DURATION_OFF, false, true, post("min_duration_off"), 0},
    };
    return info;
}

std::unique_ptr<Engine> load_sortformer(const std::string & path, ggml_backend_t backend, int) {
    return std::make_unique<SortformerEngine>(path, backend);
}
