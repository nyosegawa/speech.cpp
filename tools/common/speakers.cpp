#include "speakers.h"

#include <algorithm>
#include <cmath>
#include <map>

#include "json.h"

Request diarization_request(speech_model * diarizer, const std::vector<float> & samples, int rate, const std::vector<RequestOption> & options) {
    Request request = new_request(diarizer);
    check(speech_request_set_audio(request.get(), samples.data(), samples.size(), rate));
    apply_options(request.get(), options);
    return request;
}

std::optional<std::vector<SpeakerSegment>> diarize_segments(speech_request * diarization, Cancellation & cancellation) {
    if (check(cancellation.run(diarization, [&] { return speech_diarize(diarization); })) == SPEECH_CANCELLED) return std::nullopt;
    const speech_result * result = speech_request_result(diarization);
    std::vector<SpeakerSegment> segments(speech_result_segment_count(result));
    for (size_t i = 0; i < segments.size(); i++) {
        check(speech_result_segment(result, i, &segments[i].start, &segments[i].end, nullptr));
        segments[i].speaker = speech_result_segment_speaker(result, i);
    }
    return segments;
}

std::vector<SpeakerSegment> speakers_within(const std::vector<SpeakerSegment> & segments, double start, double end) {
    std::vector<SpeakerSegment> out;
    for (const SpeakerSegment & s : segments) {
        if (s.end > start && s.start < end) out.push_back({std::max(s.start, start), std::min(s.end, end), s.speaker});
    }
    std::sort(out.begin(), out.end(), [](const SpeakerSegment & a, const SpeakerSegment & b) {
        return a.start < b.start || (a.start == b.start && a.speaker < b.speaker);
    });
    return out;
}

int main_speaker(const std::vector<SpeakerSegment> & within) {
    std::map<int, double> spoken;
    for (const SpeakerSegment & s : within) spoken[s.speaker] += s.end - s.start;
    int main = -1;
    double longest = 0;
    for (const auto & [speaker, seconds] : spoken) {
        if (seconds > longest) {
            main = speaker;
            longest = seconds;
        }
    }
    return main;
}

std::string speaker_label(int speaker) {
    std::string label;
    for (int n = speaker + 1; n > 0; n = (n - 1) / 26) label.insert(label.begin(), char('A' + (n - 1) % 26));
    return label;
}

std::string speakers_json(const std::vector<SpeakerSegment> & segments) {
    const auto time = [](double t) { return json_number(std::round(t * 1000) / 1000); };
    std::string out = "[";
    for (size_t i = 0; i < segments.size(); i++) {
        out += std::string(i ? "," : "") + "{\"speaker\":" + std::to_string(segments[i].speaker) + ",\"start\":" + time(segments[i].start) +
               ",\"end\":" + time(segments[i].end) + "}";
    }
    return out + "]";
}

Diarization::Diarization(speech_model * model, std::shared_ptr<const void> keep, std::vector<RequestOption> options, int rate)
    : keep_(std::move(keep)), options_(std::move(options)) {
    const Request request = new_request(model);
    apply_options(request.get(), options_);
    speech_diarization * raw = nullptr;
    check(speech_diarization_start(request.get(), rate, &raw));
    diarization_.reset(raw);
    speakers_ = speech_model_info_speakers(model_info(model).get());
}

void Diarization::push(const float * samples, size_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failed_) return;
    try {
        check(speech_diarization_push(diarization_.get(), samples, n));
    } catch (...) {
        failed_ = true;
        throw;
    }
    collect();
}

void Diarization::end() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failed_) return;
    try {
        check(speech_diarization_end(diarization_.get()));
    } catch (...) {
        failed_ = true;
        throw;
    }
    collect();
}

void Diarization::collect() {
    for (size_t i = given_.size(); i < speech_diarization_segment_count(diarization_.get()); i++) {
        SpeakerSegment s;
        check(speech_diarization_segment(diarization_.get(), i, &s.start, &s.end, &s.speaker));
        given_.push_back(s);
    }
}

bool Diarization::covers(double seconds) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return failed_ || speech_diarization_settled(diarization_.get()) >= seconds;
}

std::vector<SpeakerSegment> Diarization::segments() const {
    std::vector<SpeakerSegment> segments = given_;
    // A segment under way reaches the settled time at least, which within() is asked only up to.
    const double settled = speech_diarization_settled(diarization_.get());
    for (int s = 0; s < speakers_; s++) {
        double from = 0;
        if (speech_diarization_speaking(diarization_.get(), s, &from)) segments.push_back({from, settled, s});
    }
    return segments;
}

std::vector<SpeakerSegment> Diarization::within(double start, double end) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return speakers_within(segments(), start, end);
}

int Diarization::numbers_used() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int used = 0;
    for (const SpeakerSegment & s : segments()) used = std::max(used, s.speaker + 1);
    return used;
}
