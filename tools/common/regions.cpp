#include "regions.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "failure.h"

namespace {

/**
 * The longest region, in seconds, that transcription by regions gives a recognizer. On 106 minutes of FLEURS and Common
 * Voice ja joined into recordings (measure/region_cap_compare.py, Apple M5, Metal, 2026-10-08), 10 s gave the lowest
 * CER on average over reazonspeech-nemo-v2, parakeet-tdt_ctc-0.6b-ja and Qwen3-ASR 0.6B: longer regions let the
 * FastConformer models drop more sentences, and shorter ones cost Qwen3-ASR the context it reads them with.
 */
constexpr double kLongestRegion = 10;

}  // namespace

std::string server_vad_member(const std::string & option) {
    for (const VadMember & m : kServerVad) {
        if (option == speech_option_name(m.option)) return m.name;
    }
    return "";
}

std::pair<size_t, size_t> region_samples(const Region & region, int rate, size_t n) {
    const auto at = [&](double seconds) { return std::min(n, (size_t) std::max(0.0, std::round(seconds * rate))); };
    return {at(region.start), at(region.end)};
}

std::vector<RequestOption> region_options(const std::vector<RequestOption> & given, const std::vector<RequestOption> & defaults) {
    std::vector<RequestOption> out = {
        {SPEECH_OPT_THRESHOLD, 0.5},
        {SPEECH_OPT_SPEECH_PAD_MS, (int64_t) 300},
        {SPEECH_OPT_MIN_SILENCE_DURATION_MS, (int64_t) 500},
        {SPEECH_OPT_MAX_SPEECH_DURATION_S, kLongestRegion},
    };
    for (const RequestOption & o : with_defaults(defaults, given)) {
        const auto same = std::find_if(out.begin(), out.end(), [&](const RequestOption & d) { return d.option == o.option; });
        if (same != out.end()) *same = o;
        else out.push_back(o);
    }
    return out;
}

Request detection_request(speech_model * detector, const std::vector<float> & samples, int rate, const std::vector<RequestOption> & options) {
    Request request = new_request(detector);
    check(speech_request_set_audio(request.get(), samples.data(), samples.size(), rate));
    apply_options(request.get(), options);
    return request;
}

RegionCut::RegionCut(speech_model * detector, const std::vector<RequestOption> & options) : detection_(options) {
    apply_options(new_request(detector).get(), options);
    const auto value = [&](speech_option option) -> const OptionValue & {
        const auto last = std::find_if(options.rbegin(), options.rend(), [&](const RequestOption & o) { return o.option == option; });
        if (last == options.rend()) throw std::logic_error(std::string("a region cut without ") + speech_option_name(option));
        return last->value;
    };
    const int64_t pad = std::get<int64_t>(value(SPEECH_OPT_SPEECH_PAD_MS));
    const int64_t silence = std::get<int64_t>(value(SPEECH_OPT_MIN_SILENCE_DURATION_MS));
    const double longest = std::get<double>(value(SPEECH_OPT_MAX_SPEECH_DURATION_S));
    before_ = (double) pad / 1000;
    after_ = (double) std::min(pad, silence) / 1000;
    silence_ = (double) silence / 1000;
    // Silero VAD keeps a region with its padding within max_speech_duration_s, which it shortens by twice its own padding.
    const double speech = longest - before_ - after_;
    if (!(speech > 0)) {
        char message[256];
        std::snprintf(message, sizeof message,
                      "speech_pad_ms %lld keeps %g s around the speech of a region, which leaves it no room within max_speech_duration_s %g; "
                      "give less padding or a longer max_speech_duration_s",
                      (long long) pad, before_ + after_, longest);
        throw Failure(speech_status_name(SPEECH_ERROR_INVALID_ARGUMENT), speech_option_name(SPEECH_OPT_SPEECH_PAD_MS), message);
    }
    for (RequestOption & o : detection_) {
        if (o.option == SPEECH_OPT_SPEECH_PAD_MS) o.value = (int64_t) 0;
        if (o.option == SPEECH_OPT_MAX_SPEECH_DURATION_S) o.value = speech;
    }
}

Region RegionCut::pad(const Region & speech, double from, std::optional<double> next, double length) const {
    double end = speech.end + after_;
    if (next && *next - speech.end < silence_) end = std::min(end, speech.end + (*next - speech.end) / 2);
    return {std::max(speech.start - before_, from), std::min(end, length)};
}

std::vector<Region> RegionCut::regions(const std::vector<Region> & speech, double length) const {
    std::vector<Region> out;
    double from = 0;
    for (size_t i = 0; i < speech.size(); i++) {
        const std::optional<double> next = i + 1 < speech.size() ? std::optional<double>(speech[i + 1].start) : std::nullopt;
        out.push_back(pad(speech[i], from, next, length));
        from = out.back().end;
    }
    return out;
}

std::optional<std::vector<Region>> detect_regions(speech_request * detection, Cancellation & cancellation) {
    if (check(cancellation.run(detection, [&] { return speech_detect(detection); })) == SPEECH_CANCELLED) return std::nullopt;
    const speech_result * result = speech_request_result(detection);
    std::vector<Region> regions(speech_result_segment_count(result));
    for (size_t i = 0; i < regions.size(); i++) check(speech_result_segment(result, i, &regions[i].start, &regions[i].end, nullptr));
    return regions;
}

void check_recognition(speech_model * recognizer, const std::vector<RequestOption> & options) {
    const ModelInfo info = model_info(recognizer);
    const speech_task task = speech_model_info_task(info.get());
    if (task != SPEECH_TASK_RECOGNITION) {
        throw Failure(speech_status_name(SPEECH_ERROR_UNSUPPORTED), "",
                      std::string(speech_model_info_name(info.get())) + " is a model of speech " + task_name(task) +
                          "; transcription by regions recognizes them with a speech recognition model");
    }
    apply_options(new_request(recognizer).get(), options);
}

std::optional<Transcript> transcribe_regions(speech_model * recognizer, const std::vector<float> & samples, int rate,
                                             const std::vector<Region> & regions, const std::vector<RequestOption> & options,
                                             bool timestamps, Cancellation & cancellation, std::vector<Transcript> * parts) {
    check_recognition(recognizer, options);
    Transcript whole;
    for (const Region & region : regions) {
        const auto [first, last] = region_samples(region, rate, samples.size());
        const Request request = new_request(recognizer);
        check(speech_request_set_audio(request.get(), samples.data() + first, last - first, rate));
        apply_options(request.get(), options);
        if (check(cancellation.run(request.get(), [&] { return speech_transcribe(request.get()); })) == SPEECH_CANCELLED) return std::nullopt;
        const Transcript part = transcript_of(speech_request_result(request.get()), timestamps);
        append_part(whole, part, (double) first / rate);
        if (parts) parts->push_back(part);
    }
    return whole;
}
