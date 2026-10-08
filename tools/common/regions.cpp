#include "regions.h"

#include <algorithm>
#include <cmath>

namespace {

/** The longest region, in seconds, that transcription by regions gives a recognizer. */
constexpr double kLongestRegion = 15;

}  // namespace

std::pair<size_t, size_t> region_samples(const Region & region, int rate, size_t n) {
    const auto at = [&](double seconds) { return std::min(n, (size_t) std::max(0.0, std::round(seconds * rate))); };
    return {at(region.start), at(region.end)};
}

std::vector<RequestOption> region_options(const std::vector<RequestOption> & given) {
    std::vector<RequestOption> out = {
        {SPEECH_OPT_THRESHOLD, 0.5},
        {SPEECH_OPT_SPEECH_PAD_MS, (int64_t) 300},
        {SPEECH_OPT_MIN_SILENCE_DURATION_MS, (int64_t) 500},
        {SPEECH_OPT_MAX_SPEECH_DURATION_S, kLongestRegion},
    };
    for (const RequestOption & o : given) {
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

std::optional<std::vector<Region>> detect_regions(speech_request * detection, Cancellation & cancellation) {
    if (check(cancellation.run(detection, [&] { return speech_detect(detection); })) == SPEECH_CANCELLED) return std::nullopt;
    const speech_result * result = speech_request_result(detection);
    std::vector<Region> regions(speech_result_segment_count(result));
    for (size_t i = 0; i < regions.size(); i++) check(speech_result_segment(result, i, &regions[i].start, &regions[i].end, nullptr));
    return regions;
}

std::optional<Transcript> transcribe_regions(speech_model * recognizer, const std::vector<float> & samples, int rate,
                                             const std::vector<Region> & regions, const std::vector<RequestOption> & options,
                                             bool timestamps, Cancellation & cancellation) {
    apply_options(new_request(recognizer).get(), options);
    Transcript whole;
    for (const Region & region : regions) {
        const auto [first, last] = region_samples(region, rate, samples.size());
        const Request request = new_request(recognizer);
        check(speech_request_set_audio(request.get(), samples.data() + first, last - first, rate));
        apply_options(request.get(), options);
        if (check(cancellation.run(request.get(), [&] { return speech_transcribe(request.get()); })) == SPEECH_CANCELLED) return std::nullopt;
        append_part(whole, transcript_of(speech_request_result(request.get()), timestamps), (double) first / rate);
    }
    return whole;
}
