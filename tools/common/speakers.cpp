#include "speakers.h"

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
