#include "enhancement.h"

#include <variant>

Request enhancement_request(speech_model * enhancer, const std::vector<float> & samples, int rate, const std::vector<RequestOption> & options) {
    Request request = new_request(enhancer);
    check(speech_request_set_audio(request.get(), samples.data(), samples.size(), rate));
    apply_options(request.get(), options);
    return request;
}

int enhanced_rate(const speech_model_info * info, int rate, const std::vector<RequestOption> & options) {
    if (speech_model_info_sample_rate(info) > 0) return speech_model_info_sample_rate(info);
    // The last value of an option is the one the request keeps, as apply_options() sets them in order.
    for (auto o = options.rbegin(); o != options.rend(); ++o) {
        if (o->option == SPEECH_OPT_BWE) return (int) std::get<int64_t>(o->value);
    }
    return rate;
}

std::optional<uint64_t> enhance_audio(speech_request * enhancement, speech_audio_callback on_audio, void * user_data, Cancellation & cancellation) {
    if (check(cancellation.run(enhancement, [&] { return speech_enhance(enhancement, on_audio, user_data); })) == SPEECH_CANCELLED) return std::nullopt;
    return speech_result_samples(speech_request_result(enhancement));
}

namespace {

int gather(const float * samples, size_t n, void * user_data) {
    std::vector<float> & out = *static_cast<std::vector<float> *>(user_data);
    out.insert(out.end(), samples, samples + n);
    return 0;
}

}  // namespace

std::optional<Enhanced> gather_enhanced(speech_request * enhancement, int rate, Cancellation & cancellation) {
    Enhanced out;
    out.rate = rate;
    if (!enhance_audio(enhancement, gather, &out.samples, cancellation)) return std::nullopt;
    return out;
}

std::optional<Enhanced> enhance_whole(speech_model * enhancer, const std::vector<float> & samples, int rate, const std::vector<RequestOption> & options,
                                      Cancellation & cancellation) {
    const Request request = enhancement_request(enhancer, samples, rate, options);
    return gather_enhanced(request.get(), enhanced_rate(model_info(enhancer).get(), rate, options), cancellation);
}
