#include "prompt.h"

#include <algorithm>
#include <cmath>
#include <optional>

#include "error.h"
#include "unicode.h"
#include "utf8.h"

namespace haru {

namespace {

/**
 * `s` without its leading and trailing white space as Python's str.strip() takes it, U+3000 among it. A value that is not
 * UTF-8 throws, naming the field.
 */
std::string strip(const std::string & s, const std::string & field) {
    const std::optional<std::vector<uint32_t>> cps = decode_utf8(s);
    if (!cps) throw Error(Fault::InvalidArgument, "the character card's " + field + " is not UTF-8", field);
    size_t a = 0, b = cps->size();
    while (a < b && python_space((*cps)[a])) a++;
    while (b > a && python_space((*cps)[b - 1])) b--;
    return encode_utf8(std::vector<uint32_t>(cps->begin() + (std::ptrdiff_t) a, cps->begin() + (std::ptrdiff_t) b));
}

}  // namespace

std::string render_card(const ModelFile & m, const std::string & language, const std::map<std::string, std::string> & fields) {
    const std::string k = "haru.card.";
    const std::vector<std::string> languages = m.str_array(k + "languages");
    if (std::find(languages.begin(), languages.end(), language) == languages.end()) {
        throw Error(Fault::OutOfRange, "the character card has no wording in " + language + "; give a language the model speaks", "language");
    }
    const std::vector<std::string> names = m.str_array(k + "fields");
    for (const auto & [name, value] : fields) {
        if (std::find(names.begin(), names.end(), name) == names.end()) {
            throw Error(Fault::InvalidArgument, "the character card has no field " + name, name);
        }
    }
    const std::vector<std::string> headers = m.str_array(k + language + ".headers");
    std::string text = m.str(k + language + ".lead");
    for (size_t i = 0; i < names.size(); i++) {
        const auto it = fields.find(names[i]);
        std::string value = it == fields.end() ? "" : strip(it->second, names[i]);
        if (!value.empty() && m.has(k + names[i] + "_values")) {
            const std::vector<std::string> values = m.str_array(k + names[i] + "_values");
            const auto at = std::find(values.begin(), values.end(), value);
            if (at == values.end()) throw Error(Fault::OutOfRange, "the character card's " + names[i] + " takes none of " + value, names[i]);
            value = m.str_array(k + language + "." + names[i])[(size_t) (at - values.begin())];
        }
        if (!value.empty()) text += "\n" + headers[i] + "\n" + value;
    }
    // The official server refuses a card of more characters than this, which the model file holds.
    const size_t most = (size_t) m.size("haru.dialogue.card_max_chars"), chars = decode_utf8(text)->size();
    if (chars > most) {
        const auto longest = std::max_element(fields.begin(), fields.end(),
                                              [](const auto & a, const auto & b) { return a.second.size() < b.second.size(); });
        throw Error(Fault::OutOfRange,
                    "the character card is " + std::to_string(chars) + " characters, and " + m.str("general.name") + " takes " + std::to_string(most) +
                        " at most; shorten its fields",
                    longest->first);
    }
    return text;
}

std::vector<int32_t> card_ids(const ModelFile & m, const Qwen2Tokenizer & bpe, const AddedVocabulary & added, const std::string & card) {
    if (card.empty()) return {};
    return added.encode(m.str("haru.dialogue.system_open") + card + m.str("haru.dialogue.system_close"), bpe);
}

std::vector<float> prepare_reference(const ModelFile & m, const std::vector<float> & audio) {
    const int64_t rate = m.size("speech.sample_rate");
    const int64_t window = rate / m.size("haru.dialogue.silence_window_divisor");
    const int64_t count = (int64_t) audio.size() / window;
    if (count == 0) return audio;
    const double threshold = m.f32("haru.dialogue.silence_dbfs");
    int64_t first = -1, last = -1;
    for (int64_t w = 0; w < count; w++) {
        double sum = 0;
        for (int64_t i = 0; i < window; i++) sum += (double) audio[(size_t) (w * window + i)] * audio[(size_t) (w * window + i)];
        if (20.0 * std::log10(std::sqrt(sum / (double) window + 1e-12)) > threshold) {
            if (first < 0) first = w;
            last = w;
        }
    }
    if (first < 0) return {};
    const int64_t margin = (int64_t) (m.f32("haru.dialogue.silence_margin_seconds") * (float) rate);
    const int64_t start = std::max<int64_t>(0, first * window - margin);
    const int64_t end = std::min<int64_t>((int64_t) audio.size(), (last + 1) * window + margin);
    const int64_t keep = (int64_t) (m.f32("haru.dialogue.reference_max_seconds") * (float) rate);
    return std::vector<float>(audio.begin() + std::max(start, end - keep), audio.begin() + end);
}

int64_t quietest_frame(const std::vector<float> & reference, int64_t frame_samples) {
    const int64_t whole = (int64_t) reference.size() / frame_samples;
    int64_t best = -1;
    double lowest = 0.0;
    for (int64_t f = 0; f < whole; f++) {
        // The mean of the squares, which the official code takes in float32 with torch's sum; double rounds a frame's
        // energy nearer the exact value than either.
        double sum = 0.0;
        for (int64_t i = 0; i < frame_samples; i++) sum += (double) reference[(size_t) (f * frame_samples + i)] * reference[(size_t) (f * frame_samples + i)];
        const double energy = sum / (double) frame_samples;
        if (best < 0 || energy < lowest) {
            best = f;
            lowest = energy;
        }
    }
    return best;
}

Prompt build_prompt(const std::vector<int32_t> & card, const std::vector<float> & reference_latents, bool speak_first,
                    const std::vector<float> & room_tone, const std::vector<float> & silence, const TextIds & ids, int64_t latent_dim) {
    const int64_t n_ref = (int64_t) reference_latents.size() / latent_dim;
    Prompt p;
    p.tokens = card;
    for (int64_t f = 0; f < n_ref; f++) p.tokens.push_back(f + 1 < n_ref ? ids.pad : ids.eop);
    // A reference already ends in EOP, so an opening turn starts with PAD, EOP.
    std::vector<int32_t> grid;
    if (speak_first) grid = {ids.pad, ids.eop};
    if (card.empty() && n_ref == 0 && grid.empty()) grid = {ids.pad};
    p.tokens.insert(p.tokens.end(), grid.begin(), grid.end());
    const int64_t n = p.rows(), n_card = (int64_t) card.size(), n_grid = (int64_t) grid.size();
    p.model_latents.assign((size_t) (n * latent_dim), 0.0f);
    p.user_latents.assign((size_t) (n * latent_dim), 0.0f);
    std::copy(reference_latents.begin(), reference_latents.end(), p.model_latents.begin() + n_card * latent_dim);
    for (int64_t r = n - n_grid; r < n; r++) {
        std::copy(room_tone.begin(), room_tone.end(), p.model_latents.begin() + r * latent_dim);
        std::copy(silence.begin(), silence.end(), p.user_latents.begin() + r * latent_dim);
    }
    p.model_mask.assign((size_t) n, 0.0f);
    p.user_mask.assign((size_t) n, 0.0f);
    for (int64_t r = 0; r < n; r++) {
        p.model_mask[(size_t) r] = r >= n_card ? 1.0f : 0.0f;
        p.user_mask[(size_t) r] = r >= n - n_grid ? 1.0f : 0.0f;
    }
    return p;
}

}  // namespace haru
