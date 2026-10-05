#include "prompt.h"

#include <algorithm>
#include <stdexcept>

#include "language.h"

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

}  // namespace

PromptIds::PromptIds(const ModelFile & m) {
    const std::string p = "qwen3-tts.";
    tts_bos = (int32_t) m.u32(p + "text.tts_bos_token_id");
    tts_eos = (int32_t) m.u32(p + "text.tts_eos_token_id");
    tts_pad = (int32_t) m.u32(p + "text.tts_pad_token_id");
    im_start = (int32_t) m.u32(p + "text.im_start_token_id");
    im_end = (int32_t) m.u32(p + "text.im_end_token_id");
    assistant = (int32_t) m.u32(p + "text.assistant_token_id");
    newline = (int32_t) m.u32(p + "text.newline_token_id");
    codec_bos = (int32_t) m.u32(p + "talker.codec_bos_id");
    codec_eos = (int32_t) m.u32(p + "talker.codec_eos_token_id");
    codec_pad = (int32_t) m.u32(p + "talker.codec_pad_id");
    think = (int32_t) m.u32(p + "talker.codec_think_id");
    nothink = (int32_t) m.u32(p + "talker.codec_nothink_id");
    think_bos = (int32_t) m.u32(p + "talker.codec_think_bos_id");
    think_eos = (int32_t) m.u32(p + "talker.codec_think_eos_id");
    voices = m.str_array("speech.voices");
    speaker_ids = m.i32_array(p + "speaker_ids");
    dialect_ids = m.i32_array(p + "dialect_ids");
    languages = m.str_array("speech.languages");
    language_ids = m.i32_array(p + "language_ids");
    dialect_language = m.str(p + "dialect_language");
}

size_t PromptIds::voice(const std::string & name) const {
    const auto it = std::find(voices.begin(), voices.end(), lower(name));
    if (it == voices.end()) throw std::runtime_error("unknown speaker: " + name);
    return (size_t) (it - voices.begin());
}

int32_t PromptIds::language_id(const std::string & tag, size_t voice) const {
    int language = -1;
    if (!tag.empty() && lower(tag) != "auto") {
        for (size_t i = 0; i < languages.size() && language < 0; i++) {
            if (bcp47_matches(tag, languages[i])) language = (int) i;
        }
        if (language < 0) {
            std::string list;
            for (const std::string & l : languages) list += (list.empty() ? "" : ", ") + l;
            throw std::runtime_error("Qwen3-TTS speaks " + list + ", not " + tag);
        }
    }
    // The official prompt gives a voice with a dialect its dialect when the language is Chinese or auto.
    if (dialect_ids[voice] >= 0 && (language < 0 || languages[language] == dialect_language)) return dialect_ids[voice];
    return language < 0 ? -1 : language_ids[language];
}

Prompt build_prompt(Talker & talker, const PromptIds & ids, const std::vector<int32_t> & text_ids,
                    const std::string & voice, const std::string & language) {
    const size_t n_text_ids = text_ids.size();
    // The template's 3 leading tokens are the role, and its 5 trailing ones close it and open the answer.
    if (n_text_ids < 9) throw std::runtime_error("the text is empty");
    const int h = talker.hidden();

    const size_t v = ids.voice(voice);
    const int32_t language_id = ids.language_id(language, v);
    std::vector<int32_t> codec_ids;
    if (language_id < 0) codec_ids = {ids.nothink, ids.think_bos, ids.think_eos};
    else codec_ids = {ids.think, ids.think_bos, language_id, ids.think_eos};
    codec_ids.push_back(ids.speaker_ids[v]);
    codec_ids.push_back(ids.codec_pad);
    codec_ids.push_back(ids.codec_bos);
    const size_t n_codec = codec_ids.size();

    // Text rows: role (3), tts_pad, tts_bos, the text tokens, tts_eos.
    std::vector<int32_t> text_rows(text_ids.begin(), text_ids.begin() + 3);
    text_rows.push_back(ids.tts_pad);
    text_rows.push_back(ids.tts_bos);
    text_rows.insert(text_rows.end(), text_ids.begin() + 3, text_ids.end() - 5);
    text_rows.push_back(ids.tts_eos);
    const std::vector<float> te = talker.text_embeddings(text_rows);
    const std::vector<float> ce = talker.codec_embeddings(codec_ids);
    const float * pad = &te[3 * h];
    const float * bos = &te[4 * h];
    const float * body = &te[5 * h];
    const size_t n_body = n_text_ids - 8;
    const float * eos = &te[(5 + n_body) * h];
    const float * codec_pad = &ce[(n_codec - 2) * h];
    const float * codec_bos = &ce[(n_codec - 1) * h];

    Prompt p;
    auto row = [&](const float * a, const float * b) {
        for (int i = 0; i < h; i++) p.embeds.push_back(a[i] + (b ? b[i] : 0.0f));
        p.n++;
    };
    for (int r = 0; r < 3; r++) row(&te[r * h], nullptr);
    // tts_pad over the think tags, language and speaker, then tts_bos over codec_pad.
    for (size_t c = 0; c + 1 < n_codec; c++) row(c + 2 < n_codec ? pad : bos, &ce[c * h]);
    for (size_t t = 0; t < n_body; t++) row(body + t * h, codec_pad);
    row(eos, codec_pad);
    row(pad, codec_bos);
    p.frame_extra.assign(pad, pad + h);
    return p;
}
