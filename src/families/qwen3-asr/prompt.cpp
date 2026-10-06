#include "prompt.h"

#include "error.h"

namespace qwen3_asr {

Prompt::Prompt(const ModelFile & m, const Tokenizer & tokenizer)
    : tokenizer_(tokenizer),
      before_context_(m.str("qwen3-asr.prompt.before_context")),
      before_audio_(m.str("qwen3-asr.prompt.before_audio")),
      after_audio_(m.str("qwen3-asr.prompt.after_audio")),
      language_prefix_(m.str("qwen3-asr.prompt.language_prefix")),
      asr_text_(m.str("qwen3-asr.prompt.asr_text")),
      audio_token_(tokenizer.added_id(m.str("qwen3-asr.prompt.audio_token"))),
      languages_(m.str_array("general.languages")),
      names_(m.str_array("qwen3-asr.language_names")) {}

PromptIds Prompt::ids(const std::string & context, std::optional<size_t> language, int64_t audio_tokens) const {
    PromptIds p;
    p.ids = naming("prompt", [&] { return tokenizer_.encode(before_context_ + context + before_audio_); });
    p.audio_start = (int64_t) p.ids.size();
    p.ids.insert(p.ids.end(), (size_t) audio_tokens, audio_token_);
    const std::vector<int32_t> after =
        tokenizer_.encode(after_audio_ + (language ? language_prefix_ + names_.at(*language) + asr_text_ : std::string()));
    p.ids.insert(p.ids.end(), after.begin(), after.end());
    return p;
}

}  // namespace qwen3_asr
