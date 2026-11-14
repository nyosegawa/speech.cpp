#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "added-vocabulary.h"
#include "model-file.h"
#include "qwen2-tokenizer.h"
#include "sampler.h"

namespace haru {

/**
 * The character card's text, as haru.prompt's render_card() writes it with the wording of `language`, one of the file's
 * haru.card.languages: the lead sentence, then each field of haru.card.fields that has a value, its header and its value,
 * each on a line of its own; the gender and the backchannels are labels chosen by haru.card.gender_values and
 * backchannel_values. Each value is taken without its leading and trailing white space, and an empty one is left out. A
 * language the file has no wording for, a field it does not know, a choice it does not offer and a card of more
 * characters than haru.dialogue.card_max_chars throw an Error naming the input, the longest field for the last, where
 * the official code leaves an unknown choice out.
 */
std::string render_card(const ModelFile & m, const std::string & language, const std::map<std::string, std::string> & fields);

/**
 * The ids of `card` as a ChatML system turn, as system_prompt_ids() gives them: the added tokens of the turn's markers,
 * and any written in the card, found first, the rest by the BPE; none for an empty card.
 */
std::vector<int32_t> card_ids(const ModelFile & m, const Qwen2Tokenizer & bpe, const AddedVocabulary & added, const std::string & card);

/**
 * A voice reference as prepare_reference() prepares it, mono at 24 kHz: its leading and trailing windows of 1/50 s
 * quieter than haru.dialogue.silence_dbfs cut but for silence_margin_seconds, and of the rest the last
 * reference_max_seconds; nothing where every window is that quiet.
 */
std::vector<float> prepare_reference(const ModelFile & m, const std::vector<float> & audio);

/** The index of the quietest whole frame of a reference, the speaker's noise floor, or -1 where it has none. */
int64_t quietest_frame(const std::vector<float> & reference, int64_t frame_samples);

/**
 * The rows of a conversation's prompt, as build_prompt() lays them out: the card's tokens, which carry no audio, then
 * the reference's frames, PAD on each but the last, EOP on it, with the reference's latents on the model's side; then,
 * where the model speaks first, PAD and EOP with `room_tone` on the model's side and `silence` on the user's; a prompt
 * of neither card, reference nor opening is one PAD row of that kind.
 */
struct Prompt {
    std::vector<int32_t> tokens;
    /** The model's and the user's latents, [latent, rows], as the codec gives them, 0 on a row without them. */
    std::vector<float> model_latents, user_latents;
    /** 1 on the rows that carry the model's or the user's audio, 0 on the rest. */
    std::vector<float> model_mask, user_mask;

    int64_t rows() const { return (int64_t) tokens.size(); }
};

Prompt build_prompt(const std::vector<int32_t> & card, const std::vector<float> & reference_latents, bool speak_first,
                    const std::vector<float> & room_tone, const std::vector<float> & silence, const TextIds & ids, int64_t latent_dim);

}  // namespace haru
