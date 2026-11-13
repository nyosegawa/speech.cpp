#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "model-file.h"
#include "sampling.h"

namespace haru {

/** The settings of a draw of a text token, as haru.sampling's sample_token() takes them. */
struct Sampling {
    /** 0 or less draws the most likely token. */
    float temperature = 0.7f;
    /** 0, or the vocabulary or more, draws from every token. */
    int top_k = 25;
    float top_p = 1.0f;
};

/** The settings a request samples with when it sets none, the official defaults, which the file holds. */
Sampling file_sampling(const ModelFile & m);

/**
 * The tokens sample_token() draws from and their weights: the logits divided by the temperature, the top_k largest by
 * falling logit (every token where top_k is 0 or the vocabulary or more), and the softmax of those in float32, of which,
 * where top_p is below 1, each token whose probability accumulated before it reaches top_p weighs 0. The first token
 * always stays. A temperature of 0 or less gives the largest logit alone. A logit of minus infinity, a token the session
 * bans, gets no chance; NaN or plus infinity throws an Error of the device, and a temperature that takes a finite logit
 * beyond a float temperature_overflow().
 */
Candidates candidates(const std::vector<float> & logits, const Sampling & s);

/** The DRY repetition penalty of haru.sampling, whose settings the file holds. */
struct DryConfig {
    double multiplier = 0, base = 0, max_penalty = 0;
    int allowed_length = 0, lookback = 0, max_match_length = 0;
};

DryConfig file_dry(const ModelFile & m);

/** The tokens a frame of the text stream is silent with, PAD and EOP, and the end of text. */
struct TextIds {
    int32_t pad = 0, eop = 0, eos = 0;
};

TextIds file_text_ids(const ModelFile & m);

/**
 * The DRY state of one text stream, as haru.sampling's DryState keeps it: the tokens drawn but the exempt ones, the last
 * lookback + max_match_length + 1 of them, and for each token that would extend a sequence that occurred before, at
 * least allowed_length long within the last lookback tokens, the penalty min(max_penalty, multiplier x
 * base^(length - allowed_length)) of the longest such sequence, in double.
 */
class Dry {
public:
    Dry(const DryConfig & config, std::vector<int32_t> exempt);

    /** Takes a drawn token, which an exempt one leaves the state as it was. */
    void update(int32_t token);

    /** The penalties, each a token and its penalty, by rising token. */
    const std::vector<std::pair<int32_t, double>> & penalties() const { return penalties_; }

    /** Subtracts each penalty, rounded to float, from its token's logit. */
    void apply(std::vector<float> & logits) const;

private:
    bool exempt(int32_t token) const;

    DryConfig config_;
    std::vector<int32_t> exempt_, history_;
    std::vector<std::pair<int32_t, double>> penalties_;
};

/**
 * The logits a frame's text token is drawn from, as the official session forms them from the text logits: DRY's
 * penalties subtracted, the logits of PAD and EOP replaced by `silent`, theirs in float32, `eop_bias` added to EOP's where
 * it is not 0, and, where the model must speak, the logits of PAD, EOP and EOS set to minus infinity.
 */
void shape_logits(std::vector<float> & logits, const Dry & dry, const float silent[2], const TextIds & ids, float eop_bias, bool must_speak);

}  // namespace haru
