#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "boosting.h"
#include "model-file.h"
#include "sentencepiece.h"
#include "vocabulary.h"

namespace fastconformer {

/**
 * What a file of fastconformer.vocabulary.kind haas-asr holds for Haas-ASR's custom vocabulary: the tokenizer's
 * encoding, the tree's settings, the default alpha and has_speech()'s threshold and frame.
 */
class WordsModel {
public:
    /** The settings of the file `m`, which offers the vocabulary (offers()). */
    explicit WordsModel(const ModelFile & m);

    /** Whether the file `m` offers Haas-ASR's custom vocabulary. */
    static bool offers(const ModelFile & m);

    /**
     * The token sequences a spelling is boosted as, haas_asr.boosting.word_forms(): SentencePiece's encoding, which
     * begins with the word-start piece a Japanese transcript has only where an utterance begins, and, where that piece
     * is one of its own or merged into the first, the same tokens without it, after the encoding, the merged piece's text
     * as one piece or one a character where the vocabulary has them.
     */
    std::vector<std::vector<int32_t>> word_forms(const std::string & spelling) const;

    /** The tree of every spelling of `vocabulary` in each of its word forms, in order, as boosting._compiled() builds it. */
    BoostingTree tree(const Vocabulary & vocabulary) const;

    float alpha() const { return alpha_; }
    float speech_rms() const { return speech_rms_; }
    int speech_frame() const { return speech_frame_; }

private:
    /** The ids of `pieces` where the vocabulary holds every one, boosting._pieces(); empty with `found` false otherwise. */
    std::vector<int32_t> ids_of(const std::vector<std::string> & pieces, bool & found) const;

    PieceEncoder encoder_;
    std::vector<std::string> pieces_;
    std::map<std::string, int32_t> id_of_;
    double context_score_, depth_scaling_;
    float unk_score_, alpha_, speech_rms_;
    int speech_frame_;
};

}  // namespace fastconformer
