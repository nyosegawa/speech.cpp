#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ctc-tokenizer.h"
#include "model-file.h"

namespace nemotron_transcribe {

/**
 * Why the alignment cannot time a transcript, where NeMo's aligner raises a ValueError or drops a speaker's words: the
 * message says what it could not do.
 */
class AlignmentFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/**
 * What the alignment times: a word of the transcript or, in a transcript of Japanese or Chinese, a letter, mark or
 * number of a word, as NeMo's word records give it, with its CTC token ids.
 */
struct Unit {
    std::string text;
    /** The number of the <spk:N> tag before it, -1 before the first tag, 0 in a transcript without tags. */
    int speaker = -1;
    /** The index of that tag among the transcript's tags, -1 where none comes before. */
    int turn = -1;
    bool character = false;
    /** The index of its word among the transcript's words. */
    size_t word = 0;
    std::vector<int32_t> ids;
    /** The bytes of the transcript that the unit's text stands on. */
    size_t begin = 0, end = 0;
    /**
     * The bytes of the transcript that the unit's token takes: from the end of the token before, or the start of the
     * transcript, to the next unit of its word, or to the end of its word, or the end of the transcript for the last.
     */
    size_t token_begin = 0, token_end = 0;
};

/** One speaker's units and their CTC target: blank, then each token id followed by a blank. */
struct Stream {
    int speaker;
    /** The stream's units, indices into AlignmentText::units in the transcript's order. */
    std::vector<size_t> units;
    /** The target's classes, then, once columns are taken, its columns. */
    std::vector<int32_t> labels;
    /** The stream's unit that each state of the target belongs to, an index into `units`, or -1 for a blank. */
    std::vector<int32_t> state_unit;
};

/** A transcript as the alignment takes it: its units, its speakers' streams and the classes they read. */
struct AlignmentText {
    std::vector<Unit> units;
    /** The streams in the order of their speakers' first units. */
    std::vector<Stream> streams;
    /** The blank and then the classes of the targets, ascending: the columns of the log probabilities the search reads. */
    std::vector<int32_t> columns;
};

/**
 * The text stages of NeMo Speech's MultiSpeakerSOTWordTimestampAligner (ctc_timestamp_utils.py) on a transcript as the
 * model wrote it. clean_sot_transcript() replaces each of the language model tokenizer's special tokens
 * (ctc.removed_tokens) with a space; parse_sot_words() splits the text at each <spk:N> tag, which starts a turn of
 * speaker N, and into words at whitespace and em dashes. _timestamp_units() takes a transcript with a Hangul character
 * for Korean, else one with kana for Japanese, else one with a CJK ideograph for Chinese, else one with a Thai character
 * for Thai, from the characters of its words that are not tags of the form <...>; the words of Japanese and Chinese become
 * the letters, marks and numbers of their NFC, and those of other languages stay whole. Thai's units are pythainlp's
 * words, which this release does not segment, so a Thai transcript cannot be aligned. Each speaker's units are tokenized
 * by the CTC head's SentencePiece model: a character as its own piece, or the pieces of its encoding but the bare "▁",
 * which must spell it and be known; words together and alone, which must agree. Python's re matches \d and IGNORECASE
 * more widely than ASCII; the tags the model writes are ASCII, and only those are taken here.
 */
class CtcText {
public:
    explicit CtcText(const ModelFile & m);

    /** The units and streams of `transcript`; one that cannot be aligned throws AlignmentFailure. */
    AlignmentText prepare(const std::string & transcript) const;

    /**
     * The bytes of each <spk:N> tag of `transcript` that prepare() takes for one, [begin, end); a transcript that is not
     * UTF-8 throws AlignmentFailure.
     */
    std::vector<std::pair<size_t, size_t>> speaker_tags(const std::string & transcript) const;

    /** The classes of the head, the blank last. */
    int32_t blank() const { return blank_; }
    const CtcTokenizer & tokenizer() const { return tokenizer_; }

private:
    struct Parsed;
    Parsed parse(const std::string & transcript) const;
    std::vector<int32_t> character_ids(const std::string & character) const;
    std::vector<std::vector<int32_t>> word_ids(const std::vector<const Unit *> & words, int speaker) const;

    CtcTokenizer tokenizer_;
    std::vector<std::string> removed_;
    int32_t blank_;
};

/** The letters, marks and numbers of the NFC of UTF-8 `text`, each a string, as NeMo's _lexical_characters() gives them. */
std::vector<std::string> lexical_characters(const std::string & text);

}  // namespace nemotron_transcribe
