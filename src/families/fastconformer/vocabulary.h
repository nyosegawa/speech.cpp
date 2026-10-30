#pragma once

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace fastconformer {

/** A word of a custom vocabulary: its written form and the readings kept for it, both cleaned. */
struct VocabularyWord {
    std::string text;
    std::vector<std::string> readings;
};

/** A word a text holds: from the code point `start` to `end` of the text, and the word's written form. */
struct FoundWord {
    size_t start, end;
    std::string text;
};

/**
 * The custom vocabulary of Haas-ASR (haas_asr/vocabulary.py): words to boost while decoding, and how to write them where
 * a transcript holds them, as its Vocabulary does with the Unicode tables of Python 3.10, on texts indexed by code point
 * as Python's strings are.
 *
 * A word has a written form and readings, the spellings the model may write for it. It is boosted in each reading, in
 * katakana too for a reading of kana alone, and in its written form, all in lower case. A transcript holds it wherever
 * one of these is found, matched without regard to hiragana or katakana, full-width ASCII, case, and spaces between
 * words or where the script changes; the longest match wins, and one that starts or ends in a Latin letter is not
 * matched inside a longer Latin word.
 */
class Vocabulary {
public:
    /**
     * The words of `entries`, each "written form=reading,reading" as Vocabulary.parse() reads them, and the boosting
     * strength `alpha`. Written forms and readings are stripped of whitespace and put in NFC, and readings that are empty
     * or the written form are dropped. The entries of one written form make one word, and of two written forms that NFC
     * makes one, the later's readings replace the earlier's. An entry without a written form, or not in UTF-8, is an
     * Error.
     */
    Vocabulary(const std::vector<std::string> & entries, float alpha);

    /** The words in the order their entries first give them. */
    const std::vector<VocabularyWord> & words() const { return words_; }
    /** The spellings the words are boosted in, each once, in the order of the words. */
    const std::vector<std::string> & spellings() const { return spellings_; }
    float alpha() const { return alpha_; }
    bool empty() const { return words_.empty(); }

    /** The words `text` holds, from its start, each where it was found. */
    std::vector<FoundWord> find(const std::vector<uint32_t> & text) const;

    /** `text` in UTF-8 with every word it holds in its written form. */
    std::string format(const std::string & text) const;

private:
    /** A spelling as matching compares it: its characters folded and without spaces. */
    struct Form {
        std::vector<uint32_t> chars;
        /** Where a transcript may have whitespace: before the characters at these indices. */
        std::set<size_t> gaps;
        std::string write;
    };

    /** The form of `spelling` for the word written `write` (_form()), whose chars are empty for a spelling of spaces. */
    static Form form(const std::vector<uint32_t> & spelling, const std::string & write);

    /**
     * Where `form` ends if it is read at text[i], whose characters folded are `folded`, or 0 when it is not there
     * (_match()).
     */
    static size_t match(const std::vector<uint32_t> & text, const std::vector<uint32_t> & folded, size_t i, const Form & form);

    std::vector<VocabularyWord> words_;
    std::vector<std::string> spellings_;
    float alpha_;
    /** The forms by their first character, the longest first. */
    std::unordered_map<uint32_t, std::vector<Form>> forms_;
};

/** The spellings the word `text` with the readings `readings` is boosted in (spellings() of vocabulary.py). */
std::vector<std::string> spellings(const std::string & text, const std::vector<std::string> & readings);

}  // namespace fastconformer
