#include "vocabulary.h"

#include <algorithm>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>

#include "error.h"
#include "unicode.h"
#include "utf8.h"

namespace fastconformer {

namespace {

using Text = std::vector<uint32_t>;

/** The code points of an entry or a word the caller gives. */
Text word_code_points(const std::string & s) {
    std::optional<Text> text = decode_utf8(s);
    if (!text) throw Error(Fault::InvalidArgument, "a vocabulary entry is not valid UTF-8; give its words in UTF-8");
    return *text;
}

/** A transcript's code points, which the decoding writes in UTF-8. */
Text transcript_code_points(const std::string & s) {
    std::optional<Text> text = decode_utf8(s);
    if (!text) throw std::invalid_argument("a transcript is not valid UTF-8");
    return *text;
}

/** `text` without the whitespace str.strip() removes from its ends. */
Text strip(const Text & text) {
    size_t first = 0, last = text.size();
    while (first < last && python_space(text[first])) first++;
    while (last > first && python_space(text[last - 1])) last--;
    return Text(text.begin() + (std::ptrdiff_t) first, text.begin() + (std::ptrdiff_t) last);
}

/** _clean(): stripped, in NFC. */
Text clean(const Text & text) { return nfc(strip(text), UnicodeVersion::V13); }

bool hiragana(uint32_t c) { return c >= 0x3041 && c <= 0x3096; }

/** Hiragana as katakana, other characters unchanged (_katakana()). */
Text katakana(Text text) {
    for (uint32_t & c : text) {
        if (hiragana(c)) c += 0x60;
    }
    return text;
}

/** str.lower() of one character, which a capital sigma alone lowers to the small sigma. */
Text lower(uint32_t c) { return python_lower({c}); }

/**
 * One character as matching compares it (_fold()): hiragana as katakana, full-width ASCII as ASCII, the ideographic
 * space as a space, and in lower case where that is one character, so that positions in folded text are positions in
 * the text.
 */
uint32_t fold(uint32_t c) {
    if (hiragana(c)) return c + 0x60;
    if (c >= 0xFF01 && c <= 0xFF5E) c -= 0xFEE0;
    if (c == 0x3000) return ' ';
    const Text l = lower(c);
    return l.size() == 1 ? l[0] : c;
}

/**
 * Whether "a" <= c.lower() <= "z", as Python compares strings, code point by code point and a prefix first: U+0130,
 * whose lower case is "i" and U+0307, is a Latin letter, though it folds to itself.
 */
bool latin(uint32_t c) {
    const Text l = lower(c);
    const Text a{'a'}, z{'z'};
    return !std::lexicographical_compare(l.begin(), l.end(), a.begin(), a.end()) &&
           !std::lexicographical_compare(z.begin(), z.end(), l.begin(), l.end());
}

constexpr uint32_t kLatin = 0x110000, kDigit = 0x110001, kKana = 0x110002, kKanji = 0x110003;

/**
 * The script of a character as _script() tells it: one number past the code points for each of Latin, the digits, kana
 * and kanji, and for any other character the character itself, which is a script of its own.
 */
uint32_t script(uint32_t c) {
    if (latin(c)) return kLatin;
    if (python_digit(c)) return kDigit;
    if ((c >= 0x30A0 && c <= 0x30FF) || (c >= 0x3040 && c <= 0x309F)) return kKana;
    if ((c >= 0x3400 && c <= 0x9FFF) || c == 0x3005) return kKanji;
    return c;
}

/**
 * The spellings a word is boosted in (spellings()): each reading, and in katakana too if it is all kana, then the word,
 * each in lower case and once.
 */
std::vector<Text> spelled(const Text & text, const std::vector<Text> & readings) {
    std::vector<Text> out;
    const auto add = [&out](const Text & s) {
        Text l = python_lower(s);
        if (std::find(out.begin(), out.end(), l) == out.end()) out.push_back(std::move(l));
    };
    for (const Text & reading : readings) {
        const bool kana = std::all_of(reading.begin(), reading.end(), [](uint32_t c) { return c >= 0x3040 && c <= 0x30FF; });
        add(reading);
        add(kana ? katakana(reading) : reading);
    }
    add(text);
    return out;
}

/** A word's written form and its readings. */
struct Word {
    Text text;
    std::vector<Text> readings;
};

/** The readings of each written form of `entries`, stripped, in the order the forms first come (Vocabulary.parse()). */
std::vector<Word> parsed(const std::vector<std::string> & entries) {
    std::vector<Word> words;
    std::map<Text, size_t> word_at;
    for (const std::string & entry : entries) {
        const Text e = word_code_points(entry);
        const auto equals = std::find(e.begin(), e.end(), (uint32_t) '=');
        const Text written = strip(Text(e.begin(), equals));
        if (written.empty()) {
            throw Error(Fault::InvalidArgument,
                        "the vocabulary entry \"" + entry + "\" has no written form; give the word before its readings, as in Haas=ハース");
        }
        const auto [at, added] = word_at.emplace(written, words.size());
        if (added) words.push_back({written, {}});
        std::vector<Text> & readings = words[at->second].readings;
        // str.split(","): what follows the "=" is one empty reading where nothing does.
        Text reading;
        for (auto c = equals == e.end() ? e.end() : equals + 1; c != e.end(); ++c) {
            if (*c == ',') {
                readings.push_back(reading);
                reading.clear();
            } else {
                reading.push_back(*c);
            }
        }
        readings.push_back(reading);
    }
    return words;
}

/**
 * The words as Vocabulary.__init__() cleans them: readings that are empty, the written form or there before dropped,
 * and a written form that cleans to one already there replacing its readings in its place.
 */
std::vector<Word> cleaned(const std::vector<Word> & words) {
    std::vector<Word> out;
    std::map<Text, size_t> word_at;
    for (const Word & w : words) {
        const Text text = clean(w.text);
        std::vector<Text> kept;
        for (const Text & r : w.readings) {
            const Text c = clean(r);
            if (!c.empty() && c != text && std::find(kept.begin(), kept.end(), c) == kept.end()) kept.push_back(c);
        }
        const auto [at, added] = word_at.emplace(text, out.size());
        if (added) {
            out.push_back({text, std::move(kept)});
        } else {
            out[at->second].readings = std::move(kept);
        }
    }
    return out;
}

}  // namespace

Vocabulary::Form Vocabulary::form(const Text & spelling, const std::string & write) {
    Form f;
    f.write = write;
    for (uint32_t c : spelling) {
        if (python_space(c)) {
            if (!f.chars.empty()) f.gaps.insert(f.chars.size());
            continue;
        }
        const uint32_t folded = fold(c);
        if (!f.chars.empty() && script(f.chars.back()) != script(folded)) f.gaps.insert(f.chars.size());
        f.chars.push_back(folded);
    }
    return f;
}

Vocabulary::Vocabulary(const std::vector<std::string> & entries, float alpha) : alpha_(alpha) {
    const std::vector<Word> words = cleaned(parsed(entries));
    std::vector<Text> spellings;
    for (const Word & word : words) {
        VocabularyWord w{encode_utf8(word.text), {}};
        for (const Text & r : word.readings) w.readings.push_back(encode_utf8(r));
        words_.push_back(std::move(w));
        for (Text & s : spelled(word.text, word.readings)) {
            if (std::find(spellings.begin(), spellings.end(), s) == spellings.end()) spellings.push_back(std::move(s));
        }
    }
    for (const Text & s : spellings) spellings_.push_back(encode_utf8(s));

    // A dict of the forms by their characters and written form, each the latest form of its key in the place of the first.
    std::vector<Form> forms;
    std::map<std::pair<Text, std::string>, size_t> form_at;
    const auto add = [&forms, &form_at](const Text & spelling, const std::string & write) {
        Form f = form(spelling, write);
        if (f.chars.empty()) return;
        const auto [at, added] = form_at.emplace(std::make_pair(f.chars, f.write), forms.size());
        if (added) {
            forms.push_back(std::move(f));
        } else {
            forms[at->second] = std::move(f);
        }
    };
    for (size_t w = 0; w < words.size(); w++) {
        for (const Text & r : words[w].readings) add(r, words_[w].text);
        add(words[w].text, words_[w].text);
    }
    std::stable_sort(forms.begin(), forms.end(), [](const Form & a, const Form & b) { return a.chars.size() > b.chars.size(); });
    for (Form & f : forms) forms_[f.chars.front()].push_back(std::move(f));
}

size_t Vocabulary::match(const Text & text, const Text & folded, size_t i, const Form & form) {
    if (latin(form.chars.front()) && i > 0 && latin(folded[i - 1])) return 0;
    size_t j = i;
    for (size_t k = 0; k < form.chars.size(); k++) {
        if (form.gaps.count(k)) {
            while (j < text.size() && python_space(text[j])) j++;
        }
        if (j >= text.size() || folded[j] != form.chars[k]) return 0;
        j++;
    }
    if (latin(form.chars.back()) && j < text.size() && latin(folded[j])) return 0;
    return j;
}

std::vector<FoundWord> Vocabulary::find(const Text & text) const {
    Text folded(text.size());
    std::transform(text.begin(), text.end(), folded.begin(), fold);
    std::vector<FoundWord> found;
    size_t i = 0;
    while (i < text.size()) {
        size_t end = 0;
        const auto forms = forms_.find(folded[i]);
        if (forms != forms_.end()) {
            for (const Form & f : forms->second) {
                end = match(text, folded, i, f);
                if (end > i) {
                    found.push_back({i, end, f.write});
                    break;
                }
            }
        }
        i = end > i ? end : i + 1;
    }
    return found;
}

std::string Vocabulary::format(const std::string & text) const {
    const Text t = transcript_code_points(text);
    std::string out;
    size_t last = 0;
    for (const FoundWord & w : find(t)) {
        out += encode_utf8(Text(t.begin() + (std::ptrdiff_t) last, t.begin() + (std::ptrdiff_t) w.start));
        out += w.text;
        last = w.end;
    }
    return out + encode_utf8(Text(t.begin() + (std::ptrdiff_t) last, t.end()));
}

std::vector<std::string> spellings(const std::string & text, const std::vector<std::string> & readings) {
    std::vector<Text> r;
    for (const std::string & s : readings) r.push_back(word_code_points(s));
    std::vector<std::string> out;
    for (const Text & s : spelled(word_code_points(text), r)) out.push_back(encode_utf8(s));
    return out;
}

}  // namespace fastconformer
