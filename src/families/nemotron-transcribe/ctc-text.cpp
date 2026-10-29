#include "ctc-text.h"

#include <algorithm>
#include <climits>
#include <map>

#include "unicode.h"
#include "utf8.h"

namespace nemotron_transcribe {

namespace {

const std::string kKeys = "nemotron-transcribe.";

struct CodepointRange {
    uint32_t first, last;
};

#include "scripts.inc"

template <size_t N>
bool in_ranges(const CodepointRange (&ranges)[N], uint32_t cp) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < ranges[mid].first) hi = mid;
        else if (cp > ranges[mid].last) lo = mid + 1;
        else return true;
    }
    return false;
}

bool lexical(uint32_t cp) { return unicode_letter(cp) || unicode_mark(cp) || unicode_number(cp); }

size_t utf8_length(uint32_t cp) { return cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4; }

/** A word of the transcript: its code points [first, last) and its speaker. */
struct Word {
    size_t first, last;
    int speaker, turn;
};

}  // namespace

std::vector<std::string> lexical_characters(const std::string & text) {
    std::vector<std::string> out;
    const std::optional<std::vector<uint32_t>> cps = decode_utf8(text);
    if (!cps) return out;
    // Python 3.12's NFC, of Unicode 15.0, differs from 13.0's only in the combining classes of the marks that 14.0 and
    // 15.0 added, of scripts none of the model's languages writes.
    for (uint32_t c : nfc(*cps, UnicodeVersion::V13)) {
        if (lexical(c)) out.push_back(encode_utf8({c}));
    }
    return out;
}

CtcText::CtcText(const ModelFile & m)
    : tokenizer_(m), removed_(m.str_array(kKeys + "ctc.removed_tokens")), blank_((int32_t) m.str_array(kKeys + "ctc.vocabulary").size()) {}

std::vector<int32_t> CtcText::character_ids(const std::string & character) const {
    // _character_token_ids(): the character's own piece, else its encoding without the bare "▁", which must be known,
    // below the blank, and spell the character again.
    const int32_t unknown = tokenizer_.unknown();
    std::vector<int32_t> ids;
    const int32_t id = tokenizer_.piece_to_id(character);
    if (id != unknown) {
        ids.push_back(id);
    } else {
        for (int32_t t : tokenizer_.encode(character)) {
            if (tokenizer_.piece(t) != "\xE2\x96\x81") ids.push_back(t);
        }
    }
    bool spelled = !ids.empty();
    for (int32_t t : ids) spelled = spelled && t != unknown && t >= 0 && t < blank_;
    if (spelled) {
        std::string again;
        for (const std::string & c : lexical_characters(tokenizer_.decode(ids))) again += c;
        spelled = again == character;
    }
    if (!spelled) throw AlignmentFailure("the CTC head's vocabulary cannot spell the character " + character + " of the transcript");
    return ids;
}

std::vector<std::vector<int32_t>> CtcText::word_ids(const std::vector<const Unit *> & words, int speaker) const {
    // _tokenize_word_stream(): the words joined by spaces, which must tokenize as each word alone does.
    std::string joined;
    for (size_t i = 0; i < words.size(); i++) joined += (i ? " " : "") + words[i]->text;
    const std::vector<int32_t> ids = tokenizer_.encode(joined);
    const std::string who = speaker < 0 ? "the words before the first speaker tag" : "the words of speaker " + std::to_string(speaker);
    for (int32_t t : ids) {
        if (t < 0 || t >= blank_) throw AlignmentFailure("SentencePiece gives " + who + " ids outside the CTC head's vocabulary");
    }
    std::map<std::string, std::vector<int32_t>> alone;
    std::vector<std::vector<int32_t>> out;
    std::vector<int32_t> together;
    for (const Unit * w : words) {
        auto it = alone.find(w->text);
        if (it == alone.end()) it = alone.emplace(w->text, tokenizer_.encode(w->text)).first;
        if (it->second.empty()) throw AlignmentFailure("SentencePiece gives the word " + w->text + " no tokens");
        out.push_back(it->second);
        together.insert(together.end(), it->second.begin(), it->second.end());
    }
    if (ids.empty() || together != ids) throw AlignmentFailure("SentencePiece tokenizes " + who + " otherwise together than alone");
    return out;
}

AlignmentText CtcText::prepare(const std::string & transcript) const {
    const std::optional<std::vector<uint32_t>> decoded = decode_utf8(transcript);
    if (!decoded) throw AlignmentFailure("the transcript is not UTF-8");
    const std::vector<uint32_t> & cps = *decoded;
    const size_t n = cps.size();
    std::vector<size_t> offset(n + 1, 0);
    for (size_t i = 0; i < n; i++) offset[i + 1] = offset[i] + utf8_length(cps[i]);

    // What each code point is to parse_sot_words() after clean_sot_transcript(): text, a separator (a removed special
    // token, whitespace or an em dash) or part of a <spk:N> tag.
    enum : uint8_t { kText, kSeparator, kTag };
    std::vector<uint8_t> kind(n, kText);
    for (const std::string & token : removed_) {
        const std::optional<std::vector<uint32_t>> t = decode_utf8(token);
        if (!t || t->empty()) continue;
        for (size_t i = 0; i + t->size() <= n;) {
            bool match = true;
            for (size_t j = 0; j < t->size() && match; j++) match = kind[i + j] == kText && cps[i + j] == (*t)[j];
            if (!match) {
                i++;
                continue;
            }
            std::fill(kind.begin() + (std::ptrdiff_t) i, kind.begin() + (std::ptrdiff_t) (i + t->size()), kSeparator);
            i += t->size();
        }
    }
    for (size_t i = 0; i < n; i++) {
        if (kind[i] == kText && (python_space(cps[i]) || cps[i] == 0x2014)) kind[i] = kSeparator;
    }
    struct Tag {
        size_t first;
        int speaker;
    };
    std::vector<Tag> tags;
    auto is = [&](size_t i, char lower) { return i < n && kind[i] == kText && (cps[i] == (uint32_t) lower || cps[i] == (uint32_t) (lower - 32)); };
    for (size_t i = 0; i < n; i++) {
        if (kind[i] != kText || cps[i] != '<' || !is(i + 1, 's') || !is(i + 2, 'p') || !is(i + 3, 'k') || i + 4 >= n || cps[i + 4] != ':' || kind[i + 4] != kText) continue;
        size_t j = i + 5;
        long long number = 0;
        while (j < n && kind[j] == kText && cps[j] >= '0' && cps[j] <= '9') {
            number = std::min<long long>(number * 10 + (cps[j] - '0'), (long long) INT_MAX + 1);
            j++;
        }
        if (j == i + 5 || j >= n || kind[j] != kText || cps[j] != '>') continue;
        if (number > INT_MAX) throw AlignmentFailure("a speaker tag of the transcript has a number past " + std::to_string(INT_MAX));
        std::fill(kind.begin() + (std::ptrdiff_t) i, kind.begin() + (std::ptrdiff_t) (j + 1), kTag);
        tags.push_back({i, (int) number});
        i = j;
    }

    // The words between separators and tags, each with the speaker of the tag before it.
    std::vector<Word> words;
    size_t next_tag = 0;
    int speaker = tags.empty() ? 0 : -1, turn = -1;
    for (size_t i = 0; i < n;) {
        if (kind[i] != kText) {
            if (kind[i] == kTag && next_tag < tags.size() && tags[next_tag].first == i) {
                speaker = tags[next_tag].speaker;
                turn = (int) next_tag++;
            }
            i++;
            continue;
        }
        size_t j = i;
        while (j < n && kind[j] == kText) j++;
        words.push_back({i, j, speaker, turn});
        i = j;
    }

    // _timestamp_units(): the script, from the characters of the words that are not tags of the form <...>.
    auto tag_like = [&](const Word & w) {
        if (w.last - w.first < 3 || cps[w.first] != '<' || cps[w.last - 1] != '>') return false;
        for (size_t i = w.first + 1; i + 1 < w.last; i++) {
            if (cps[i] == '<' || cps[i] == '>') return false;
        }
        return true;
    };
    bool korean = false, japanese = false, chinese = false, thai = false;
    for (const Word & w : words) {
        if (tag_like(w)) continue;
        for (size_t i = w.first; i < w.last; i++) {
            korean = korean || in_ranges(kKorean, cps[i]);
            japanese = japanese || in_ranges(kJapanese, cps[i]);
            chinese = chinese || in_ranges(kChinese, cps[i]);
            thai = thai || in_ranges(kThai, cps[i]);
        }
    }
    if (!korean && !japanese && !chinese && thai) {
        throw AlignmentFailure("the transcript is Thai, whose words NeMo's alignment segments with pythainlp, which this release does not run");
    }
    const bool characters = !korean && (japanese || chinese);

    AlignmentText out;
    std::vector<size_t> word_of;
    auto text_of = [&](size_t first, size_t last) { return transcript.substr(offset[first], offset[last] - offset[first]); };
    for (size_t k = 0; k < words.size(); k++) {
        const Word & w = words[k];
        if (!characters || tag_like(w)) {
            out.units.push_back({text_of(w.first, w.last), w.speaker, w.turn, false, {}, offset[w.first], offset[w.last]});
            word_of.push_back(k);
            continue;
        }
        // The word's units are the letters, marks and numbers of its NFC. Each stands on the cluster it comes from, a
        // character and the marks after it, which NFC composes within; where the clusters give other units than the word,
        // as Hangul's jamo would, the word's first unit stands on all of it.
        const std::vector<std::string> units = lexical_characters(text_of(w.first, w.last));
        std::vector<std::pair<size_t, size_t>> spans;
        for (size_t i = w.first; i < w.last;) {
            size_t j = i + 1;
            while (j < w.last && unicode_mark(cps[j])) j++;
            const size_t count = lexical_characters(text_of(i, j)).size();
            for (size_t c = 0; c < count; c++) spans.push_back(c == 0 ? std::make_pair(offset[i], offset[j]) : std::make_pair(offset[j], offset[j]));
            i = j;
        }
        if (spans.size() != units.size()) {
            spans.assign(units.size(), {offset[w.last], offset[w.last]});
            if (!spans.empty()) spans[0] = {offset[w.first], offset[w.last]};
        }
        for (size_t c = 0; c < units.size(); c++) {
            out.units.push_back({units[c], w.speaker, w.turn, true, {}, spans[c].first, spans[c].second});
            word_of.push_back(k);
        }
    }
    // Each token takes the bytes from the token before it to the next unit of its word or the end of its word; the
    // first from the start of the transcript and the last to its end.
    for (size_t i = 0; i < out.units.size(); i++) {
        Unit & u = out.units[i];
        u.token_begin = i == 0 ? 0 : out.units[i - 1].token_end;
        const bool same_word = i + 1 < out.units.size() && word_of[i + 1] == word_of[i];
        u.token_end = i + 1 == out.units.size() ? transcript.size() : same_word ? out.units[i + 1].begin : offset[words[word_of[i]].last];
        u.token_end = std::max(u.token_end, u.token_begin);
    }

    // _tokenize_words(): each speaker's stream on its own; characters alone, the other units of a stream with
    // characters each alone as a stream of one, and a stream without characters whole.
    std::map<int, size_t> stream_of;
    for (size_t i = 0; i < out.units.size(); i++) {
        const auto it = stream_of.emplace(out.units[i].speaker, out.streams.size());
        if (it.second) out.streams.push_back({out.units[i].speaker, {}, {}, {}});
        out.streams[it.first->second].units.push_back(i);
    }
    for (Stream & s : out.streams) {
        const bool any_character = std::any_of(s.units.begin(), s.units.end(), [&](size_t u) { return out.units[u].character; });
        if (any_character) {
            for (size_t u : s.units) {
                Unit & unit = out.units[u];
                unit.ids = unit.character ? character_ids(unit.text) : word_ids({&unit}, s.speaker)[0];
            }
        } else {
            std::vector<const Unit *> stream_words;
            for (size_t u : s.units) stream_words.push_back(&out.units[u]);
            const std::vector<std::vector<int32_t>> ids = word_ids(stream_words, s.speaker);
            for (size_t k = 0; k < s.units.size(); k++) out.units[s.units[k]].ids = ids[k];
        }
        // _build_ctc_target(): a blank, then each token and a blank.
        s.labels = {blank_};
        s.state_unit = {-1};
        for (size_t k = 0; k < s.units.size(); k++) {
            for (int32_t id : out.units[s.units[k]].ids) {
                s.labels.insert(s.labels.end(), {id, blank_});
                s.state_unit.insert(s.state_unit.end(), {(int32_t) k, -1});
            }
        }
    }
    std::vector<int32_t> classes;
    for (const Stream & s : out.streams) {
        for (int32_t l : s.labels) {
            if (l != blank_) classes.push_back(l);
        }
    }
    std::sort(classes.begin(), classes.end());
    classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
    out.columns = {blank_};
    out.columns.insert(out.columns.end(), classes.begin(), classes.end());
    return out;
}

}  // namespace nemotron_transcribe
