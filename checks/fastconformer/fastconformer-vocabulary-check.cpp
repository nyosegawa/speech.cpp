// Checks the port of Haas-ASR's custom vocabulary against Haas-ASR itself on the cases of
// reference/fastconformer/vocabulary_cases.py: the words, readings and spellings each vocabulary's entries make and the
// entries refused, spellings() of a word, the words find() finds in each transcript and the text format() writes,
// difflib.SequenceMatcher's opcodes and guarded() of texts decoded without and with a vocabulary, and has_speech() of
// signals at and just under the loudness they are given. It prints how many cases of each kind give what Haas-ASR
// gives, and the first that do not.
//
// usage: fastconformer-vocabulary-check <vocabulary-cases.json>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "error.h"
#include "fastconformer/guard.h"
#include "fastconformer/vocabulary.h"
#include "json-reader.h"
#include "json.h"
#include "utf8.h"

using namespace fastconformer;

namespace {

/** The cases of one kind, and how many of them give what Haas-ASR gives. */
struct Tally {
    const char * kind;
    size_t agree = 0, total = 0;
};

constexpr int kShown = 10;
int shown = 0;

/** Counts a case, and prints it where it differs, the first kShown of all. */
void count(Tally & t, bool agree, const std::string & what) {
    t.total++;
    if (agree) {
        t.agree++;
        return;
    }
    if (shown++ < kShown) std::printf("DIFFERS in %s: %s\n", t.kind, what.c_str());
}

const JsonValue & member(const JsonValue & v, const char * name) {
    const JsonValue * m = v.member(name);
    if (!m) throw std::runtime_error(std::string("a case has no member ") + name + ": " + json_excerpt(v));
    return *m;
}

size_t integer(const JsonValue & v) { return std::stoull(v.text); }

std::vector<std::string> strings(const JsonValue & v) {
    std::vector<std::string> out;
    for (const JsonValue & item : v.items) out.push_back(item.text);
    return out;
}

std::vector<uint32_t> code_points(const std::string & s) {
    std::optional<std::vector<uint32_t>> cps = decode_utf8(s);
    if (!cps) throw std::runtime_error("a case's text is not UTF-8");
    return *cps;
}

std::string listed(const std::vector<std::string> & items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); i++) out += (i ? ", " : "") + json_string(items[i]);
    return out + "]";
}

std::string listed(const std::vector<VocabularyWord> & words) {
    std::string out = "[";
    for (size_t i = 0; i < words.size(); i++) out += (i ? ", [" : "[") + json_string(words[i].text) + ", " + listed(words[i].readings) + "]";
    return out + "]";
}

std::string listed(const std::vector<FoundWord> & found) {
    std::string out = "[";
    for (size_t i = 0; i < found.size(); i++) {
        out += (i ? ", [" : "[") + std::to_string(found[i].start) + ", " + std::to_string(found[i].end) + ", " + json_string(found[i].text) + "]";
    }
    return out + "]";
}

const char * tag_name(Opcode::Tag tag) {
    switch (tag) {
        case Opcode::Tag::Equal: return "equal";
        case Opcode::Tag::Replace: return "replace";
        case Opcode::Tag::Delete: return "delete";
        case Opcode::Tag::Insert: return "insert";
    }
    return "";
}

std::string listed(const std::vector<Opcode> & ops) {
    std::string out = "[";
    for (size_t i = 0; i < ops.size(); i++) {
        const Opcode & o = ops[i];
        out += std::string(i ? ", [" : "[") + "\"" + tag_name(o.tag) + "\", " + std::to_string(o.i1) + ", " + std::to_string(o.i2) + ", " +
               std::to_string(o.j1) + ", " + std::to_string(o.j2) + "]";
    }
    return out + "]";
}

std::vector<VocabularyWord> words_of(const JsonValue & v) {
    std::vector<VocabularyWord> out;
    for (const JsonValue & w : v.items) out.push_back({w.items.at(0).text, strings(w.items.at(1))});
    return out;
}

std::vector<FoundWord> found_of(const JsonValue & v) {
    std::vector<FoundWord> out;
    for (const JsonValue & f : v.items) out.push_back({integer(f.items.at(0)), integer(f.items.at(1)), f.items.at(2).text});
    return out;
}

std::vector<Opcode> opcodes_of(const JsonValue & v) {
    std::vector<Opcode> out;
    for (const JsonValue & o : v.items) {
        const std::string & tag = o.items.at(0).text;
        const Opcode::Tag t = tag == "equal"     ? Opcode::Tag::Equal
                              : tag == "replace" ? Opcode::Tag::Replace
                              : tag == "delete"  ? Opcode::Tag::Delete
                                                 : Opcode::Tag::Insert;
        out.push_back({t, integer(o.items.at(1)), integer(o.items.at(2)), integer(o.items.at(3)), integer(o.items.at(4))});
    }
    return out;
}

bool same(const std::vector<VocabularyWord> & a, const std::vector<VocabularyWord> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].text != b[i].text || a[i].readings != b[i].readings) return false;
    }
    return true;
}

bool same(const std::vector<FoundWord> & a, const std::vector<FoundWord> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].start != b[i].start || a[i].end != b[i].end || a[i].text != b[i].text) return false;
    }
    return true;
}

bool same(const std::vector<Opcode> & a, const std::vector<Opcode> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].tag != b[i].tag || a[i].i1 != b[i].i1 || a[i].i2 != b[i].i2 || a[i].j1 != b[i].j1 || a[i].j2 != b[i].j2) return false;
    }
    return true;
}

/** Samples written as the hex of their float32 bytes, little-endian. */
std::vector<float> samples_of(const std::string & hex) {
    std::vector<float> out(hex.size() / 8);
    for (size_t i = 0; i < out.size(); i++) {
        uint32_t bits = 0;
        for (size_t b = 0; b < 4; b++) bits |= (uint32_t) std::stoul(hex.substr(i * 8 + b * 2, 2), nullptr, 16) << (8 * b);
        std::memcpy(&out[i], &bits, sizeof bits);
    }
    return out;
}

std::string read_file(const std::string & path) {
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 2) {
        std::fprintf(stderr, "usage: %s <vocabulary-cases.json>\n", args[0].c_str());
        return 2;
    }
    try {
        const JsonValue cases = parse_json(read_file(args[1]));
        Tally words{"words"}, refused{"refused"}, spelled{"spellings"}, finds{"find"}, formats{"format"}, ops{"opcodes"},
            guards{"guarded"}, speech{"has_speech"};

        std::vector<std::unique_ptr<Vocabulary>> vocabularies;
        for (const JsonValue & c : member(cases, "vocabularies").items) {
            const std::vector<std::string> entries = strings(member(c, "entries"));
            const float alpha = std::strtof(member(c, "alpha").text.c_str(), nullptr);
            try {
                vocabularies.push_back(std::make_unique<Vocabulary>(entries, alpha));
            } catch (const Error & e) {
                vocabularies.push_back(nullptr);
                count(words, false, listed(entries) + ": refused, " + e.what());
                continue;
            }
            const Vocabulary & v = *vocabularies.back();
            const std::vector<VocabularyWord> want_words = words_of(member(c, "words"));
            const std::vector<std::string> want_spellings = strings(member(c, "spellings"));
            count(words, same(v.words(), want_words) && v.spellings() == want_spellings && v.alpha() == alpha,
                  listed(entries) + ": want words " + listed(want_words) + " spelled " + listed(want_spellings) + ", got " + listed(v.words()) +
                      " spelled " + listed(v.spellings()));
            for (const JsonValue & t : member(c, "transcripts").items) {
                const std::string & text = member(t, "text").text;
                const std::vector<FoundWord> want = found_of(member(t, "find"));
                const std::vector<FoundWord> got = v.find(code_points(text));
                count(finds, same(got, want), listed(entries) + " in " + json_string(text) + ": want " + listed(want) + ", got " + listed(got));
                const std::string & want_text = member(t, "format").text;
                const std::string got_text = v.format(text);
                count(formats, got_text == want_text,
                      listed(entries) + " in " + json_string(text) + ": want " + json_string(want_text) + ", got " + json_string(got_text));
            }
        }

        for (const JsonValue & c : member(cases, "refused").items) {
            const std::vector<std::string> entries = strings(c);
            bool agree = false;
            try {
                Vocabulary v(entries, 2.0f);
            } catch (const Error & e) {
                agree = e.fault() == Fault::InvalidArgument;
            }
            count(refused, agree, listed(entries) + ": not refused as an invalid argument");
        }

        for (const JsonValue & c : member(cases, "spellings").items) {
            const std::string & text = member(c, "text").text;
            const std::vector<std::string> readings = strings(member(c, "readings"));
            const std::vector<std::string> want = strings(member(c, "spellings"));
            const std::vector<std::string> got = spellings(text, readings);
            count(spelled, got == want, json_string(text) + " " + listed(readings) + ": want " + listed(want) + ", got " + listed(got));
        }

        for (const JsonValue & c : member(cases, "guarded").items) {
            const std::unique_ptr<Vocabulary> & v = vocabularies.at(integer(member(c, "vocabulary")));
            const std::string & plain = member(c, "plain").text;
            const std::string & boosted = member(c, "boosted").text;
            const std::string what = json_string(plain) + " and " + json_string(boosted);
            const std::vector<Opcode> want_ops = opcodes_of(member(c, "opcodes"));
            const std::vector<Opcode> got_ops = opcodes(code_points(plain), code_points(boosted));
            count(ops, same(got_ops, want_ops), what + ": want " + listed(want_ops) + ", got " + listed(got_ops));
            const std::string & want = member(c, "guarded").text;
            const std::string got = v ? guarded(plain, boosted, *v) : "";
            count(guards, v && got == want, what + ": want " + json_string(want) + ", got " + json_string(got));
        }

        for (const JsonValue & c : member(cases, "has_speech").items) {
            const std::vector<float> samples = samples_of(member(c, "samples").text);
            const float rms = std::strtof(member(c, "rms").text.c_str(), nullptr);
            const size_t frame = integer(member(c, "frame"));
            const bool want = member(c, "speech").boolean;
            const bool got = has_speech(samples, rms, frame);
            count(speech, got == want,
                  std::to_string(samples.size()) + " samples, rms " + member(c, "rms").text + ", frame " + std::to_string(frame) + ": want " +
                      (want ? "speech" : "none") + ", got " + (got ? "speech" : "none"));
        }

        size_t agree = 0, total = 0;
        for (const Tally * t : {&words, &refused, &spelled, &finds, &formats, &ops, &guards, &speech}) {
            std::printf("%-10s %zu of %zu as Haas-ASR gives them\n", t->kind, t->agree, t->total);
            agree += t->agree;
            total += t->total;
        }
        std::printf("%zu of %zu cases as Haas-ASR gives them\n", agree, total);
        return agree == total && total > 0 ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
