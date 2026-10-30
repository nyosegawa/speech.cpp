// Checks FastConformer's SentencePiece encoder against sentencepiece on the cases reference/fastconformer/
// sentencepiece_cases.py writes, for the model as it is and for each of its variants: what the normalizer makes of each
// text, the ids of what it made, and the ids of the text, which must all be identical. It prints, for each kind of
// text, how many cases agree, and the first cases that differ with the stage that departs.
//
// usage: fastconformer-sentencepiece-check <cases dir>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "fastconformer/sentencepiece.h"
#include "utf8.h"

using namespace fastconformer;

namespace {

constexpr int kShown = 10;

std::string read_file(const std::filesystem::path & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path.u8string());
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

/** The lines of a TSV file, each split at its tabs. */
std::vector<std::vector<std::string>> read_tsv(const std::filesystem::path & path) {
    std::istringstream in(read_file(path));
    std::vector<std::vector<std::string>> rows;
    for (std::string line; std::getline(in, line);) {
        std::vector<std::string> fields;
        size_t start = 0;
        for (size_t tab = line.find('\t'); tab != std::string::npos; start = tab + 1, tab = line.find('\t', start)) {
            fields.push_back(line.substr(start, tab - start));
        }
        fields.push_back(line.substr(start));
        rows.push_back(fields);
    }
    return rows;
}

std::string from_hex(const std::string & hex) {
    if (hex.size() % 2 != 0) throw std::runtime_error("odd hex: " + hex);
    std::string bytes;
    for (size_t i = 0; i < hex.size(); i += 2) bytes += (char) std::stoi(hex.substr(i, 2), nullptr, 16);
    return bytes;
}

std::vector<int32_t> parse_ids(const std::string & s) {
    std::vector<int32_t> ids;
    std::istringstream in(s);
    for (int32_t id; in >> id;) ids.push_back(id);
    return ids;
}

PieceModel read_model(const std::filesystem::path & dir) {
    PieceModel m;
    std::map<std::string, bool> flags;
    for (const auto & row : read_tsv(dir / "normalizer.tsv")) flags[row.at(0)] = row.at(1) == "1";
    for (const char * name : {"add_dummy_prefix", "remove_extra_whitespaces", "escape_whitespaces"}) {
        if (!flags.count(name)) throw std::runtime_error((dir / "normalizer.tsv").u8string() + " has no " + name);
    }
    m.add_dummy_prefix = flags["add_dummy_prefix"];
    m.remove_extra_whitespaces = flags["remove_extra_whitespaces"];
    m.escape_whitespaces = flags["escape_whitespaces"];
    m.charsmap = read_file(dir / "charsmap.bin");
    for (const auto & row : read_tsv(dir / "pieces.tsv")) {
        m.types.push_back((PieceType) std::stoi(row.at(0)));
        // float.hex() writes the score exactly, -0.0 included, and strtof() reads it back so.
        m.scores.push_back(std::strtof(row.at(1).c_str(), nullptr));
        m.pieces.push_back(from_hex(row.at(2)));
    }
    return m;
}

/** The text with its controls escaped and what is not well-formed UTF-8 as U+FFFD, for a line of the output. */
std::string shown(const std::string & s) {
    std::string out;
    for (unsigned char c : s) {
        if (c < 0x20 || c == 0x7F) {
            char e[5];
            std::snprintf(e, sizeof e, "\\x%02X", c);
            out += e;
        } else {
            out += (char) c;
        }
    }
    return replace_invalid_utf8(out);
}

std::string joined(const std::vector<int32_t> & ids) {
    std::string out;
    for (int32_t id : ids) out += (out.empty() ? "" : " ") + std::to_string(id);
    return out;
}

/** Checks the cases of one folder and returns whether every one agrees. */
bool check(const std::filesystem::path & dir) {
    const PieceEncoder encoder(read_model(dir));
    std::map<std::string, std::pair<int, int>> kinds;
    int differ = 0;
    int cases = 0;
    for (const auto & row : read_tsv(dir / "cases.tsv")) {
        const std::string & kind = row.at(0);
        const std::string text = from_hex(row.at(1));
        const std::string normalized = from_hex(row.at(2));
        const std::vector<int32_t> want = parse_ids(row.at(3));
        const std::string got_normalized = encoder.normalize(text);
        const std::vector<int32_t> got_merged = encoder.encode_normalized(normalized);
        const std::vector<int32_t> got = encoder.encode(text);
        const bool agree = got_normalized == normalized && got_merged == want && got == want;
        cases++;
        kinds[kind].second++;
        if (agree) {
            kinds[kind].first++;
            continue;
        }
        if (differ++ >= kShown) continue;
        std::printf("DIFFERS (%s): \"%s\" (%s)\n", kind.c_str(), shown(text).c_str(), row.at(1).c_str());
        if (got_normalized != normalized) {
            std::printf("  normalized: want \"%s\"\n              got  \"%s\"\n", shown(normalized).c_str(), shown(got_normalized).c_str());
        }
        if (got_merged != want) std::printf("  ids of the normalized text: want %s\n                             got  %s\n", joined(want).c_str(), joined(got_merged).c_str());
        if (got != want) std::printf("  ids of the text: want %s\n                   got  %s\n", joined(want).c_str(), joined(got).c_str());
    }
    std::printf("%s:", dir.u8string().c_str());
    for (const auto & [kind, counts] : kinds) std::printf(" %s %d/%d", kind.c_str(), counts.first, counts.second);
    std::printf("\n%d of %d texts encode as sentencepiece encodes them\n", cases - differ, cases);
    return differ == 0 && cases > 0;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 2) {
        std::fprintf(stderr, "usage: %s <cases dir>\n", args[0].c_str());
        return 2;
    }
    try {
        const std::filesystem::path dir = std::filesystem::u8path(args[1]);
        bool ok = check(dir);
        const std::filesystem::path variants = dir / "variants";
        if (std::filesystem::is_directory(variants)) {
            std::vector<std::filesystem::path> folders;
            for (const auto & entry : std::filesystem::directory_iterator(variants)) folders.push_back(entry.path());
            std::sort(folders.begin(), folders.end());
            for (const auto & folder : folders) ok = check(folder) && ok;
        }
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
