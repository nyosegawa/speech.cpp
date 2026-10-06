// Compares the tokenizer with the model's own tokenizer.json (reference/qwen3-tts/tokenizer_cases.py): the ids it
// encodes each text to, and the text it decodes each sequence of ids to.
//
// usage: tokenizer-check <model.gguf> <cases.tsv> <decode-cases.tsv>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "args.h"
#include "qwen2-tokenizer.h"
#include "qwen3-tts/layout.h"
#include "qwen3-tts/synthesizer.h"

namespace {

std::string from_hex(const std::string & hex) {
    std::string bytes;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) bytes += (char) std::stoi(hex.substr(i, 2), nullptr, 16);
    return bytes;
}

std::vector<int32_t> parse_ids(const std::string & s) {
    std::vector<int32_t> ids;
    std::istringstream in(s);
    for (int32_t id; in >> id;) ids.push_back(id);
    return ids;
}

void print_ids(const char * what, const std::vector<int32_t> & ids) {
    std::printf("  %s", what);
    for (int32_t id : ids) std::printf(" %d", id);
    std::printf("\n");
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <cases.tsv> <decode-cases.tsv>\n", args[0].c_str());
        return 2;
    }
    const ModelFile model(args[1], qwen3_tts_layout);
    const Qwen2Tokenizer tokenizer(model, kTextTokenizer);

    std::ifstream encodings(std::filesystem::u8path(args[2]));
    std::string line;
    int texts = 0, texts_failed = 0;
    while (std::getline(encodings, line)) {
        const size_t tab = line.find('\t');
        const std::string text = from_hex(line.substr(0, tab));
        const std::vector<int32_t> want = parse_ids(line.substr(tab + 1));
        const std::vector<int32_t> got = tokenizer.encode(text);
        texts++;
        if (got != want) {
            texts_failed++;
            std::printf("MISMATCH: %s\n", text.c_str());
            print_ids("want:", want);
            print_ids("got: ", got);
            std::printf("  pieces:");
            for (const std::string & p : tokenizer.pre_tokenize(text)) std::printf(" [%s]", p.c_str());
            std::printf("\n");
        }
    }
    std::printf("%d of %d texts encode as the model's tokenizer encodes them\n", texts - texts_failed, texts);

    std::ifstream decodings(std::filesystem::u8path(args[3]));
    int sequences = 0, sequences_failed = 0;
    while (std::getline(decodings, line)) {
        const size_t tab = line.find('\t');
        const std::vector<int32_t> ids = parse_ids(line.substr(0, tab));
        const std::string want = from_hex(line.substr(tab + 1));
        const std::string got = tokenizer.decode(ids);
        sequences++;
        if (got != want) {
            sequences_failed++;
            std::printf("MISMATCH decoding:\n");
            print_ids("ids: ", ids);
            std::printf("  want: %s\n  got:  %s\n", want.c_str(), got.c_str());
        }
    }
    std::printf("%d of %d sequences of ids decode as the model's tokenizer decodes them\n", sequences - sequences_failed, sequences);
    return texts_failed == 0 && sequences_failed == 0 && texts > 0 && sequences > 0 ? 0 : 1;
}
