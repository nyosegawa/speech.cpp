// Checks Nemotron 3.5 Transcribe's generation-loop guardrail against the model card's cleanup_generation_loops(), run
// with its own regular expressions by reference/nemotron-transcribe/guardrail_cases.py: each text's cleaned text.
//
// usage: nemotron-transcribe-guardrail-check <model.gguf> <guardrail-cases.tsv>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "args.h"
#include "nemotron-transcribe/layout.h"
#include "nemotron-transcribe/transcript.h"

namespace {

std::string from_hex(const std::string & hex) {
    std::string bytes;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) bytes += (char) std::stoi(hex.substr(i, 2), nullptr, 16);
    return bytes;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <guardrail-cases.tsv>\n", args[0].c_str());
        return 2;
    }
    try {
        const ModelFile model(args[1], nemotron_transcribe::layout);
        const nemotron_transcribe::Guardrail guardrail(model);
        std::ifstream cases(std::filesystem::u8path(args[2]));
        std::string line;
        int total = 0, failed = 0;
        while (std::getline(cases, line)) {
            const size_t tab = line.find('\t');
            const std::string text = from_hex(line.substr(0, tab)), want = from_hex(line.substr(tab + 1));
            const std::string got = guardrail.apply(text);
            total++;
            if (got != want) {
                failed++;
                std::printf("MISMATCH: %s\n  want: %s\n  got:  %s\n", text.c_str(), want.c_str(), got.c_str());
            }
        }
        std::printf("%s: %d of %d texts cleaned as the model card's guardrail cleans them\n", failed == 0 && total > 0 ? "OK" : "FAIL", total - failed,
                    total);
        return failed == 0 && total > 0 ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
