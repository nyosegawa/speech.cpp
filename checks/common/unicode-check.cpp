// Compares the library's NFC and NFKC with the references' on the cases of reference/unicode/normalization_cases.py:
// those of Unicode 9.0 with the tokenizers library's, and those of Unicode 13.0 with Python 3.10's unicodedata.
//
// usage: unicode-check <normalization-cases.tsv>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "args.h"
#include "unicode.h"
#include "utf8.h"

namespace {

std::vector<uint32_t> code_points_of_hex(const std::string & hex) {
    std::string bytes;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) bytes += (char) std::stoi(hex.substr(i, 2), nullptr, 16);
    return decode_utf8(bytes).value();
}

void print_code_points(const char * what, const std::vector<uint32_t> & text) {
    std::printf("  %s", what);
    for (uint32_t cp : text) std::printf(" %04X", cp);
    std::printf("\n");
}

struct Form {
    const char * name;
    std::vector<uint32_t> (*normalize)(const std::vector<uint32_t> &, UnicodeVersion);
    UnicodeVersion version;
    int failed = 0;
};

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 2) {
        std::fprintf(stderr, "usage: %s <normalization-cases.tsv>\n", args[0].c_str());
        return 2;
    }
    std::ifstream cases(std::filesystem::u8path(args[1]));
    if (!cases) {
        std::fprintf(stderr, "cannot open %s\n", args[1].c_str());
        return 1;
    }
    Form forms[] = {{"NFC, Unicode 9.0 (tokenizers)", nfc, UnicodeVersion::V9},
                    {"NFKC, Unicode 9.0 (tokenizers)", nfkc, UnicodeVersion::V9},
                    {"NFC, Unicode 13.0 (Python 3.10)", nfc, UnicodeVersion::V13},
                    {"NFKC, Unicode 13.0 (Python 3.10)", nfkc, UnicodeVersion::V13}};
    int texts = 0, shown = 0;
    std::string line;
    while (std::getline(cases, line)) {
        std::istringstream fields(line);
        std::string hex;
        std::getline(fields, hex, '\t');
        const std::vector<uint32_t> text = code_points_of_hex(hex);
        texts++;
        for (Form & form : forms) {
            std::getline(fields, hex, '\t');
            const std::vector<uint32_t> want = code_points_of_hex(hex);
            const std::vector<uint32_t> got = form.normalize(text, form.version);
            if (got == want) continue;
            form.failed++;
            if (shown++ < 20) {
                std::printf("MISMATCH in %s:\n", form.name);
                print_code_points("text:", text);
                print_code_points("want:", want);
                print_code_points("got: ", got);
            }
        }
    }
    bool ok = texts > 0;
    for (const Form & form : forms) {
        std::printf("%s: %d of %d texts as the reference normalizes them\n", form.name, texts - form.failed, texts);
        ok = ok && form.failed == 0;
    }
    return ok ? 0 : 1;
}
