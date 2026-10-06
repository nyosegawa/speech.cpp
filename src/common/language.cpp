#include "language.h"

#include <cctype>

bool bcp47_matches(const std::string & tag, const std::string & language) {
    if (tag.size() < language.size()) return false;
    for (size_t i = 0; i < language.size(); i++) {
        if (std::tolower((unsigned char) tag[i]) != std::tolower((unsigned char) language[i])) return false;
    }
    return tag.size() == language.size() || tag[language.size()] == '-';
}

bool language_is_auto(const std::string & tag) {
    static const char auto_tag[] = "auto";
    if (tag.size() != sizeof auto_tag - 1) return false;
    for (size_t i = 0; i < tag.size(); i++) {
        if (std::tolower((unsigned char) tag[i]) != auto_tag[i]) return false;
    }
    return true;
}
