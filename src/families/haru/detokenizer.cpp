#include "detokenizer.h"

#include <algorithm>

namespace haru {

StreamDetokenizer::StreamDetokenizer(const Qwen2Tokenizer & tokenizer, std::vector<int32_t> silent)
    : tokenizer_(tokenizer), silent_(std::move(silent)) {}

std::string StreamDetokenizer::push(int32_t token) {
    if (std::find(silent_.begin(), silent_.end(), token) != silent_.end()) {
        if (in_gap_) return "";
        in_gap_ = true;
        return " ";
    }
    in_gap_ = false;
    pending_.push_back(token);
    std::string text = tokenizer_.decode(pending_);
    // U+FFFD at the end, which decode() gives for bytes that end inside a character.
    static const std::string kReplacement = "\xEF\xBF\xBD";
    if (text.size() >= kReplacement.size() && text.compare(text.size() - kReplacement.size(), kReplacement.size(), kReplacement) == 0) return "";
    pending_.clear();
    return text;
}

}  // namespace haru
