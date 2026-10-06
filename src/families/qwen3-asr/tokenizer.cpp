#include "tokenizer.h"

#include <algorithm>
#include <stdexcept>

namespace qwen3_asr {

Tokenizer::Tokenizer(const ModelFile & m) : bpe_(m, "qwen3-asr.tokenizer") {
    const std::vector<std::string> tokens = m.str_array("qwen3-asr.tokenizer.tokens");
    for (int32_t id : m.i32_array("qwen3-asr.tokenizer.added_ids")) added_.push_back({tokens[(size_t) id], id});
    std::stable_sort(added_.begin(), added_.end(), [](const auto & a, const auto & b) { return a.first.size() > b.first.size(); });
    special_.assign(tokens.size(), false);
    for (int32_t id : m.i32_array("qwen3-asr.tokenizer.special_ids")) special_[(size_t) id] = true;
}

std::vector<int32_t> Tokenizer::encode(const std::string & text) const {
    std::vector<int32_t> ids;
    const auto encode_between = [&](size_t from, size_t to) {
        if (to == from) return;
        const std::vector<int32_t> piece = bpe_.encode(text.substr(from, to - from));
        ids.insert(ids.end(), piece.begin(), piece.end());
    };
    size_t last = 0;
    for (size_t i = 0; i < text.size();) {
        const auto match = std::find_if(added_.begin(), added_.end(), [&](const auto & a) { return text.compare(i, a.first.size(), a.first) == 0; });
        if (match == added_.end()) {
            i++;
            continue;
        }
        encode_between(last, i);
        ids.push_back(match->second);
        i += match->first.size();
        last = i;
    }
    encode_between(last, text.size());
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t> & ids) const {
    std::vector<int32_t> kept;
    for (int32_t id : ids) {
        if (!special(id)) kept.push_back(id);
    }
    return bpe_.decode(kept);
}

bool Tokenizer::special(int32_t id) const {
    return id >= 0 && (size_t) id < special_.size() && special_[(size_t) id];
}

int32_t Tokenizer::added_id(const std::string & text) const {
    for (const auto & [token, id] : added_) {
        if (token == text) return id;
    }
    throw std::logic_error(text + " is not an added token of the tokenizer, which the layout checks");
}

}  // namespace qwen3_asr
