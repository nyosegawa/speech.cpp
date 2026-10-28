#include "tokenizer.h"

namespace nemotron_transcribe {

Tokenizer::Tokenizer(const ModelFile & m)
    : bpe_(m, "nemotron-transcribe.tokenizer", Qwen2Pattern::Nemotron, true), added_(m, "nemotron-transcribe.tokenizer") {}

std::vector<int32_t> Tokenizer::encode(const std::string & text) const {
    return added_.encode(text, bpe_);
}

std::string Tokenizer::decode(const std::vector<int32_t> & ids) const {
    return bpe_.decode(ids);
}

}  // namespace nemotron_transcribe
