#include "decoder.h"

#include <stdexcept>
#include <string>

#include "alsd.h"
#include "tdt.h"

namespace fastconformer {

std::unique_ptr<Decoder> make_decoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint) {
    const std::string decoder = m.str("fastconformer.decoder");
    if (decoder == "tdt") return std::make_unique<TdtDecoder>(m, prediction, joint);
    if (decoder == "rnnt") return std::make_unique<AlsdDecoder>(m, prediction, joint);
    throw std::runtime_error("fastconformer.decoder is \"" + decoder + "\", which is neither \"tdt\" nor \"rnnt\"");
}

}  // namespace fastconformer
