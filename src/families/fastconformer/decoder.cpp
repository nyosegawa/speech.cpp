#include "decoder.h"

#include <stdexcept>
#include <string>

#include "alsd.h"
#include "tdt.h"

namespace fastconformer {

std::unique_ptr<Decoder> make_decoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint) {
    if (m.one_of("fastconformer.decoder.kind", {"tdt", "rnnt"}) == "tdt") return std::make_unique<TdtDecoder>(m, prediction, joint);
    return std::make_unique<AlsdDecoder>(m, prediction, joint);
}

}  // namespace fastconformer
