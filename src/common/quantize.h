#pragma once

#include <string>

#include "model-file.h"

/**
 * Writes the model file at `in`, a file of `layout` every tensor of which is F32, to `out` in the weight type `type`,
 * which is not F32: each tensor in the type that the layout's storage gives it in a file of that type
 * (TensorSpec::type_in()), as ggml_quantize_chunk() converts it, and every key and tensor in the order `in` has them,
 * as `in` has them, but general.file_type, which names `type`, general.quantization_version after it in a quantized
 * file, and speech.requires, which names the latest of `in`'s own, the first release that reads `type`'s
 * general.file_type, and the first release whose reader of the family takes each tensor in its type
 * (TensorSpec::first_release()). The file written is read back as the layout reads a file, and removed when that or the
 * writing fails.
 *
 * Every failure throws an Error naming the input at fault as the C API names it: "model_path" for `in`, which a file
 * that is not F32 throws as the caller's mistake and a weight that is not finite as a file error, and "out_path" for
 * `out`.
 */
void quantize_model_file(const std::string & in, const std::string & out, const WeightType & type, const Layout & layout);
