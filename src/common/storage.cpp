#include "storage.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "error.h"

Shape::Shape(std::initializer_list<int64_t> axes) {
    if (axes.size() > GGML_MAX_DIMS) throw std::logic_error("a shape has more axes than ggml's tensors");
    std::copy(axes.begin(), axes.end(), ne);
}

std::optional<std::array<unsigned long, 3>> release_numbers(const std::string & release) {
    std::array<unsigned long, 3> numbers{};
    size_t at = 0;
    for (size_t i = 0; i < numbers.size(); i++) {
        const size_t end = i + 1 < numbers.size() ? release.find('.', at) : release.size();
        if (end == std::string::npos || end == at || end - at > 9) return std::nullopt;
        for (size_t c = at; c < end; c++) {
            if (release[c] < '0' || release[c] > '9') return std::nullopt;
        }
        numbers[i] = std::stoul(release.substr(at, end - at));
        at = end + 1;
    }
    return numbers;
}

const std::vector<WeightType> & weight_types() {
    // gguf-py's LlamaFileType has two values for Q4_K and for Q5_K, llama.cpp's mixes of types _S and _M; speech.cpp
    // writes the first, MOSTLY_Q4_K_S and MOSTLY_Q5_K_S, whose files hold the type wherever the rows allow it.
    static const std::vector<WeightType> types = {
        {0, GGML_TYPE_F32, kFirstLayoutRelease}, {1, GGML_TYPE_F16, kFirstLayoutRelease}, {7, GGML_TYPE_Q8_0, kFirstLayoutRelease},
        {18, GGML_TYPE_Q6_K, "0.8.0"},           {16, GGML_TYPE_Q5_K, "0.8.0"},           {14, GGML_TYPE_Q4_K, "0.8.0"},
    };
    return types;
}

std::string tensor_type_text(ggml_type type) {
    std::string name = ggml_type_name(type);
    for (char & c : name) c = (char) std::toupper((unsigned char) c);
    return name;
}

const Storage kFloat32 = {{
    {GGML_TYPE_F32, {GGML_TYPE_F32}},
    {GGML_TYPE_F16, {GGML_TYPE_F32}},
    {GGML_TYPE_Q8_0, {GGML_TYPE_F32}},
    {GGML_TYPE_Q6_K, {GGML_TYPE_F32}},
    {GGML_TYPE_Q5_K, {GGML_TYPE_F32}},
    {GGML_TYPE_Q4_K, {GGML_TYPE_F32}},
}};

const Storage kQuantized = {{
    {GGML_TYPE_F32, {GGML_TYPE_F32}},
    {GGML_TYPE_F16, {GGML_TYPE_F16}},
    {GGML_TYPE_Q8_0, {GGML_TYPE_Q8_0, GGML_TYPE_F16}},
    {GGML_TYPE_Q6_K, {GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_F16}},
    {GGML_TYPE_Q5_K, {GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, GGML_TYPE_F16}},
    {GGML_TYPE_Q4_K, {GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, GGML_TYPE_F16}},
}};

const Storage kHalf = {{
    {GGML_TYPE_F32, {GGML_TYPE_F32}},
    {GGML_TYPE_F16, {GGML_TYPE_F16}},
    {GGML_TYPE_Q8_0, {GGML_TYPE_F16}},
    {GGML_TYPE_Q6_K, {GGML_TYPE_F16}},
    {GGML_TYPE_Q5_K, {GGML_TYPE_F16}},
    {GGML_TYPE_Q4_K, {GGML_TYPE_F16}},
}};

std::vector<ggml_type> TensorSpec::types() const {
    std::vector<ggml_type> out;
    for (const Storage::Choice & choice : storage.get().choices) {
        for (ggml_type t : choice.types) {
            if (std::find(out.begin(), out.end(), t) == out.end()) out.push_back(t);
        }
    }
    return out;
}

ggml_type TensorSpec::type_in(ggml_type file) const {
    for (const Storage::Choice & choice : storage.get().choices) {
        if (choice.file != file) continue;
        for (ggml_type t : choice.types) {
            if (shape.ne[0] % ggml_blck_size(t) == 0) return t;
        }
        throw std::logic_error("the storage of " + name + " gives a file of " + tensor_type_text(file) + " no type whose blocks its rows of " +
                               std::to_string(shape.ne[0]) + " values are");
    }
    throw std::logic_error("the storage of " + name + " says nothing of a file of " + tensor_type_text(file));
}

bool release_after(const std::string & a, const std::string & b) {
    const auto numbers = [](const std::string & release) {
        const auto n = release_numbers(release);
        if (!n) throw Error(Fault::File, "\"" + release + "\" is not a release of speech.cpp, MAJOR.MINOR.PATCH");
        return *n;
    };
    return numbers(a) > numbers(b);
}
