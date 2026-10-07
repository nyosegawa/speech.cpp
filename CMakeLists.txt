cmake_minimum_required(VERSION 3.20)
# VERSION holds the release's number and nothing else does; the library reports it through speech_version().
file(STRINGS VERSION SPEECH_VERSION LIMIT_COUNT 1)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS VERSION)
project(speech-cpp VERSION ${SPEECH_VERSION} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# One executable per tool, with ggml linked in, so that a release is a single file per tool.
option(BUILD_SHARED_LIBS "Build ggml as shared libraries" OFF)
# The shared libspeech links the same static libraries as the executables.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# CI builds the submodule's ggml once, installs it with static libraries, and reuses it while the
# submodule's commit stays the same; Vulkan's shaders alone take minutes to compile.
option(SPEECH_INSTALLED_GGML "Link a ggml installed from the submodule, found through CMAKE_PREFIX_PATH" OFF)
if (SPEECH_INSTALLED_GGML)
    find_package(ggml REQUIRED CONFIG)
else()
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(ggml)
endif()

# Source files are UTF-8; MSVC otherwise reads them in the system code page.
if (MSVC)
    add_compile_options(/utf-8)
endif()
# On Linux the C++ runtime goes into each executable and libspeech.so, so that a release runs on any
# distribution with the glibc it was built against, whatever libstdc++ that distribution has.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_link_options(-static-libstdc++ -static-libgcc)
endif()

add_library(speechcommon STATIC
    src/common/model-file.cpp
    src/common/backend.cpp
    src/common/wav.cpp
    src/common/args.cpp
    src/common/language.cpp
    src/common/graph.cpp
    src/common/resample.cpp
    src/common/fft.cpp
    src/common/log.cpp
    src/common/qwen3-decoder.cpp
    src/common/qwen2-tokenizer.cpp
    src/common/utf8.cpp
    src/common/unicode.cpp
)
target_include_directories(speechcommon PUBLIC src/common)
target_link_libraries(speechcommon PUBLIC ggml::ggml)
# A file of a newer layout is refused naming the release that reads it and this one.
target_compile_definitions(speechcommon PRIVATE SPEECH_VERSION="${PROJECT_VERSION}")
if (WIN32)
    target_link_libraries(speechcommon PUBLIC shell32)
endif()

add_library(qwen3tts STATIC
    src/families/qwen3-tts/layout.cpp
    src/families/qwen3-tts/codec.cpp
    src/families/qwen3-tts/talker.cpp
    src/families/qwen3-tts/prompt.cpp
    src/families/qwen3-tts/sampler.cpp
    src/families/qwen3-tts/synthesizer.cpp
)
# Code names the family in its includes ("qwen3-tts/codec.h"), since both families have a codec.h.
target_include_directories(qwen3tts PUBLIC src/families)
target_link_libraries(qwen3tts PUBLIC speechcommon)

add_library(irodoritts STATIC
    src/families/irodori-tts/layout.cpp
    src/families/irodori-tts/text-normalizer.cpp
    src/families/irodori-tts/tokenizer.cpp
    src/families/irodori-tts/text-encoder.cpp
    src/families/irodori-tts/loudness.cpp
    src/families/irodori-tts/codec.cpp
    src/families/irodori-tts/reference.cpp
    src/families/irodori-tts/speaker-encoder.cpp
    src/families/irodori-tts/duration.cpp
    src/families/irodori-tts/dit.cpp
    src/families/irodori-tts/sampler.cpp
    src/families/irodori-tts/synthesizer.cpp
    src/families/irodori-tts/voice-file.cpp
)
target_include_directories(irodoritts PUBLIC src/families)
target_link_libraries(irodoritts PUBLIC speechcommon)

# FastConformer speech recognition (NVIDIA NeMo): the family.
add_library(fastconformer STATIC
    src/families/fastconformer/layout.cpp
    src/families/fastconformer/frontend.cpp
    src/families/fastconformer/encoder.cpp
    src/families/fastconformer/transducer.cpp
    src/families/fastconformer/decoder.cpp
    src/families/fastconformer/tdt.cpp
    src/families/fastconformer/alsd.cpp
    src/families/fastconformer/rnnt-greedy.cpp
    src/families/fastconformer/detokenizer.cpp
    src/families/fastconformer/times.cpp
    src/families/fastconformer/recognizer.cpp
)
target_include_directories(fastconformer PUBLIC src/families)
target_link_libraries(fastconformer PUBLIC speechcommon)

# Qwen3-ASR speech recognition (Qwen): the family.
add_library(qwen3asr STATIC
    src/families/qwen3-asr/layout.cpp
    src/families/qwen3-asr/frontend.cpp
    src/families/qwen3-asr/encoder.cpp
    src/families/qwen3-asr/tokenizer.cpp
    src/families/qwen3-asr/prompt.cpp
    src/families/qwen3-asr/decoder.cpp
    src/families/qwen3-asr/transcript.cpp
    src/families/qwen3-asr/split.cpp
    src/families/qwen3-asr/recognizer.cpp
)
target_include_directories(qwen3asr PUBLIC src/families)
target_link_libraries(qwen3asr PUBLIC speechcommon)

# The C API, include/speech.h, twice: static for the executables, so that each stays one file, and shared as
# libspeech for bindings and other programs. On Windows the two would both write speech.lib, the static one as
# the library and the shared one as its import library, hence speech-static.
set(SPEECH_SOURCES src/speech.cpp src/info.cpp src/request.cpp src/qwen3-tts-engine.cpp src/irodori-tts-engine.cpp
    src/fastconformer-engine.cpp src/qwen3-asr-engine.cpp)

add_library(speech-static STATIC ${SPEECH_SOURCES})
target_include_directories(speech-static PUBLIC include)
target_link_libraries(speech-static PUBLIC qwen3tts irodoritts fastconformer qwen3asr)
target_compile_definitions(speech-static PRIVATE SPEECH_VERSION="${PROJECT_VERSION}")

add_library(speech SHARED ${SPEECH_SOURCES})
target_include_directories(speech PUBLIC include)
target_link_libraries(speech PRIVATE qwen3tts irodoritts fastconformer qwen3asr)
target_compile_definitions(speech PRIVATE SPEECH_BUILD_SHARED SPEECH_VERSION="${PROJECT_VERSION}" INTERFACE SPEECH_SHARED)
set_target_properties(speech PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
# The shared library's SOVERSION is the C API's major version, which speech.h defines and nothing else repeats.
file(STRINGS include/speech.h SPEECH_API_MAJOR_LINE REGEX "^#define SPEECH_API_VERSION_MAJOR [0-9]+$")
string(REGEX REPLACE "^#define SPEECH_API_VERSION_MAJOR ([0-9]+)$" "\\1" SPEECH_API_VERSION_MAJOR "${SPEECH_API_MAJOR_LINE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS include/speech.h)
set_target_properties(speech PROPERTIES SOVERSION ${SPEECH_API_VERSION_MAJOR})
# ggml's static libraries are built with default visibility, so the linker is told to export the C API alone.
if (APPLE)
    target_link_options(speech PRIVATE "LINKER:-exported_symbol,_speech_*")
elseif (UNIX)
    target_link_options(speech PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/speech.map")
    set_property(TARGET speech APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/speech.map)
endif()

# Every program on the library is one executable, speech, with a subcommand each (docs/adr/0017), the library and
# ggml linked in. The target is speech-cli since the shared library holds the name speech. It has the one dependency
# besides ggml, cpp-httplib's single header for `speech serve`, vendored in vendor/cpp-httplib from
# yhirose/cpp-httplib at commit cf3693cb5cc0d39b0e6f4122ba89bda9edb5ac6b (v0.59.0), with its license.
add_executable(speech-cli
    tools/cli/main.cpp
    tools/cli/tts.cpp
    tools/cli/asr.cpp
    tools/cli/model.cpp
    tools/common/command-line.cpp
    tools/common/json-reader.cpp
    tools/common/request-options.cpp
    tools/worker/worker.cpp
    tools/worker/inbox.cpp
    tools/server/server.cpp
    tools/server/openai-api.cpp
)
set_target_properties(speech-cli PROPERTIES OUTPUT_NAME speech)
target_include_directories(speech-cli PRIVATE tools/common tools/cli vendor/cpp-httplib)
target_link_libraries(speech-cli PRIVATE speech-static)
if (WIN32)
    target_compile_definitions(speech-cli PRIVATE NOMINMAX)
    target_link_libraries(speech-cli PRIVATE ws2_32)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(speech-cli PRIVATE Threads::Threads)
endif()

# The checks compare each stage with the reference dumps; they are built but not released.
add_executable(resample-check checks/resample-check.cpp)
target_link_libraries(resample-check PRIVATE speechcommon)

add_executable(unicode-check checks/unicode-check.cpp)
target_link_libraries(unicode-check PRIVATE speechcommon)

add_executable(codec-check checks/codec-check.cpp)
target_link_libraries(codec-check PRIVATE qwen3tts)

add_executable(talker-check checks/talker-check.cpp)
target_link_libraries(talker-check PRIVATE qwen3tts)

# The sampler against transformers' logits processors, on the cases of reference/qwen3-tts/sampling_cases.py.
add_executable(sampler-check checks/sampler-check.cpp tools/common/json-reader.cpp)
target_include_directories(sampler-check PRIVATE tools/common)
target_link_libraries(sampler-check PRIVATE qwen3tts)

# The tokenizer of a Qwen3-TTS or a Qwen3-ASR file, each its family's.
add_executable(tokenizer-check checks/tokenizer-check.cpp)
target_link_libraries(tokenizer-check PRIVATE qwen3tts qwen3asr)

# The decoder's runs in blocks, on the talker of a Qwen3-TTS file.
add_executable(qwen3-decoder-check checks/qwen3-decoder-check.cpp)
target_link_libraries(qwen3-decoder-check PRIVATE qwen3tts)

add_executable(irodori-text-check checks/irodori-text-check.cpp)
target_link_libraries(irodori-text-check PRIVATE irodoritts)

add_executable(irodori-codec-check checks/irodori-codec-check.cpp)
target_link_libraries(irodori-codec-check PRIVATE irodoritts)

add_executable(irodori-condition-check checks/irodori-condition-check.cpp)
target_link_libraries(irodori-condition-check PRIVATE irodoritts)

add_executable(irodori-dit-check checks/irodori-dit-check.cpp)
target_link_libraries(irodori-dit-check PRIVATE irodoritts)

# It reads the dump's meta.json with the JSON reader of `speech`.
add_executable(irodori-synthesis-check checks/irodori-synthesis-check.cpp tools/common/json-reader.cpp)
target_include_directories(irodori-synthesis-check PRIVATE tools/common)
target_link_libraries(irodori-synthesis-check PRIVATE irodoritts)

add_executable(speech-api-check checks/speech-api-check.c checks/speech-api-recognition.c checks/speech-api-common.c)
target_link_libraries(speech-api-check PRIVATE speech)
# The check compares the version the library reports with the one it was built against.
target_compile_definitions(speech-api-check PRIVATE SPEECH_EXPECTED_VERSION="${PROJECT_VERSION}")
if (WIN32)
    target_link_libraries(speech-api-check PRIVATE shell32)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(speech-api-check PRIVATE Threads::Threads)
endif()

# The checks of FastConformer's stages.
foreach(stage frontend encoder transducer times)
    add_executable(fastconformer-${stage}-check checks/fastconformer-${stage}-check.cpp)
    target_link_libraries(fastconformer-${stage}-check PRIVATE fastconformer)
endforeach()

# The checks of Qwen3-ASR's stages.
foreach(stage frontend encoder)
    add_executable(qwen3-asr-${stage}-check checks/qwen3-asr-${stage}-check.cpp)
    target_link_libraries(qwen3-asr-${stage}-check PRIVATE qwen3asr)
endforeach()
# They read each request's meta.json with the JSON reader of `speech`.
foreach(stage decoder split)
    add_executable(qwen3-asr-${stage}-check checks/qwen3-asr-${stage}-check.cpp tools/common/json-reader.cpp)
    target_include_directories(qwen3-asr-${stage}-check PRIVATE tools/common)
    target_link_libraries(qwen3-asr-${stage}-check PRIVATE qwen3asr)
endforeach()
# Times the stages of a recognition on the dumps' audio, against checks/llama-server-timing.py.
add_executable(qwen3-asr-timing checks/qwen3-asr-timing.cpp tools/common/json-reader.cpp)
target_include_directories(qwen3-asr-timing PRIVATE tools/common)
target_link_libraries(qwen3-asr-timing PRIVATE qwen3asr)
