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

add_library(speechcommon STATIC
    src/common/model-file.cpp
    src/common/backend.cpp
    src/common/wav.cpp
    src/common/args.cpp
    src/common/language.cpp
)
target_include_directories(speechcommon PUBLIC src/common)
target_link_libraries(speechcommon PUBLIC ggml::ggml)
if (WIN32)
    target_link_libraries(speechcommon PUBLIC shell32)
endif()

add_library(qwen3tts STATIC
    src/families/qwen3-tts/codec.cpp
    src/families/qwen3-tts/talker.cpp
    src/families/qwen3-tts/prompt.cpp
    src/families/qwen3-tts/sampler.cpp
    src/families/qwen3-tts/tokenizer.cpp
    src/families/qwen3-tts/synthesizer.cpp
)
# Code names the family in its includes ("qwen3-tts/codec.h"), since both families have a codec.h.
target_include_directories(qwen3tts PUBLIC src/families)
target_link_libraries(qwen3tts PUBLIC speechcommon)

add_library(irodoritts STATIC
    src/families/irodori-tts/graph.cpp
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
)
target_include_directories(irodoritts PUBLIC src/families)
target_link_libraries(irodoritts PUBLIC speechcommon)

# The C API, include/speech.h, twice: static for the executables, so that each stays one file, and shared as
# libspeech for bindings and other programs. On Windows the two would both write speech.lib, the static one as
# the library and the shared one as its import library, hence speech-static.
set(SPEECH_SOURCES src/speech.cpp src/qwen3-tts-engine.cpp src/irodori-tts-engine.cpp)

add_library(speech-static STATIC ${SPEECH_SOURCES})
target_include_directories(speech-static PUBLIC include)
target_link_libraries(speech-static PUBLIC qwen3tts irodoritts)
target_compile_definitions(speech-static PRIVATE SPEECH_VERSION="${PROJECT_VERSION}")

add_library(speech SHARED ${SPEECH_SOURCES})
target_include_directories(speech PUBLIC include)
target_link_libraries(speech PRIVATE qwen3tts irodoritts)
target_compile_definitions(speech PRIVATE SPEECH_BUILD_SHARED SPEECH_VERSION="${PROJECT_VERSION}" INTERFACE SPEECH_SHARED)
set_target_properties(speech PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
# ggml's static libraries are built with default visibility, so the linker is told to export the C API alone.
if (APPLE)
    target_link_options(speech PRIVATE "LINKER:-exported_symbol,_speech_*")
elseif (UNIX)
    target_link_options(speech PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/speech.map")
    set_property(TARGET speech APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/speech.map)
endif()

# The HTTP server is the one program with a dependency besides ggml: cpp-httplib's single header, vendored in
# vendor/cpp-httplib from yhirose/cpp-httplib at commit cf3693cb5cc0d39b0e6f4122ba89bda9edb5ac6b (v0.59.0), with
# its license (docs/adr/0008). It reads requests with the worker's JSON parser.
add_executable(speech-server tools/server/speech-server.cpp)
target_include_directories(speech-server PRIVATE tools/worker vendor/cpp-httplib)
target_link_libraries(speech-server PRIVATE speech-static)
if (WIN32)
    target_compile_definitions(speech-server PRIVATE NOMINMAX)
    target_link_libraries(speech-server PRIVATE ws2_32)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(speech-server PRIVATE Threads::Threads)
endif()

add_executable(speech-worker tools/worker/speech-worker.cpp)
target_link_libraries(speech-worker PRIVATE speech-static)

add_executable(qwen3-tts tools/qwen3-tts.cpp)
target_link_libraries(qwen3-tts PRIVATE qwen3tts)

add_executable(irodori-tts tools/irodori-tts.cpp)
target_link_libraries(irodori-tts PRIVATE irodoritts)

# The checks compare each stage with the reference dumps; they are built but not released.
add_executable(codec-check checks/codec-check.cpp)
target_link_libraries(codec-check PRIVATE qwen3tts)

add_executable(talker-check checks/talker-check.cpp)
target_link_libraries(talker-check PRIVATE qwen3tts)

add_executable(tokenizer-check checks/tokenizer-check.cpp)
target_link_libraries(tokenizer-check PRIVATE qwen3tts)

add_executable(irodori-text-check checks/irodori-text-check.cpp)
target_link_libraries(irodori-text-check PRIVATE irodoritts)

add_executable(irodori-codec-check checks/irodori-codec-check.cpp)
target_link_libraries(irodori-codec-check PRIVATE irodoritts)

add_executable(irodori-condition-check checks/irodori-condition-check.cpp)
target_link_libraries(irodori-condition-check PRIVATE irodoritts)

add_executable(irodori-dit-check checks/irodori-dit-check.cpp)
target_link_libraries(irodori-dit-check PRIVATE irodoritts)

add_executable(irodori-synthesis-check checks/irodori-synthesis-check.cpp)
# It reads the dump's meta.json with the worker's string parser.
target_include_directories(irodori-synthesis-check PRIVATE tools/worker)
target_link_libraries(irodori-synthesis-check PRIVATE irodoritts)

add_executable(speech-api-check checks/speech-api-check.c)
target_link_libraries(speech-api-check PRIVATE speech)
# The check compares the version the library reports with the one it was built against.
target_compile_definitions(speech-api-check PRIVATE SPEECH_EXPECTED_VERSION="${PROJECT_VERSION}")
if (WIN32)
    target_link_libraries(speech-api-check PRIVATE shell32)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(speech-api-check PRIVATE Threads::Threads)
endif()
