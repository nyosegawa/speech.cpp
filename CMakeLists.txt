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

# ggml is linked in statically, so that a release is speech beside the shared libspeech and needs no library of ggml.
option(BUILD_SHARED_LIBS "Build ggml as shared libraries" OFF)
# The shared libspeech links the same static libraries as the executables.
if (NOT EMSCRIPTEN)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
# A build configured with Emscripten's emcmake is the library alone, as WebAssembly for a page's Worker (wasm/). Its C++
# throws and catches as every build does, with WebAssembly's exceptions, which Chrome 95, Firefox 100 and Safari 15.2
# and later run; without them Emscripten aborts at the first throw. Emscripten's toolchain names the processor x86,
# for projects that read a 32-bit platform from it, and ggml compiles its SIMD128 code only for a processor named wasm;
# for any other it falls back to its generic C. ggml repacks weights only for instructions of x86, Arm and RISC-V,
# and its repacking code, never run in WebAssembly, was 50 KB of the 410 KB of a gzipped module of Silero VAD
# (Emscripten 7.0.0, 2026-11-21). With SPEECH_WASM_THREADS every file, ggml's included, is compiled for threads, since
# wasm-ld links no file compiled without atomics into a module of shared memory.
if (EMSCRIPTEN)
    option(SPEECH_WASM_THREADS "Build the WebAssembly library with threads, which a page runs only cross-origin isolated" OFF)
    set(CMAKE_SYSTEM_PROCESSOR wasm32)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_REPACK OFF CACHE BOOL "" FORCE)
    add_compile_options(-fwasm-exceptions)
    add_link_options(-fwasm-exceptions)
    if (SPEECH_WASM_THREADS)
        add_compile_options(-pthread)
        add_link_options(-pthread)
    endif()
endif()
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
    src/common/storage.cpp
    src/common/quantize.cpp
    src/common/backend.cpp
    src/common/wav.cpp
    src/common/args.cpp
    src/common/language.cpp
    src/common/graph.cpp
    src/common/resample.cpp
    src/common/fft.cpp
    src/common/nemo-frontend.cpp
    src/common/nemo-transformer.cpp
    src/common/sortformer-network.cpp
    src/common/sortformer-cache.cpp
    src/common/keys-values.cpp
    src/common/expert-placement.cpp
    src/common/sampling.cpp
    src/common/sentence-segments.cpp
    src/common/log.cpp
    src/common/qwen3-decoder.cpp
    src/common/qwen35-decoder.cpp
    src/common/qwen2-tokenizer.cpp
    src/common/added-vocabulary.cpp
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

# The families the library runs, by their folders in src/families/, each with its engine, src/engines/<family>-engine.cpp.
# A WebAssembly build takes the ones SPEECH_WASM_FAMILIES names, so that a page fetches the code of the models it runs
# and no more; every other build takes them all, which speech and the checks need.
set(SPEECH_ALL_FAMILIES qwen3-tts irodori-tts drama3 haru fastconformer qwen3-asr nemotron-transcribe silero-vad sortformer semamba)
if (EMSCRIPTEN)
    set(SPEECH_WASM_FAMILIES silero-vad CACHE STRING "The families of the WebAssembly build, by their folders in src/families/, or all")
    if (SPEECH_WASM_FAMILIES STREQUAL "all")
        set(SPEECH_FAMILIES ${SPEECH_ALL_FAMILIES})
    else()
        set(SPEECH_FAMILIES ${SPEECH_WASM_FAMILIES})
    endif()
    if (NOT SPEECH_FAMILIES)
        message(FATAL_ERROR "SPEECH_WASM_FAMILIES names no family; name one or more of ${SPEECH_ALL_FAMILIES}, or all")
    endif()
    foreach(family ${SPEECH_FAMILIES})
        if (NOT family IN_LIST SPEECH_ALL_FAMILIES)
            message(FATAL_ERROR "SPEECH_WASM_FAMILIES names ${family}, which is not a family; the families are ${SPEECH_ALL_FAMILIES}, or all")
        endif()
    endforeach()
else()
    set(SPEECH_FAMILIES ${SPEECH_ALL_FAMILIES})
endif()
# A family's code is the library of its folder's name without dashes, made only where the build takes the family. Code
# names the family in its includes ("qwen3-tts/codec.h"), since several families have a codec.h.
function(speech_family family)
    if (NOT family IN_LIST SPEECH_FAMILIES)
        return()
    endif()
    string(REPLACE "-" "" target ${family})
    add_library(${target} STATIC ${ARGN})
    target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/families)
    target_link_libraries(${target} PUBLIC speechcommon)
endfunction()

speech_family(qwen3-tts
    src/families/qwen3-tts/layout.cpp
    src/families/qwen3-tts/codec.cpp
    src/families/qwen3-tts/talker.cpp
    src/families/qwen3-tts/prompt.cpp
    src/families/qwen3-tts/sampler.cpp
    src/families/qwen3-tts/graph-sampler.cpp
    src/families/qwen3-tts/synthesizer.cpp
)

speech_family(irodori-tts
    src/families/irodori-tts/layout.cpp
    src/families/irodori-tts/text-normalizer.cpp
    src/families/irodori-tts/tokenizer.cpp
    src/families/irodori-tts/text-encoder.cpp
    src/families/irodori-tts/layers.cpp
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

# FastConformer speech recognition (NVIDIA NeMo): the family.
speech_family(fastconformer
    src/families/fastconformer/layout.cpp
    src/families/fastconformer/encoder.cpp
    src/families/fastconformer/transducer.cpp
    src/families/fastconformer/kda.cpp
    src/families/fastconformer/decoder-step.cpp
    src/families/fastconformer/decoder.cpp
    src/families/fastconformer/tdt.cpp
    src/families/fastconformer/boosting.cpp
    src/families/fastconformer/words.cpp
    src/families/fastconformer/alsd.cpp
    src/families/fastconformer/rnnt-greedy.cpp
    src/families/fastconformer/detokenizer.cpp
    src/families/fastconformer/sentencepiece.cpp
    src/families/fastconformer/times.cpp
    src/families/fastconformer/recognizer.cpp
    src/families/fastconformer/vocabulary.cpp
    src/families/fastconformer/guard.cpp
)

# Qwen3-ASR speech recognition (Qwen): the family.
speech_family(qwen3-asr
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

# Silero VAD speech detection (Silero Team): the family.
speech_family(silero-vad
    src/families/silero-vad/layout.cpp
    src/families/silero-vad/network.cpp
    src/families/silero-vad/regions.cpp
    src/families/silero-vad/detector.cpp
)

speech_family(sortformer
    src/families/sortformer/layout.cpp
    src/families/sortformer/segments.cpp
    src/families/sortformer/diarizer.cpp
)
# The speaker cache's scores and the segments' times are float32 arithmetic whose roundings decide which frames the cache
# keeps and whether two padded segments meet, rounded after each operation as torch rounds them: a fused multiply-add
# moves a value by a unit in its last place. GCC fuses across statements by default and Clang within one; MSVC's
# /fp:precise does not fuse.
if (NOT MSVC)
    set_source_files_properties(src/common/sortformer-cache.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
    if (TARGET sortformer)
        target_compile_options(sortformer PRIVATE -ffp-contract=off)
    endif()
endif()

# NVIDIA NemotronLabs RE-USE speech enhancement (NVIDIA): the family of SEMamba.
speech_family(semamba
    src/families/semamba/layout.cpp
    src/families/semamba/stft.cpp
    src/families/semamba/network.cpp
    src/families/semamba/enhancer.cpp
)

# NVIDIA Nemotron 3.5 Transcribe speech recognition (NVIDIA): the family.
speech_family(nemotron-transcribe
    src/families/nemotron-transcribe/layout.cpp
    src/families/nemotron-transcribe/tokenizer.cpp
    src/families/nemotron-transcribe/encoder.cpp
    src/families/nemotron-transcribe/speakers.cpp
    src/families/nemotron-transcribe/perception.cpp
    src/families/nemotron-transcribe/language-model.cpp
    src/families/nemotron-transcribe/transcript.cpp
    src/families/nemotron-transcribe/recognizer.cpp
    src/families/nemotron-transcribe/ctc-head.cpp
    src/families/nemotron-transcribe/ctc-tokenizer.cpp
    src/families/nemotron-transcribe/ctc-text.cpp
    src/families/nemotron-transcribe/alignment.cpp
    src/families/nemotron-transcribe/timing.cpp
)
# The alignment computes as NumPy and PyTorch do, every product and sum rounded on its own.
if(NOT MSVC)
    set_source_files_properties(src/families/nemotron-transcribe/alignment.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
endif()

# Fish Audio Drama 3 speech synthesis (Fish Audio): the family.
speech_family(drama3
    src/families/drama3/layout.cpp
    src/families/drama3/codec.cpp
    src/families/drama3/sampler.cpp
    src/families/drama3/audio-decoder.cpp
    src/families/drama3/text-model.cpp
    src/families/drama3/prompt.cpp
    src/families/drama3/chunking.cpp
    src/families/drama3/synthesizer.cpp
    src/families/drama3/voice-file.cpp
    src/families/drama3/placement.cpp
)

# Haru full-duplex spoken dialogue (higurashi): the family.
speech_family(haru
    src/families/haru/layout.cpp
    src/families/haru/codec.cpp
    src/families/haru/head.cpp
    src/families/haru/sampler.cpp
    src/families/haru/detokenizer.cpp
    src/families/haru/prompt.cpp
    src/families/haru/dialogue.cpp
    src/families/haru/voice-file.cpp
)

# The C API, include/speech.h, twice: static for the executables, so that each stays one file, and shared as
# libspeech for bindings and other programs. On Windows the two would both write speech.lib, the static one as
# the library and the shared one as its import library, hence speech-static. The table of families in
# src/api/speech.cpp holds the line of each family the build takes, which SPEECH_FAMILY_<FAMILY> marks.
set(SPEECH_SOURCES src/api/speech.cpp src/api/info.cpp src/api/request.cpp src/api/detection.cpp src/api/diarization.cpp src/api/enhancement.cpp src/api/dialogue.cpp src/api/voice.cpp)
set(SPEECH_FAMILY_LIBRARIES "")
set(SPEECH_FAMILY_DEFINITIONS "")
foreach(family ${SPEECH_FAMILIES})
    list(APPEND SPEECH_SOURCES src/engines/${family}-engine.cpp)
    string(REPLACE "-" "" library ${family})
    list(APPEND SPEECH_FAMILY_LIBRARIES ${library})
    string(MAKE_C_IDENTIFIER "SPEECH_FAMILY_${family}" definition)
    string(TOUPPER ${definition} definition)
    list(APPEND SPEECH_FAMILY_DEFINITIONS ${definition})
endforeach()
set_source_files_properties(src/api/speech.cpp PROPERTIES COMPILE_DEFINITIONS "${SPEECH_FAMILY_DEFINITIONS}")

# A WebAssembly build makes the C API once, as libspeech.js and libspeech.wasm (libspeech-threads.js and .wasm with
# threads), an ES module for a Worker that wasm/speech.mjs wraps, and nothing else: the executables and the checks run
# processes, terminals, devices and the network, which a browser does not give them. The module exports every function
# speech.h declares, and Emscripten's file system with WORKERFS, which reads a model from a Blob as it is asked, so
# that the weights are held once, in the module's memory. The memory grows as the model needs, up to the 4 GB a
# 32-bit address reaches, or 1 GB with threads: Safari on iOS counts a shared memory's maximum against the memory of the
# process as it makes it, and refused one of 2 GB on iOS 16.2. The module with threads starts a Worker for each of the
# browser's logical processors before it runs, since a thread that the library starts while it computes would wait for
# a Worker to start, which it cannot until the module returns; a thread past them fails to start rather than waits.
# Emscripten gives the stack 64 KB, and ggml builds a graph by recursing once for each node of its longest chain:
# RE-USE's graph on 2 s of audio overflowed 64 KB and ran in 128 KB (Emscripten 7.0.0, 2026-11-21). The stack is 1 MB,
# as on Windows' main thread, the least the library runs on in a native build.
if (EMSCRIPTEN)
    if (SPEECH_WASM_THREADS)
        set(output libspeech-threads)
        set(memory -sMAXIMUM_MEMORY=1GB -sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency -sPTHREAD_POOL_SIZE_STRICT=2)
    else()
        set(output libspeech)
        set(memory -sMAXIMUM_MEMORY=4GB)
    endif()
    file(STRINGS include/speech.h api_lines REGEX "^SPEECH_API .*speech_[a-z0-9_]+\\(")
    set(exports _malloc _free)
    foreach(line ${api_lines})
        string(REGEX MATCH "speech_[a-z0-9_]+\\(" name "${line}")
        string(REPLACE "(" "" name "${name}")
        list(APPEND exports _${name})
    endforeach()
    list(JOIN exports "," exports)
    add_executable(speech-wasm ${SPEECH_SOURCES})
    target_include_directories(speech-wasm PRIVATE include src/engines)
    target_link_libraries(speech-wasm PRIVATE ${SPEECH_FAMILY_LIBRARIES})
    target_compile_definitions(speech-wasm PRIVATE SPEECH_VERSION="${PROJECT_VERSION}")
    set_target_properties(speech-wasm PROPERTIES OUTPUT_NAME ${output} SUFFIX .js)
    target_link_options(speech-wasm PRIVATE
        -sMODULARIZE -sEXPORT_ES6 -sENVIRONMENT=worker
        -sALLOW_MEMORY_GROWTH -sSTACK_SIZE=1MB ${memory}
        -sEXPORTED_FUNCTIONS=${exports}
        -sEXPORTED_RUNTIME_METHODS=FS,WORKERFS,UTF8ToString,stringToNewUTF8,HEAPU8,HEAP32,HEAPU32,HEAPF32,HEAPF64,HEAP64
        -lworkerfs.js)
    # The module that wraps the library, the page that tries it and the server that serves the page cross-origin
    # isolated go beside it, as a release's archive holds them.
    foreach(file speech.mjs wav.mjs worker.mjs page.mjs index.html serve.py)
        configure_file(wasm/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file} COPYONLY)
    endforeach()
    return()
endif()

add_library(speech-static STATIC ${SPEECH_SOURCES})
target_include_directories(speech-static PUBLIC include PRIVATE src/engines)
target_link_libraries(speech-static PUBLIC ${SPEECH_FAMILY_LIBRARIES})
target_compile_definitions(speech-static PRIVATE SPEECH_VERSION="${PROJECT_VERSION}")

add_library(speech SHARED ${SPEECH_SOURCES})
target_include_directories(speech PUBLIC include PRIVATE src/engines)
target_link_libraries(speech PRIVATE ${SPEECH_FAMILY_LIBRARIES})
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
    target_link_options(speech PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/api/speech.map")
    set_property(TARGET speech APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/api/speech.map)
endif()

# Files that go into speech, the catalog of the models it fetches by name and the files of its page, as arrays of bytes
# with a zero after them, since MSVC refuses a string literal of more than 16380 bytes. A change to a file configures
# the build again.
function(speech_embed NAME FILE OUT)
    file(READ ${FILE} hex HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
    set(${OUT} "extern const unsigned char ${NAME}[] = {${bytes}0};\n" PARENT_SCOPE)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${FILE})
endfunction()

speech_embed(speech_catalog_json ${CMAKE_CURRENT_SOURCE_DIR}/tools/catalog/catalog.json SPEECH_CATALOG_SOURCE)
file(CONFIGURE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/catalog-json.cpp CONTENT "@SPEECH_CATALOG_SOURCE@" @ONLY)

file(GLOB SPEECH_PAGE_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/server/page/*.html
     ${CMAKE_CURRENT_SOURCE_DIR}/tools/server/page/*.css ${CMAKE_CURRENT_SOURCE_DIR}/tools/server/page/*.js)
list(SORT SPEECH_PAGE_FILES)
set(SPEECH_PAGE_SOURCE "#include \"page-files.h\"\n")
set(SPEECH_PAGE_TABLE "")
foreach(path ${SPEECH_PAGE_FILES})
    get_filename_component(name ${path} NAME)
    string(MAKE_C_IDENTIFIER "speech_page_${name}" id)
    speech_embed(${id} ${path} bytes)
    string(APPEND SPEECH_PAGE_SOURCE "${bytes}")
    string(APPEND SPEECH_PAGE_TABLE "    {\"${name}\", ${id}, sizeof ${id} - 1},\n")
endforeach()
list(LENGTH SPEECH_PAGE_FILES SPEECH_PAGE_COUNT)
string(APPEND SPEECH_PAGE_SOURCE "extern const PageFile speech_page_files[] = {\n${SPEECH_PAGE_TABLE}};\n"
       "extern const size_t speech_page_file_count = ${SPEECH_PAGE_COUNT};\n")
file(CONFIGURE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/page-files.cpp CONTENT "@SPEECH_PAGE_SOURCE@" @ONLY)

# Every program on the library is one executable, speech, with a subcommand each, the library and
# ggml linked in. The target is speech-cli since the shared library holds the name speech. It has two dependencies
# besides ggml, each a single header vendored with its license: cpp-httplib for `speech serve`, in vendor/cpp-httplib from
# yhirose/cpp-httplib at commit cf3693cb5cc0d39b0e6f4122ba89bda9edb5ac6b (v0.59.0), and miniaudio for the microphone of
# `speech asr --live` and the microphone and audio output of `speech talk --live`, in vendor/miniaudio from
# mackron/miniaudio at commit 9634bedb5b5a2ca38c1ee7108a9358a4e233f14d (v0.11.25).
add_executable(speech-cli
    tools/cli/main.cpp
    tools/cli/tts.cpp
    tools/cli/asr.cpp
    tools/cli/asr-stream.cpp
    tools/cli/microphone.cpp
    tools/cli/speaker.cpp
    tools/cli/miniaudio.c
    tools/cli/vad.cpp
    tools/cli/diarize.cpp
    tools/cli/enhance.cpp
    tools/cli/talk.cpp
    tools/cli/wave-output.cpp
    tools/cli/model.cpp
    tools/cli/models.cpp
    tools/cli/quantize.cpp
    tools/cli/update.cpp
    tools/catalog/catalog.cpp
    tools/catalog/cache.cpp
    tools/catalog/curl.cpp
    tools/catalog/fetch.cpp
    tools/catalog/release.cpp
    tools/catalog/sha256.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/catalog-json.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/page-files.cpp
    tools/common/command-line.cpp
    tools/common/enhancement.cpp
    tools/common/json-reader.cpp
    tools/common/request-options.cpp
    tools/common/openai-error.cpp
    tools/common/realtime-events.cpp
    tools/common/realtime-session.cpp
    tools/common/regions.cpp
    tools/common/speakers.cpp
    tools/common/sentences.cpp
    tools/common/spelling.cpp
    tools/common/terminal.cpp
    tools/common/transcript.cpp
    tools/common/utterances.cpp
    tools/worker/worker.cpp
    tools/worker/inbox.cpp
    tools/server/server.cpp
    tools/server/audio-speech.cpp
    tools/server/audio-enhancements.cpp
    tools/server/audio-stream.cpp
    tools/server/openai-api.cpp
    tools/server/served-models.cpp
    tools/server/log.cpp
    tools/server/access.cpp
    tools/server/page.cpp
    tools/server/realtime.cpp
    tools/server/dialogue.cpp
)
set_target_properties(speech-cli PROPERTIES OUTPUT_NAME speech)
target_include_directories(speech-cli PRIVATE tools/common tools/cli tools/catalog tools/server vendor/cpp-httplib vendor/miniaudio)
target_link_libraries(speech-cli PRIVATE speech-static)
# miniaudio captures and plays audio and nothing else, so that its decoders, encoders and engine stay out of speech, and
# without its null backend, which opens "NULL Capture Device" when no real one does and gives silence, where speech asr
# --live is to say that the microphone did not open. On macOS it links the system's audio frameworks; elsewhere it loads
# the system's audio library when a device opens, so that no build needs the development packages of ALSA or PulseAudio and a
# Linux release needs no audio library to start.
target_compile_definitions(speech-cli PRIVATE MA_NO_DECODING MA_NO_ENCODING MA_NO_GENERATION MA_NO_ENGINE MA_NO_NODE_GRAPH
                           MA_NO_RESOURCE_MANAGER MA_NO_NULL)
if (APPLE)
    target_compile_definitions(speech-cli PRIVATE MA_NO_RUNTIME_LINKING)
    target_link_libraries(speech-cli PRIVATE "-framework CoreFoundation" "-framework CoreAudio" "-framework AudioToolbox")
elseif (UNIX)
    target_link_libraries(speech-cli PRIVATE ${CMAKE_DL_LIBS})
endif()
if (WIN32)
    target_compile_definitions(speech-cli PRIVATE NOMINMAX)
    target_link_libraries(speech-cli PRIVATE ws2_32)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(speech-cli PRIVATE Threads::Threads)
endif()

# The checks compare each stage with the reference dumps; they are built but not released. compare.h, npy.h and
# reference-dumps.h at the top of checks/ are shared by the checks of several folders.
add_executable(resample-check checks/common/resample-check.cpp)
target_include_directories(resample-check PRIVATE checks)
target_link_libraries(resample-check PRIVATE speechcommon)

add_executable(unicode-check checks/common/unicode-check.cpp)
target_link_libraries(unicode-check PRIVATE speechcommon)

add_executable(qwen3-tts-codec-check checks/qwen3-tts/qwen3-tts-codec-check.cpp)
target_include_directories(qwen3-tts-codec-check PRIVATE checks)
target_link_libraries(qwen3-tts-codec-check PRIVATE qwen3tts)

add_executable(qwen3-tts-talker-check checks/qwen3-tts/qwen3-tts-talker-check.cpp tools/common/json-reader.cpp)
target_include_directories(qwen3-tts-talker-check PRIVATE checks tools/common)
target_link_libraries(qwen3-tts-talker-check PRIVATE qwen3tts)

# The sampler against transformers' logits processors, on the cases of reference/qwen3-tts/sampling_cases.py.
add_executable(qwen3-tts-sampler-check checks/qwen3-tts/qwen3-tts-sampler-check.cpp tools/common/json-reader.cpp)
target_include_directories(qwen3-tts-sampler-check PRIVATE checks tools/common)
target_link_libraries(qwen3-tts-sampler-check PRIVATE qwen3tts)

# The picks built into a graph against the sampler, on the same cases, on the CPU or a GPU.
add_executable(qwen3-tts-graph-sampler-check checks/qwen3-tts/qwen3-tts-graph-sampler-check.cpp tools/common/json-reader.cpp)
target_include_directories(qwen3-tts-graph-sampler-check PRIVATE checks tools/common)
target_link_libraries(qwen3-tts-graph-sampler-check PRIVATE qwen3tts)

# The tokenizer of a Qwen3-TTS or a Qwen3-ASR file, each its family's.
add_executable(tokenizer-check checks/common/tokenizer-check.cpp)
target_link_libraries(tokenizer-check PRIVATE qwen3tts qwen3asr nemotrontranscribe)

# The decoder's runs in blocks, on the talker of a Qwen3-TTS file.
add_executable(qwen3-decoder-check checks/common/qwen3-decoder-check.cpp)
target_include_directories(qwen3-decoder-check PRIVATE checks)
target_link_libraries(qwen3-decoder-check PRIVATE qwen3tts)

add_executable(irodori-text-check checks/irodori-tts/irodori-text-check.cpp tools/common/json-reader.cpp)
target_include_directories(irodori-text-check PRIVATE checks tools/common)
target_link_libraries(irodori-text-check PRIVATE irodoritts)

add_executable(irodori-window-check checks/irodori-tts/irodori-window-check.cpp)
target_link_libraries(irodori-window-check PRIVATE irodoritts)

add_executable(irodori-codec-check checks/irodori-tts/irodori-codec-check.cpp tools/common/json-reader.cpp)
target_include_directories(irodori-codec-check PRIVATE checks tools/common)
target_link_libraries(irodori-codec-check PRIVATE irodoritts)

# These read the dump's meta.json with the JSON reader of `speech`.
add_executable(irodori-condition-check checks/irodori-tts/irodori-condition-check.cpp tools/common/json-reader.cpp)
target_include_directories(irodori-condition-check PRIVATE checks tools/common)
target_link_libraries(irodori-condition-check PRIVATE irodoritts)

add_executable(irodori-dit-check checks/irodori-tts/irodori-dit-check.cpp tools/common/json-reader.cpp)
target_include_directories(irodori-dit-check PRIVATE checks tools/common)
target_link_libraries(irodori-dit-check PRIVATE irodoritts)

add_executable(irodori-synthesis-check checks/irodori-tts/irodori-synthesis-check.cpp tools/common/json-reader.cpp)
target_include_directories(irodori-synthesis-check PRIVATE checks tools/common)
target_link_libraries(irodori-synthesis-check PRIVATE irodoritts)

# What speech quantize refuses of the weights it is given, on a model file of a layout of the check's own.
add_executable(quantize-check checks/common/quantize-check.cpp)
target_link_libraries(quantize-check PRIVATE speechcommon)

add_executable(speech-api-check checks/api/speech-api-check.c checks/api/speech-api-irodori.c checks/api/speech-api-drama3.c checks/api/speech-api-recognition.c
               checks/api/speech-api-detection.c checks/api/speech-api-detection-stream.c checks/api/speech-api-diarization.c
               checks/api/speech-api-enhancement.c checks/api/speech-api-dialogue.c
               checks/api/speech-api-common.c
               checks/api/speech-api-platform.c)
target_link_libraries(speech-api-check PRIVATE speech)
# The check compares the version the library reports with the one it was built against.
target_compile_definitions(speech-api-check PRIVATE SPEECH_EXPECTED_VERSION="${PROJECT_VERSION}")
if (WIN32)
    target_link_libraries(speech-api-check PRIVATE shell32)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(speech-api-check PRIVATE Threads::Threads)
endif()
# C's math functions, sinf among them, are a library of their own on Linux, which glibc does not link by itself.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(speech-api-check PRIVATE m)
endif()

# The checks of FastConformer's stages.
foreach(stage frontend encoder transducer times sentencepiece)
    add_executable(fastconformer-${stage}-check checks/fastconformer/fastconformer-${stage}-check.cpp)
    target_include_directories(fastconformer-${stage}-check PRIVATE checks)
    target_link_libraries(fastconformer-${stage}-check PRIVATE fastconformer)
endforeach()
# Haas-ASR's custom vocabulary, on the cases of reference/fastconformer/vocabulary_cases.py, and the recognition with
# it, on the dumps of dump.py --words, which both read with the JSON reader of `speech`.
foreach(stage vocabulary words)
    add_executable(fastconformer-${stage}-check checks/fastconformer/fastconformer-${stage}-check.cpp tools/common/json-reader.cpp)
    target_include_directories(fastconformer-${stage}-check PRIVATE checks tools/common)
    target_link_libraries(fastconformer-${stage}-check PRIVATE fastconformer)
endforeach()

# The checks of Qwen3-ASR's stages.
foreach(stage frontend encoder)
    add_executable(qwen3-asr-${stage}-check checks/qwen3-asr/qwen3-asr-${stage}-check.cpp)
    target_include_directories(qwen3-asr-${stage}-check PRIVATE checks)
    target_link_libraries(qwen3-asr-${stage}-check PRIVATE qwen3asr)
endforeach()
# They read each request's meta.json with the JSON reader of `speech`.
foreach(stage decoder split)
    add_executable(qwen3-asr-${stage}-check checks/qwen3-asr/qwen3-asr-${stage}-check.cpp tools/common/json-reader.cpp)
    target_include_directories(qwen3-asr-${stage}-check PRIVATE checks tools/common)
    target_link_libraries(qwen3-asr-${stage}-check PRIVATE qwen3asr)
endforeach()
# Times the stages of a recognition on the dumps' audio, against measure/llama-server-timing.py.
add_executable(fastconformer-timing measure/fastconformer-timing.cpp)
target_include_directories(fastconformer-timing PRIVATE checks tools/common)
target_link_libraries(fastconformer-timing PRIVATE fastconformer)

add_executable(qwen3-asr-timing measure/qwen3-asr-timing.cpp tools/common/json-reader.cpp)
target_include_directories(qwen3-asr-timing PRIVATE checks tools/common)
target_link_libraries(qwen3-asr-timing PRIVATE qwen3asr)

# Times the stages of a synthesis on the texts given.
add_executable(qwen3-tts-timing measure/qwen3-tts-timing.cpp)
target_link_libraries(qwen3-tts-timing PRIVATE qwen3tts)

add_executable(irodori-tts-timing measure/irodori-tts-timing.cpp)
target_link_libraries(irodori-tts-timing PRIVATE irodoritts)

# Times a conversation's start and steps on a recording of the user.
add_executable(haru-timing measure/haru-timing.cpp)
target_include_directories(haru-timing PRIVATE checks)
target_link_libraries(haru-timing PRIVATE haru)

# Times the stages of an enhancement on a chunk of noise.
add_executable(semamba-timing measure/semamba-timing.cpp)
target_include_directories(semamba-timing PRIVATE checks)
target_link_libraries(semamba-timing PRIVATE semamba)

# Silero VAD's stages and regions against the dumps of reference/silero-vad/dump.py, read with the JSON reader of `speech`.
add_executable(silero-vad-check checks/silero-vad/silero-vad-check.cpp tools/common/json-reader.cpp)
target_include_directories(silero-vad-check PRIVATE checks tools/common)
target_link_libraries(silero-vad-check PRIVATE silerovad)

# Nemotron 3.5 Transcribe's stages against the dumps of reference/nemotron-transcribe/dump.py.
foreach(stage encoder speakers perception decoder recognizer guardrail ctc-head alignment timestamps)
    add_executable(nemotron-transcribe-${stage}-check checks/nemotron-transcribe/nemotron-transcribe-${stage}-check.cpp tools/common/json-reader.cpp)
    target_include_directories(nemotron-transcribe-${stage}-check PRIVATE checks tools/common)
    target_link_libraries(nemotron-transcribe-${stage}-check PRIVATE nemotrontranscribe)
endforeach()

# Sortformer's stages against the dumps of reference/sortformer/dump.py, read with the JSON reader of `speech`.
foreach(stage frontend network cache segments stream)
    add_executable(sortformer-${stage}-check checks/sortformer/sortformer-${stage}-check.cpp tools/common/json-reader.cpp)
    target_include_directories(sortformer-${stage}-check PRIVATE checks tools/common)
    target_link_libraries(sortformer-${stage}-check PRIVATE sortformer)
endforeach()

# SEMamba's stages against the dumps of reference/semamba/dump.py.
foreach(stage stft network enhance)
    add_executable(semamba-${stage}-check checks/semamba/semamba-${stage}-check.cpp tools/common/json-reader.cpp)
    target_include_directories(semamba-${stage}-check PRIVATE checks tools/common)
    target_link_libraries(semamba-${stage}-check PRIVATE semamba)
endforeach()

# Drama 3's codec against the dumps of reference/drama3/dump.py.
add_executable(drama3-codec-check checks/drama3/drama3-codec-check.cpp)
target_include_directories(drama3-codec-check PRIVATE checks)
target_link_libraries(drama3-codec-check PRIVATE drama3)

# Haru's codec against the dumps of reference/haru/dump.py.
add_executable(haru-codec-check checks/haru/haru-codec-check.cpp)
target_include_directories(haru-codec-check PRIVATE checks)
target_link_libraries(haru-codec-check PRIVATE haru)

# Haru's output and MeanFlow head against the dumps of reference/haru/dump.py.
add_executable(haru-text-check checks/haru/haru-text-check.cpp)
target_include_directories(haru-text-check PRIVATE checks)
target_link_libraries(haru-text-check PRIVATE haru)

add_executable(haru-head-check checks/haru/haru-head-check.cpp tools/common/json-reader.cpp)
target_include_directories(haru-head-check PRIVATE checks tools/common)
target_link_libraries(haru-head-check PRIVATE haru)

# Haru's prompt against the cases of reference/haru/prompt_cases.py and the dumps of reference/haru/dump.py.
add_executable(haru-prompt-check checks/haru/haru-prompt-check.cpp tools/common/json-reader.cpp)
target_include_directories(haru-prompt-check PRIVATE checks tools/common)
target_link_libraries(haru-prompt-check PRIVATE haru)

# A whole Haru conversation against a dump of reference/haru/dump.py, its draws replayed.
add_executable(haru-dialogue-check checks/haru/haru-dialogue-check.cpp tools/common/json-reader.cpp)
target_include_directories(haru-dialogue-check PRIVATE checks tools/common)
target_link_libraries(haru-dialogue-check PRIVATE haru)

# Haru's text sampling against the cases of reference/haru/sampling_cases.py.
add_executable(haru-sampling-check checks/haru/haru-sampling-check.cpp tools/common/json-reader.cpp)
target_include_directories(haru-sampling-check PRIVATE checks tools/common)
target_link_libraries(haru-sampling-check PRIVATE haru)

# Drama 3's sampler against fish-speech's, on the cases of reference/drama3/sampling_cases.py.
add_executable(drama3-sampler-check checks/drama3/drama3-sampler-check.cpp tools/common/json-reader.cpp)
target_include_directories(drama3-sampler-check PRIVATE checks tools/common)
target_link_libraries(drama3-sampler-check PRIVATE drama3)

add_executable(drama3-audio-decoder-check checks/drama3/drama3-audio-decoder-check.cpp tools/common/json-reader.cpp)
target_include_directories(drama3-audio-decoder-check PRIVATE checks tools/common)
target_link_libraries(drama3-audio-decoder-check PRIVATE drama3)

add_executable(drama3-text-model-check checks/drama3/drama3-text-model-check.cpp)
target_include_directories(drama3-text-model-check PRIVATE checks)
target_link_libraries(drama3-text-model-check PRIVATE drama3)

# Drama 3's tokenizer, prompt and chunking against fish-speech's, on the cases of reference/drama3/prompt_cases.py.
add_executable(drama3-prompt-check checks/drama3/drama3-prompt-check.cpp tools/common/json-reader.cpp)
target_include_directories(drama3-prompt-check PRIVATE checks tools/common)
target_link_libraries(drama3-prompt-check PRIVATE drama3)

add_executable(fft-check checks/common/fft-check.cpp)
target_link_libraries(fft-check PRIVATE speechcommon)
