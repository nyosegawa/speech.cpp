# speech.cpp

Speech synthesis and speech recognition in C++ on [ggml](https://github.com/ggml-org/ggml), as a library with a C
API (`include/speech.h`) and the programs built on it, among them the worker process that
[ASIST](https://github.com/nyosegawa/asist) starts. It targets Metal, Vulkan and CUDA; it is checked on
Metal, on Vulkan (NVIDIA) and on the CPU. Every stage of a port is checked against the official
implementation.

| Family | Model | Task | Converted weights |
|---|---|---|---|
| Qwen3-TTS | Qwen3-TTS 12Hz 0.6B and 1.7B CustomVoice | speech synthesis with the named speakers, streamed frame by frame | [sakasegawa/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF), [sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF) |
| Irodori-TTS | Irodori-TTS v4.1-Small-MF and v4.1-Small | Japanese speech synthesis in the voice of a reference recording, a sentence at a time, streamed as the codec decodes it | [sakasegawa/Irodori-TTS-v4.1-Small-MF-GGUF](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-MF-GGUF), [sakasegawa/Irodori-TTS-v4.1-Small-GGUF](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-GGUF) |
| FastConformer | NVIDIA's parakeet-tdt_ctc-0.6b-ja and parakeet-tdt-0.6b-v3, reazon-research's reazonspeech-nemo-v2 | speech recognition with the decoder NeMo uses for each, a recording at a time: Japanese, 25 European languages the model tells apart itself, and Japanese recordings of many minutes | [sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF](https://huggingface.co/sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF), [sakasegawa/parakeet-tdt-0.6b-v3-GGUF](https://huggingface.co/sakasegawa/parakeet-tdt-0.6b-v3-GGUF), [sakasegawa/reazonspeech-nemo-v2-GGUF](https://huggingface.co/sakasegawa/reazonspeech-nemo-v2-GGUF) |

## Binaries

[Releases](https://github.com/nyosegawa/speech.cpp/releases) carry, for macOS arm64 (Metal), Windows x64
(Vulkan) and Linux x64 (Vulkan, and the CPU alone), `speech-worker-<version>-<platform>.zip` with the worker
alone, which is what ASIST bundles, and `speech-cpp-tools-<version>-<platform>.zip` with the command-line
tools `speech-tts` and `speech-asr`, the HTTP server `speech-server` and the shared library with its header
(`libspeech.3.dylib` with the link `libspeech.dylib`, `libspeech.so.3` with the link `libspeech.so`, or `speech.dll`
with its import library `speech.lib`, and `speech.h`; 3 is the C API's major version), with their SHA-256 sums. A release is the tag `v<version>` of the number in the file `VERSION`, which CI checks before it
publishes; the library reports the same number through `speech_version()`, and the worker in its `ready`
message. Versions follow [Semantic Versioning](https://semver.org): while they are 0.x, a release whose
change a caller must adapt to (the worker protocol, the C API, the GGUF layout, a voice file's form, a tool's
arguments) raises the minor version, and any other release the patch. The Vulkan build needs no particular driver version; on the first run the GPU driver compiles its
shaders, which takes seconds and is cached by the driver until it is updated. The Metal build compiles its
kernels on its first run as well (16 s for an Irodori-TTS worker on an Apple M5, 1.5 s on the runs after).

The Linux binaries (`linux-x64-vulkan` and `linux-x64-cpu`) run on glibc 2.34 or later: Ubuntu 22.04, Debian
12, Fedora 35, RHEL 9 and the releases after them (docs/adr/0010). They need an x86-64 CPU with AVX2, FMA and
F16C (Intel Haswell and AMD Excavator or later) and no shared library beyond glibc, since the C++ runtime is
linked in. The Vulkan build also needs the Vulkan loader, `libvulkan.so.1` (`libvulkan1` on Debian and
Ubuntu, `vulkan-loader` on Fedora), without which it does not start, and the GPU's Vulkan driver (Mesa's for
AMD and Intel, NVIDIA's own for NVIDIA). The CPU build needs neither, for a machine without a GPU or its
driver, and runs on the CPU without `--device`. A Vulkan driver that runs on the CPU, such as Mesa's
llvmpipe, is not offered as a device.

## Build

```sh
git clone --recurse-submodules https://github.com/nyosegawa/speech.cpp.git
cd speech.cpp
cmake -B build                  # Metal on macOS, the CPU elsewhere
cmake --build build --config Release -j
```

For Vulkan or CUDA, configure with `-DGGML_VULKAN=ON` (the Vulkan SDK is needed to build) or
`-DGGML_CUDA=ON` instead.

On Linux the build needs CMake 3.20 or later and GCC or Clang with C++17; it is checked with Ubuntu 22.04's
CMake 3.22 and GCC 11.4. For Vulkan, the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#linux) brings the
loader, the headers, SPIRV-Headers and `glslc` in one archive that needs no installation; CI uses 1.4.357.0:

```sh
curl -LO https://sdk.lunarg.com/sdk/download/1.4.357.0/linux/vulkansdk-linux-x86_64-1.4.357.0.tar.xz
tar -xJf vulkansdk-linux-x86_64-1.4.357.0.tar.xz
source 1.4.357.0/setup-env.sh
cmake -B build -DGGML_VULKAN=ON
cmake --build build -j
```

A build from source is tuned to the CPU it is built on and uses OpenMP, so it needs `libgomp.so.1`; the
releases are built for any CPU with AVX2 (`-DGGML_NATIVE=OFF`) and with ggml's own threads
(`-DGGML_OPENMP=OFF`), as `.github/workflows/build.yml` shows. On a machine without a GPU, the CPU build
(`cmake -B build`) runs the tools on the CPU without `--device`.

The build makes the library twice from the same sources: statically into every executable, so that each
tool is one file with ggml inside, and as the shared library `libspeech` (`libspeech.3.dylib`, `libspeech.so.3`,
`speech.dll`), which exports the functions of `speech.h` and nothing else. It also builds the checks, which
need the weights and the reference dumps to run.

On Metal, the library turns off Metal 4's tensor API for its own ggml: it sets `GGML_METAL_TENSOR_DISABLE` while
ggml first lists its devices, when ggml reads it, and then restores it. ggml uses that API on the M5 and later chips,
and its matrix kernel in ggml v0.25.3 writes past its output when the output has 64 modulo 128 columns; on an M5 this
turned a codec window of 64 frames into noise ([#13](https://github.com/nyosegawa/speech.cpp/issues/13)). Without it Metal comes closer to the CPU, and on
the M5 Irodori-TTS takes a fifth to a third longer to its first audio while Qwen3-TTS keeps its speed.

## Command line

`speech-tts` speaks text with any model speech.cpp runs, through the C API, into a WAVE file (16-bit mono PCM
at the model's rate) or to stdout. It loads the model once and speaks the text given, or, without one, every
line of stdin as a request of its own, blank lines skipped, all into the one WAVE in the order of the lines.
The first line takes the seed and each later line the next one, as the worker's requests do, so a line gives
the same audio as the same request to the worker.

```sh
# One sentence to a file, with a Qwen3-TTS speaker
speech-tts qwen3-tts-0.6b-customvoice-q8_0.gguf --voice-name ono_anna --seed 42 -o out.wav "明日の東京は晴れです。"

# A text file, one sentence per line, into one WAVE file, in the voice of a reference recording
speech-tts irodori-tts-v4.1-small-mf-f16.gguf --voice bright=bright-young-woman-10s.voice.gguf -o story.wav < story.txt

# Straight into a player, which starts as the first audio arrives
echo "こんにちは。" | speech-tts irodori-tts-v4.1-small-mf-f16.gguf \
    --voice bright=bright-young-woman-10s.voice.gguf -o - | ffplay -nodisp -autoexit -

# An Irodori-TTS voice file from a reference recording (see Irodori-TTS voices below)
speech-tts make-voice irodori-tts-v4.1-small-mf-f16.gguf bright-young-woman-10s.wav bright-young-woman-10s.voice.gguf \
    --device cpu
```

A model is one GGUF file, its codec included (see GGUF files below).

The options set the C API's load parameters and request options of the same meaning (Options, below):

| Option | For | Meaning |
|---|---|---|
| `-o FILE` | both | the WAVE file to write, or `-` for stdout. Needed |
| `--device NAME`, `gpu`, `cpu` | both | the device as `speech-tts --devices` lists it, the first GPU or the CPU; the first GPU, or the CPU on a machine without one, unless given |
| `--seed n` | both | the seed of the first text; each later line takes the next one. Without it the seed is random |
| `--voice-name NAME` | both | the voice to speak with: a Qwen3-TTS speaker or the name of a `--voice`. Needed unless the model has a single voice; an error lists the model's voices |
| `--language TAG` | both | a BCP 47 tag of one of the model's languages, or `auto` (the default) |
| `--voice NAME=FILE` | Irodori-TTS | a voice added to the model, repeated for more: a reference WAVE file or a voice file. At least one is needed |
| `--steps n` | Irodori-TTS | the sampler's steps of every text: 4 for v4.1-Small-MF and 40 for v4.1-Small unless given |
| `--speed x` | Irodori-TTS | the speaking rate, 0.25 to 4 (1) |
| `--seconds s` | Irodori-TTS | the length of each text's speech, 0.5 to 30 s after `--speed`, instead of the predicted one |
| `--duration-scale x` | Irodori-TTS | the factor of the predicted length (1); not together with `--seconds` |
| `-v` | both | also report the release, the model's voices, languages and steps, and each text's seed |

A text that begins with `-` follows `--`. A model refuses what it cannot do, as in the C API: Qwen3-TTS
answers `--seconds` and any `--speed` or `--duration-scale` but 1 with an error. A text whose speech reaches the
longest the model makes, which Qwen3-TTS's file gives (655 s), is stopped there and reported on stderr.

`speech-tts` reports on stderr where the time went, which makes it a tool for measuring as well: the load
(which includes a warm-up that compiles the GPU's kernels), and for each text its seconds of audio, the time to its first
audio and to its end, and the real-time factor, with the sums when stdin gave several lines. Nothing else is
written unless `-v` asks for it. With `-o -`, stdout carries the WAVE and nothing else, as the worker's
stdout carries its protocol alone. The audio is written as it is made, so a player reading the pipe starts
before the rest is made. Into a regular file, `> out.wav` included, the RIFF and data sizes are set once the
audio is complete. Anywhere else, a pipe or a file appended to with `>>`, the header cannot be written again
in place, so the sizes stay `0xFFFFFFFF`, as ffmpeg writes them to a pipe, and players and ffmpeg read to the
end of the stream.

A run that fails exits with 1 (2 for a command line it cannot run) and a message that names what failed,
and removes the WAVE file it was writing, so a file it leaves is always complete. Text on stdin is UTF-8;
a byte order mark and CRLF line endings are accepted.

`speech-asr` recognizes the speech in WAVE files with a recognition model, through the C API, and writes the text of
each file to stdout as one line, in the order the files are given. A file is 16-, 24- or 32-bit PCM or 32-bit float
at any rate, its channels averaged, and the library resamples it to the model's rate (16 kHz for FastConformer; see
Audio at another rate below); audio in another format is converted first (`ffmpeg -i in.mp3 out.wav`).

```sh
speech-asr parakeet-tdt_ctc-0.6b-ja-f16.gguf meeting.wav
speech-asr parakeet-tdt_ctc-0.6b-ja-f16.gguf --device cpu one.wav two.wav > texts.txt
```

| Option | Meaning |
|---|---|
| `--device NAME`, `gpu`, `cpu` | as for `speech-tts` |
| `--language TAG` | a BCP 47 tag of one of the model's languages, or `auto` (the default); another language is an error |
| `-v` | also report the release, the sample rate and the model's languages |

It reports on stderr the load and, for each file, its seconds of audio, the time to its text and the real-time
factor. A run that fails exits with 1 (2 for a command line it cannot run) and a message that names the file and
what failed; the lines of the files before it are already on stdout.

## Layout

- `include/speech.h` is the C API, the one way into the library.
- `src/` is the library: `speech.cpp`, `info.cpp` and `request.cpp` implement the C API over one engine per family
  (`src/<family>-engine.cpp`, which declares the options the family takes), `src/families/<family>/` runs one
  architecture of model, whichever weights it is given, and `src/common/` holds what the families share.
- `tools/` holds the programs built on the library: the worker (`tools/worker/`), the command-line tools
  `speech-tts` and `speech-asr` (`tools/cli/`) and the HTTP server `speech-server` (`tools/server/`), which
  serves a model over HTTP with OpenAI's audio API.
- `vendor/cpp-httplib/` holds cpp-httplib's header and license, which the server alone uses.
- `checks/` holds a check per ported stage that compares it with the official implementation, and
  `speech-api-check`, which runs the C API through the shared library with a synthesis model and, with
  `transcribe`, with a recognition model and the dumps of `reference/fastconformer/`.
- `reference/<model>/` pins the official implementation in a uv environment and the checkpoints by revision,
  converts the weights to one GGUF file per model (GGUF files, below) and dumps the tensors the checks compare with;
  `reference/resample/` dumps torchaudio's resampling, which `resample-check` compares the library's with.

## Audio at another rate

The library resamples the audio it is given, a recording to recognize and the reference recording of an Irodori-TTS
voice, to the model's rate (`speech_model_info_sample_rate()`), and never the audio it makes. It uses the method of
torchaudio's `functional.resample()` with the parameters torchaudio's documentation gives for librosa's `kaiser_best`:
a rational polyphase windowed sinc with 64 zero crossings on each side, cut off at 0.9475937167399596 of the lower
rate's Nyquist frequency, under a Kaiser window of beta 14.769656459379492, computed in double precision. From 48 to
16 kHz it passes everything up to 0.9 of the lower Nyquist frequency within 0.022 dB and keeps everything from 1.05
of it at -146 dB or below; torchaudio's defaults would lose 2.4 dB at 0.9 and fold a tone at 1.1 back into the band
at -14 dB. Its output is torchaudio's with these parameters, with the same length and to double precision: 301 dB SNR
or more on chirps and noise at eight pairs of rates. torchaudio rounds the window's beta and the output's length
through float32, and the library does the same. Audio at the model's rate passes unchanged, so the checks against the
dumps are unaffected. Two rates whose ratio in lowest terms has a term above 4096 (44101 and 16000 Hz) are refused
with an error.

```sh
cd reference/resample
uv run python dump.py out
cd ../..
build/resample-check reference/resample/out
```

The official implementations resample each in their own way, NeMo's `transcribe()` with librosa's soxr and the
Irodori-TTS runtime with torchaudio's defaults, so audio at another rate gives a text or a voice slightly different
from theirs.

## The C API

`include/speech.h` declares everything; this is the shape of a program that speaks one sentence:

```c
#include "speech.h"

static int on_audio(const float * samples, size_t n, void * user_data) {
    /* n mono float samples at the model's sample rate, at least one. Returning nonzero stops the request. */
    return 0;
}

static int check(speech_status status) {
    if (status < 0) {
        const char * option = speech_last_error_option();
        fprintf(stderr, "%s (%s): %s\n", speech_status_name(status), option ? option : "-", speech_last_error());
    }
    return status >= 0;
}

speech_model * model = NULL;
speech_request * request = NULL;
if (check(speech_model_load("irodori-tts-v4.1-small-mf-f16.gguf", NULL, &model)) &&
    check(speech_voice_add(model, "bright", "bright-young-woman-10s.voice.gguf")) &&
    check(speech_request_new(model, &request)) &&
    check(speech_request_set_text(request, "明日の東京は晴れです。")) &&
    check(speech_request_set_string(request, SPEECH_OPT_VOICE, "bright")) &&
    check(speech_request_set_int(request, SPEECH_OPT_SEED, 42)) &&
    check(speech_synthesize(request, on_audio, NULL))) {
    const speech_result * result = speech_request_result(request);
    printf("%s, seed %lld, %llu samples\n", speech_stop_name(speech_result_stop(result)),
           (long long) speech_result_seed(result), (unsigned long long) speech_result_samples(result));
}
speech_request_free(request);
speech_model_free(model);
```

and of one that recognizes the speech in mono samples at any rate, with the times of its segments:

```c
speech_model * model = NULL;
speech_request * request = NULL;
if (check(speech_model_load("parakeet-tdt_ctc-0.6b-ja-f16.gguf", NULL, &model)) &&
    check(speech_request_new(model, &request)) &&
    check(speech_request_set_audio(request, samples, n_samples, 48000)) &&
    check(speech_request_set_bool(request, SPEECH_OPT_TIMESTAMPS, 1)) &&
    check(speech_transcribe(request))) {
    const speech_result * result = speech_request_result(request);
    for (size_t i = 0; i < speech_result_segment_count(result); i++) {
        double start, end;
        const char * text;
        speech_result_segment(result, i, &start, &end, &text);
        printf("%.2f %.2f %s\n", start, end, text);
    }
}
speech_request_free(request);
speech_model_free(model);
```

- **Loading.** `speech_model_load()` loads a model from its one GGUF file, which holds its codec, and chooses the
  family from the file's `general.architecture`; a file of a layout this library does not read is refused with a
  message that names the release that reads it (GGUF files, below). Its parameters, from `speech_load_params_new()`
  or NULL for the defaults, are the same for every family: the device (`auto`, the default, for the first GPU or the
  CPU when there is none; `gpu`; `cpu`; or a device's name, compared without case), the CPU's threads (a preference
  that applies to whatever runs on the CPU; by default the machine's performance cores, or its physical cores where the
  system does not tell them apart), and a warm-up that runs a short request while the model loads so that a GPU
  compiles its kernels before the first request (off by default; the worker, the server and the command-line tools
  turn it on). A device asked for by `gpu` or by name that is not there or does not start is an error, and no other
  device takes its place.
- **Model information.** `speech_model_info_open()` reads what a model file says of its model from its metadata,
  without its weights and without a device: its name, architecture and layout, task and sample rate, languages,
  voices, whether it takes voice files and the codec they must carry, the longest text, the options it takes with
  their types, defaults, ranges and choices (Options, below), its sizes, and every metadata entry as JSON.
  `speech_model_get_info()` gives the same of a loaded model as it is at the call, with the device it runs on, the
  threads in effect and the voices added since. `speech_model_info_text_tokens()` counts a text's tokens as a
  synthesis counts them against the longest text, so that a caller can split a long text before it sends it, and
  `speech_model_info_json()` writes the whole as one JSON object (Model information as JSON, below).
- **Voices.** Irodori-TTS has no voices of its own: `speech_voice_add()` adds one to a loaded model from a voice
  file or a reference WAVE file at any rate, under a name compared with case, and `speech_voice_make()` writes a voice
  file from a reference, reading only the codec's encoder from the model file (Irodori-TTS voices, below). A model
  that takes no voice files answers both with `SPEECH_ERROR_UNSUPPORTED`.
- **Requests.** A request is made for one model with `speech_request_new()`. It takes a text
  (`speech_request_set_text()`) or audio (`speech_request_set_audio()`, mono at any rate, resampled to the model's;
  Audio at another rate, above), and options through the setter of each option's type. Each value is checked against
  what the model declares as it is set, and the request as a whole when it runs, before any work; a refused value
  leaves the request as it was. `speech_synthesize()` passes the audio to its callback as it is made and
  `speech_transcribe()` recognizes the whole audio at once; a request runs once. `speech_request_set_progress()` gives
  a callback that hears how far a request has come while it passes no audio: Irodori-TTS's sampler steps and a
  recognition's stages and decoding. `speech_request_cancel()` stops one request, from any thread, at the next audio,
  step or stage, and a request cancelled before it runs returns at once; either returns `SPEECH_CANCELLED`.
- **Results.** `speech_request_result()` gives what a request that returned `SPEECH_OK` or `SPEECH_CANCELLED` did:
  why it stopped (`complete`, `max_seconds`, `model_limit` or `cancelled`), the seed of a synthesis (the request's,
  or one the library drew from 0 to 2^53 - 1, with which the same request repeats its audio on the same device), the
  samples it passed, and the text of a recognition with, when the request set `timestamps`, its segments and tokens
  with their times in seconds (FastConformer, below).
- **Errors.** A function that can fail returns a `speech_status`, a negative one for an error, whose category says
  what kind of failure it is; `speech_status_name()` gives its name, `speech_last_error()` the message and
  `speech_last_error_option()` the input it concerns (an option's name, `text`, `audio`, `device`, `threads`, `name`,
  `path`, or for `speech_voice_make()` the parameter's name), on the same thread. No C++ exception crosses the API,
  and the library checks what it is given before ggml sees it.

| Status | Name | Meaning |
|---|---|---|
| `SPEECH_OK` | `ok` | the call did what it was asked |
| `SPEECH_CANCELLED` | `cancelled` | `speech_request_cancel()` or a callback stopped the request |
| `SPEECH_ERROR_INVALID_ARGUMENT` | `invalid_argument` | the caller's mistake: a NULL pointer, an empty text, a value of the wrong type, a required option left out, a request run twice |
| `SPEECH_ERROR_UNSUPPORTED` | `unsupported` | the model cannot do it: an option it does not take, at a value other than the neutral one, or the other task |
| `SPEECH_ERROR_OUT_OF_RANGE` | `out_of_range` | a value the model does not take: outside its range or choices, or a text too long |
| `SPEECH_ERROR_MODEL_FILE` | `model_file` | a model or voice file that cannot be used |
| `SPEECH_ERROR_DEVICE` | `device` | a device that is not there, does not start, or fails while it computes |
| `SPEECH_ERROR_OUT_OF_MEMORY` | `out_of_memory` | the host's or the device's memory ran out |
| `SPEECH_ERROR_IO` | `io` | a file that cannot be opened, read or written |
| `SPEECH_ERROR_INTERNAL` | `internal` | a defect of the library |

- **Devices.** `speech_device_count()`, `speech_device_name()`, `speech_device_description()`,
  `speech_device_get_kind()` and `speech_device_memory()` list the CPU and the GPUs a model can run on, in ggml's
  order; an accelerator that ggml runs beside the CPU, such as BLAS, is not listed. On macOS the library sets
  `GGML_METAL_TENSOR_DISABLE` for its first listing of the devices alone, when ggml reads it, and then restores it,
  so a host's own ggml keeps Metal's tensor API (Build, above).
- **Logs.** `speech_log_set()` sends the library's messages and ggml's to a callback; until it is called, warnings
  and errors go to stderr and the rest is dropped.
- **Ownership.** Every object the library returns is freed only through the function named for it, and every string
  it returns lives as long as the object it was read from: an information's until `speech_model_info_free()`, a
  result's until `speech_request_free()`, a device's and a name's as long as the process, and the error message
  until the next call on the thread. Nothing the caller passes is kept after the call.
- **Threads.** A model serves one request at a time; requests that several threads run on the same model wait for
  each other. A request is used by one thread at a time, but `speech_request_cancel()` may be called from any thread.
  Information never changes once made and may be read from any thread. Separate models are independent.
- **Versions.** `speech_version()` gives the release the library was built from (`"0.7.0"`).
  `SPEECH_API_VERSION_MAJOR` and `SPEECH_API_VERSION_MINOR`, and `speech_api_version_major()` and
  `speech_api_version_minor()` for a caller that loads the library at run time, give the API's version, 3.0: the
  major rises when a declaration changes in a way an existing caller notices, and the minor when a function, an
  option or an enum value is added. A program built against major version M and minor version m runs against a
  library of the same major version and a minor version of m or more. The shared library's SOVERSION is the major
  version (`libspeech.3.dylib`, `libspeech.so.3`).

Link `libspeech` (on Windows, define `SPEECH_SHARED` and link `speech.lib`), or, within this CMake project,
the target `speech` (shared) or `speech-static`.

### Options

What a request may ask of a model is one vocabulary of options (`speech_option`), each with a fixed name in
snake_case (`speech_option_name()`, `speech_option_from_name()`) and one type. Each family declares the options it
takes in one table in its engine (`src/<family>-engine.cpp`); ranges and defaults that the model defines come from its
GGUF file (the keys are named in parentheses). The setters, the model information and its JSON read that table and
nothing else.

| Option | Type | Neutral | Qwen3-TTS | Irodori-TTS | FastConformer |
|---|---|---|---|---|---|
| `voice` | string | none | required; one of the speakers (`speech.voices`); steers | required; one of the voices added since loading; steers | not taken |
| `language` | string | `auto` | default `auto`; one of `speech.languages`; steers | default `auto`; one of `speech.languages` (`ja`); checked | default `auto`; one of `speech.languages`; checked |
| `seed` | int | none | 0 to 2^53 - 1; drawn when not set | 0 to 2^53 - 1; drawn when not set | not taken |
| `speed` | float | 1 | not taken | 0.25 to 4 (`irodori-tts.length.min_speed`, `max_speed`), default 1 | not taken |
| `seconds` | float | none | not taken | 0.5 to 30 (`irodori-tts.length.min_seconds`, `max_seconds`), no default | not taken |
| `duration_scale` | float | 1 | not taken | above 0, default 1 | not taken |
| `steps` | int | none | not taken | 1 to 2147483647, default `irodori-tts.sampler.default_steps` (4 for MeanFlow, 40 for RF) | not taken |
| `max_seconds` | float | none | above 0 to the model's limit, `qwen3-tts.generation.max_frames` frames (8192 × 0.08 s = 655.36 s); no default | not taken | not taken |
| `timestamps` | bool | false | not taken | not taken | default false |

A value at an option's neutral value is accepted by every model; any other value of an option a model does not take
is `unsupported`, and an option marked "none" has no neutral value. A string option's value must be one of its
choices (`voice` compared with case; `language` also takes `auto` and a region or script of a choice, compared
without case); a number outside the range is `out_of_range`. "Checked" means the language is compared with the
model's languages and then not used: Irodori-TTS and the Japanese recognizers have one language, and
parakeet-tdt-0.6b-v3 finds the language of the audio itself.

What only the whole request shows is refused when the request runs, before any work, naming the option:

- `voice` left out where it is required: `invalid_argument`.
- Irodori-TTS, `seconds` together with a `duration_scale` other than 1: `invalid_argument`, option `seconds`.
- Irodori-TTS, `seconds / speed` outside 0.5 to 30 s: `out_of_range`, option `seconds`.
- Irodori-TTS, when `duration_scale` or `speed` is not 1, the predicted length times `duration_scale / speed` outside
  0.5 to 30 s: `out_of_range`, option `duration_scale`, or `speed` when the scale is 1. At 1 and 1 the predicted
  length is kept within the bounds, as the official runtime keeps it.
- A text longer than `speech_model_info_max_text_tokens()`: `out_of_range`, option `text`. Irodori-TTS counts the
  tokens of the normalized text and takes at most `irodori-tts.text.max_tokens` (256). Qwen3-TTS takes what leaves its
  talker room for the longest speech: `qwen3-tts.talker.max_position_embeddings` - `qwen3-tts.generation.max_frames` -
  11, the rows its prompt adds to the text (32768 - 8192 - 11 = 24565 for both sizes).
- Audio too short for the model to recognize, two of its mel frames (20 ms for FastConformer): `out_of_range`, option
  `audio`.

### Model information as JSON

`speech_model_info_json()` writes one object, its members in this order; a member that does not apply is left out
rather than null:

```json
{
  "name": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
  "architecture": "qwen3-tts",
  "layout": 1,
  "task": "synthesis",
  "sample_rate": 24000,
  "incremental": true,
  "languages": ["de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "zh"],
  "voices": [
    {"name": "aiden", "language": "en", "gender": "male", "description": "Sunny American male voice."},
    {"name": "ono_anna", "language": "ja", "gender": "female", "description": "Playful Japanese female voice."}
  ],
  "voice_files": false,
  "max_text_tokens": 24565,
  "options": [
    {"name": "voice", "type": "string", "required": true, "steers": true, "choices": ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]},
    {"name": "language", "type": "string", "required": false, "steers": true, "default": "auto", "choices": ["de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "zh"]},
    {"name": "seed", "type": "int", "required": false, "steers": true, "minimum": 0, "maximum": 9007199254740991},
    {"name": "max_seconds", "type": "float", "required": false, "steers": true, "exclusive_minimum": 0, "maximum": 655.36}
  ],
  "file_bytes": 1213534080,
  "weight_bytes": 1208175044,
  "device": "MTL0",
  "threads": 0
}
```

- `incremental` is true for a model that passes audio while it is still generating the rest (Qwen3-TTS), so that its
  first audio does not wait on the length of the text, and false for one that makes a request's whole speech before
  it decodes it (Irodori-TTS).
- `voices`: each voice's `language`, `gender` and `description` are "" where the file does not say; a voice added
  from a voice file or a recording says none. Left out for a recognition model, as are `incremental` and
  `max_text_tokens`.
- `voice_files` is true for a model that takes voices made from recordings; `voice_codec` then follows it, the hash a
  voice file must carry.
- An option has `type` (`string`, `int`, `float`, `bool`), `required`, `steers`, and where they apply `default`,
  `minimum` or `exclusive_minimum`, `maximum`, and `choices`. A bound that does not exist is left out.
- `device` and `threads` are there for a loaded model alone. `threads` is the number of CPU threads in effect: the
  number the load parameters set, or the library's default, for a model on the CPU, and 0 for a model on a GPU.
  Setting threads is never an error, since with the device `auto` a caller cannot know where the model will run.
- The example is `qwen3-tts-0.6b-customvoice-q8_0.gguf` loaded on Metal, with two of its nine voices shown.

## The worker

`speech-worker` is a process that another program starts to speak texts or to recognize speech, as ASIST does, and
a program on the C API like any other. It speaks [JSON Lines](https://jsonlines.org): it reads one JSON message
per line on stdin and answers with one JSON object per line on stdout, and runs the family that
`general.architecture` of the model GGUF names. Its `ready` message says the model's task, `"synthesis"` or
`"recognition"`, and the messages it takes follow from the task (below). Nothing else is written to stdout: every log goes to stderr, and so does anything
ggml, a system library or the GPU driver prints to stdout. A caller treats a line on stdout that is not a JSON
object as a defect of the worker and fails, rather than skipping it.

```sh
speech-worker qwen3-tts-0.6b-customvoice-q8_0.gguf
speech-worker irodori-tts-v4.1-small-mf-f16.gguf \
    --voice bright=bright-young-woman-10s.voice.gguf --voice calm=calm-reference.wav
speech-worker parakeet-tdt_ctc-0.6b-ja-f16.gguf
```

| Option | For | Meaning |
|---|---|---|
| `--device NAME`, `gpu`, `cpu` | all | the device as `--devices` names it (`MTL0`, `Vulkan1`), the first GPU or the CPU; the first GPU, or the CPU on a machine without one, unless given |
| `--seed n` | synthesis | the seed of the first request, 0 to 2^53 - 1; each later request takes the next one. Without it the seed is random |
| `--voice NAME=FILE` | Irodori-TTS | a voice added to the model, repeated for more: a reference WAVE file or a voice file (below). At least one is needed |
| `--steps n` | Irodori-TTS | the sampler's steps of every request: 4 for v4.1-Small-MF and 40 for v4.1-Small unless given |

The worker loads the model with a warm-up, so that a GPU has compiled its kernels before the first request, and
adds the voices before it is ready. `speech-worker --devices` prints the devices it can run on, the CPU and the GPUs
without the accelerators ggml runs beside the CPU, with their memory, and exits.

The messages of a synthesis worker, one JSON object per line:

| Direction | Message |
|---|---|
| out | `{"type":"ready","task":"synthesis","model":"Irodori-TTS-v4.1-Small-MF","architecture":"irodori-tts","sampleRate":48000,"streaming":"sentence","voices":["bright","calm"],"languages":["ja"],"languageSelectable":false,"steps":4,"backend":"MTL0","version":"0.4.0"}`, `version` being the release of speech.cpp |
| in | `{"id":"1","text":"明日の東京は晴れです。","voice":"bright"}`, with `"language"`, `"speed"`, `"seconds"` and `"durationScale"` optional |
| out | `{"type":"chunk","id":"1","seq":0,"pcm":"<base64 of 16-bit little-endian mono PCM at sampleRate>"}`, one or more |
| out | `{"type":"end","id":"1","samples":278400}` |
| out | `{"type":"error","id":"1","error":"..."}` when a request cannot be spoken or a line cannot be read; without `"id"` when the line has none the worker could read |
| out | `{"type":"fatal","error":"..."}` when the worker cannot start |
| in | `{"type":"cancel","id":"1"}`: the request stops between two chunks (Irodori-TTS also between two of its sampler's steps, before the first chunk) and sends no `end`; a request cancelled before it starts is dropped |

Requests are served one at a time in arrival order.

Every line on stdin gets an answer, a recognition request's chunks through the answer to their request. A line
the worker cannot read as a message of its task (not one JSON object, a member that is not a string or a number
such as `null` or a nested object, no `id`, or a `type` the task does not take: other than `cancel` for
synthesis) is answered at once with an `error` that names the problem. It carries the
line's `id` when one could be read, and has no `id` otherwise. Such an error is a defect of the caller, which
should fail rather than wait for an answer to the line it meant to send.

`speed`, `seconds` and `durationScale` are JSON numbers, the C API's request options `speed`, `seconds` and
`duration_scale` (Options, above). `speed` is the speaking rate against the model's own (1); `seconds` fixes the
length of the speech, and 0 or no `seconds` leaves it to the model; `durationScale` scales the length the model
predicts (1). A model that cannot follow one of them answers the request with an `error` instead of ignoring it, and
so does a value that is not a number or lies out of its range. Irodori-TTS takes all three (see Length and speed
below). Qwen3-TTS refuses any `speed` or `durationScale` but 1 and any `seconds`: its official implementation has no
control of either (docs/adr/0007). A Qwen3-TTS request whose speech reaches the longest the model makes, 655 s, is
stopped there and ends as one that came to its end does.

`languages` lists the languages as BCP 47 tags. A request's `language`, when given, must be one of them or
a region or script of one (`ja`, `ja-JP`, `zh-Hant`), or `auto`; `auto` or no `language` leaves the choice
to the model. Any other language is an error.

- **Qwen3-TTS** streams frame by frame (`"streaming":"frame"`, 24 kHz). Its voices are the model's speakers,
  and it speaks `de`, `en`, `es`, `fr`, `it`, `ja`, `ko`, `pt`, `ru` and `zh` (`"languageSelectable":true`):
  the language goes into its prompt. Two speakers speak a Chinese dialect, `dylan` (Beijing) and `eric`
  (Sichuan), when the language is `zh` or left to the model, as in the official implementation.
- **Irodori-TTS** makes a sentence at once and streams it as the codec decodes it (`"streaming":"sentence"`,
  48 kHz), so a request should be one sentence; a text longer than the model's 256 tokens is refused. Its
  voices are those given with `--voice`. It speaks `ja` and is not told a language
  (`"languageSelectable":false`).

A recognition worker takes the audio of a request in the chunks a synthesis worker sends, and the request itself
in the request's `end`:

| Direction | Message |
|---|---|
| out | `{"type":"ready","task":"recognition","model":"parakeet-tdt_ctc-0.6b-ja","architecture":"fastconformer","sampleRate":16000,"languages":["ja"],"languageSelectable":false,"backend":"MTL0","version":"0.5.0"}`, without `streaming`, `voices` or `steps` |
| in | `{"type":"chunk","id":"1","seq":0,"pcm":"<base64 of 16-bit little-endian mono PCM>"}`, zero or more, `seq` counting from 0; the chunks of several requests may interleave |
| in | `{"type":"end","id":"1","sampleRate":16000}`, with `"language"` optional: the request, recognized once it is its turn |
| out | `{"type":"text","id":"1","text":"群島や湖では必ずしもヨットは必要ありません。"}` |
| out | `{"type":"error","id":"1","error":"..."}` when the request cannot be recognized or one of its lines is refused |
| in | `{"type":"cancel","id":"1"}`: the request sends no `text`, whether it is still taking chunks, waits or is being recognized |

A request is answered once, by its `text` or by one `error`. `sampleRate` is the rate of the request's audio, any
rate; the library resamples it to the model's `sampleRate`, 16000 for FastConformer. A chunk whose `seq` is not the
next one, whose `pcm` is not base64 or holds an odd number of bytes, and an `end` without a whole-number `sampleRate`
are answered at once with an `error`, and the request's other lines are then dropped. Requests are recognized one at a
time in the order of their ends. A request to speak sent to a recognition worker is an error, and so is a chunk or an
end sent to a synthesis worker. FastConformer recognizes a request's audio at once. The parakeet models' encoders
attend over the whole of it, so a request to them should be one utterance: on an Apple M5, the 25.5 s FLEURS utterance
takes 0.22 s on Metal and its memory grows with the square of the length. reazonspeech-nemo-v2 attends locally and
takes a recording of minutes in one request. A cancel takes effect before the encoder starts, once it has run, or
between two steps of the decoding.

### Irodori-TTS voices

Irodori-TTS has no voices of its own; it speaks in the voice of a reference. A voice is either:

- a reference WAVE file: at most 120 s, 16-, 24- or 32-bit PCM or 32-bit float at any rate, the channels
  averaged and resampled to 48 kHz. `speech_voice_add()`, and so the worker when it starts, normalizes its loudness
  and encodes it with the codec, as the official runtime does for every request.
- a voice file, which `speech-tts make-voice` or `speech_voice_make()` writes from a reference WAVE file, reading
  only the codec's encoder from the model file: the reference's codec latent in a GGUF file of its own (Voice files,
  below) that carries the hash of the codec's tensors. It works with every model file of the same codec, v4.1-Small-MF and v4.1-Small in any type, and a model
  of another codec refuses it. Voice files made before 0.7.0 have no layout and are refused; make them again from
  their WAVE files.

```sh
speech-tts make-voice irodori-tts-v4.1-small-mf-f16.gguf bright-young-woman-10s.wav bright-young-woman-10s.voice.gguf \
    --device cpu
```

For the 10.7 s reference bright-young-woman-10s.wav, the voice file is 35 KB against the WAVE file's 1 MB,
and loads in 0.016 s on an Apple M5 (Metal) and 0.025 s on an RTX 2080 (Vulkan), against 0.72 s and 0.43 s
to encode the WAVE file (5.05 s on the M5's CPU). Made on the CPU (`--device cpu`), its latent is the
official encoder's to 99 dB SNR; on Metal it is 40 dB and on Vulkan 33 dB (see Accuracy below). ASIST
carries voice files (docs/adr/0002).

## The server

`speech-server` serves one model over HTTP with OpenAI's audio API, so that a web app, a Python script or curl
can speak a text or recognize speech without starting the worker. It loads the model as the worker does, listens
once the model is ready, and logs to stderr.

```sh
speech-server qwen3-tts-0.6b-customvoice-q8_0.gguf
speech-server irodori-tts-v4.1-small-mf-f16.gguf \
    --voice bright=bright-young-woman-10s.voice.gguf --port 8080 --cors-origin http://localhost:5173
speech-server parakeet-tdt_ctc-0.6b-ja-f16.gguf
```

| Option | Meaning |
|---|---|
| `--host ADDRESS` | the address to listen on, 127.0.0.1 unless given; the server has no authentication and no TLS, so a server reachable from other machines belongs behind a proxy that adds them |
| `--port n` | the port, 8080 unless given |
| `--cors-origin ORIGIN` | an origin a web page may call the server from, such as `http://localhost:5173`, repeated for more, or `*` for any. Without it the server sends no CORS headers. Preflight requests are answered |
| `--device`, `--voice NAME=FILE`, `--steps` | as for the worker (above) |

The endpoints:

- `POST /v1/audio/speech` speaks a text with a synthesis model, as [OpenAI's create speech](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create) does.
- `POST /v1/audio/transcriptions` recognizes the speech in a WAV file with a recognition model, as
  [OpenAI's create transcription](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create)
  does (below).
- `GET /v1/models` lists the loaded model as OpenAI's model object, with speech.cpp's members added: `task`
  (`synthesis` or `recognition`), `architecture`, `sample_rate`, `streaming` (`frame` or `sentence`) and `voices`
  (synthesis), `languages`, `language_selectable`, `steps` (Irodori-TTS), `backend` and `version`.
  `GET /v1/models/<id>` gives it alone.

The endpoint of the other task answers a 404 whose message names the right one. The speech request is described
first; the transcription request follows it.
- `GET /health` answers `{"status":"ok"}`.

The request is a JSON object:

| Member | Meaning |
|---|---|
| `input` | the text, required |
| `voice` | one of the model's voices, required: a Qwen3-TTS speaker or a voice given with `--voice` |
| `model` | the loaded model's `id` from `/v1/models`, or left out. Any other model is a 404 (`model_not_found`) |
| `response_format` | `wav` (the default) or `pcm`. OpenAI's default is `mp3`, which speech.cpp does not encode; `mp3`, `opus`, `aac` and `flac` are refused |
| `stream_format` | `audio` (the default) or `sse`, which needs `pcm` |
| `speed` | the speaking rate, as in the worker: Irodori-TTS takes 0.25 to 4, Qwen3-TTS only 1 |
| `language` | speech.cpp's own: a BCP 47 tag of one of the model's languages, or `auto` (the default) |
| `seed` | speech.cpp's own: the seed, an integer from 0 to 2^53 - 1. Without it the seed is random |
| `seconds`, `duration_scale` | speech.cpp's own: Irodori-TTS's length (see Length and speed below) |

A member speech.cpp does not take, OpenAI's `instructions` among them, is refused rather than ignored; a
member set to `null` counts as left out.

The response:

- `wav` is the whole file, 16-bit mono at the model's rate, sent once the sentence is made.
- `pcm` is raw 16-bit little-endian mono at the model's rate, `Content-Type: audio/pcm`, streamed with chunked
  transfer as the model makes it: the first bytes leave with the worker's first chunk. On an Apple M5 under
  heavy load from other work, the same requests alternated between the server and the worker gave a median
  first byte of 0.106 s against the worker's 0.102 s for Qwen3-TTS 0.6B Q8_0, and 0.64 s against 0.60 s for
  Irodori-TTS v4.1-Small-MF F16, and the same audio, byte for byte, for the same seed.
- `stream_format: "sse"` sends the same PCM as server-sent events, each a `data:` line:
  `{"type":"speech.audio.delta","audio":"<base64 PCM>"}` for each chunk, then
  `{"type":"speech.audio.done","samples":134400}`. OpenAI's done event carries the usage in tokens, which
  speech.cpp does not count, so it gives the number of samples instead.
- Every audio response carries `X-Sample-Rate` and `X-Speech-Seed`, the seed it was made with: the same
  request with that seed gives the same audio on the same device.

Errors have OpenAI's shape, `{"error":{"message":...,"type":...,"param":...,"code":...}}`. A request the server
cannot read (malformed JSON, a missing or unknown member, an unsupported format) is a 400 that names the
member in `param`; a request the model cannot follow (an unknown voice, a speed on Qwen3-TTS, a length out of
range) is a 400 with the library's message; another model is a 404, and a failure during the synthesis a
500. Once a stream has begun, a failure ends a `pcm` stream without its last chunk, which the client reads as
a broken transfer, and an SSE stream with `{"type":"error","error":{...}}`.

The model speaks one request at a time, in the order they arrive; the others wait. A client that disconnects
while it waits is dropped, and one that disconnects while its audio streams stops its synthesis at the next
chunk (Qwen3-TTS) or codec window (Irodori-TTS), as a cancel of the worker does, so the next request starts
then.

```sh
# A WAV file.
curl http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"input": "明日の東京は晴れです。", "voice": "bright"}' -o out.wav

# PCM played as it arrives (48000 for Irodori-TTS, 24000 for Qwen3-TTS: the X-Sample-Rate header).
curl -sN http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"input": "明日の東京は晴れです。", "voice": "bright", "response_format": "pcm"}' |
    ffplay -nodisp -autoexit -f s16le -ar 48000 -ch_layout mono -i -
# On Linux with ALSA: ... | aplay -f S16_LE -r 48000 -c 1

# Server-sent events.
curl -N http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"input": "明日の東京は晴れです。", "voice": "bright", "response_format": "pcm", "stream_format": "sse"}'
```

From Python with [requests](https://requests.readthedocs.io), streaming PCM into a WAV file as it arrives
(hand each chunk to an audio output instead to play it):

```python
import requests
import wave

with requests.post("http://127.0.0.1:8080/v1/audio/speech",
                   json={"input": "明日の東京は晴れです。", "voice": "bright", "response_format": "pcm"},
                   stream=True) as r:
    r.raise_for_status()
    with wave.open("out.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(r.headers["X-Sample-Rate"]))
        for chunk in r.iter_content(chunk_size=None):
            w.writeframes(chunk)
```

A transcription request is a `multipart/form-data` form, as OpenAI's API reference defines
`CreateTranscriptionRequest` (github.com/openai/openai-openapi at commit 31af4fc, 2026-10-05):

| Member | Meaning |
|---|---|
| `file` | the audio, required: a WAV file, 16-, 24- or 32-bit PCM or 32-bit float at any rate, its channels averaged and resampled to the model's `sample_rate` (16000 for FastConformer). Any other file is refused with a 400 (`param` `file`) rather than guessed at; convert it first (`ffmpeg -i in.mp3 out.wav`) |
| `model` | the loaded model's `id`, or left out; any other model is a 404 (`model_not_found`) |
| `language` | a BCP 47 tag of one of the model's languages, or `auto` (the default) |
| `response_format` | `json` (the default), which answers `{"text":"..."}`, or `text`, which answers the text alone as `text/plain`. `srt`, `vtt`, `verbose_json` and `diarized_json` are refused: speech.cpp gives neither timestamps nor speakers |

OpenAI's other members (`prompt`, `temperature`, `timestamp_granularities[]`, `stream`, `include[]` and the rest)
are refused with a 400 rather than ignored, and so is a member given twice. OpenAI's json answer also carries the
usage in tokens or seconds, which speech.cpp does not count. Audio the model cannot take (no samples, a rate the
library cannot resample from, a language it does not recognize) is a 400 with the library's message. The upload may
be up to 25 MB, OpenAI's limit; the model recognizes the whole file at once, so a file should be one utterance (see
the worker above). The request waits its turn like a speech request, and a client that goes away cancels it.

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@utterance.wav -F response_format=text
```

## Qwen3-TTS

It decodes audio frame by frame with the same samples as decoding the whole utterance, so streaming does
not add artifacts at the frame boundaries. Implemented:

- the talker (a Qwen3 decoder) that predicts the first codebook of each frame,
- the code predictor that predicts the other 15 codebooks,
- the 12Hz codec decoder (RVQ dequantization, sliding-window transformer, ConvNeXt upsampling and the
  SnakeBeta decoder), with each stage's causal state carried from one call to the next,
- the Qwen2 byte-level BPE tokenizer and the CustomVoice prompt,
- sampling as in transformers' `generate()` (temperature, top-k, top-p, repetition penalty).

Voice cloning, VoiceDesign, the codec encoder and the speaker encoder are out of scope. The model has no
control of its speaking rate or its length, so a request with a speed other than 1 or with a length is
refused (docs/adr/0007). A request stops at its `max_seconds`, when it sets one, and at the model's limit, 8192
frames (655 s of speech), the checkpoint's `max_new_tokens`, as the official implementation stops; the result says
which (docs/adr/0016). The talker's key/value cache grows with the speech rather
than holding the longest from the start, so a short sentence does not take the memory of the longest. A text of
more than 24565 tokens, what the talker's 32768 positions leave beside the longest speech and its prompt, is refused.

### Models

A synthesis needs one file per model, the codec inside it:
[sakasegawa/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF)
holds `qwen3-tts-0.6b-customvoice-q8_0.gguf` and
[sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF)
`qwen3-tts-1.7b-customvoice-q8_0.gguf`. The repositories get these files with the release of 0.7.0; until then
they hold the talker and the separate codec of earlier releases, which this one refuses. To convert them yourself
from the official checkpoints, which `reference/qwen3-tts/pins.py` pins by revision:

```sh
cd reference/qwen3-tts
uv run python convert.py 0.6b ../../models --type q8_0    # qwen3-tts-0.6b-customvoice-q8_0.gguf, 1.2 GB
uv run python convert.py 1.7b ../../models --type q8_0    # qwen3-tts-1.7b-customvoice-q8_0.gguf, 2.3 GB
```

`--type` also takes `f16` and `f32`, whose files are 4.1 GB for 0.6B and 8.1 GB for 1.7B. The codec's large weights
are float16 in a Q8_0 or F16 file, as the released files have them, and float32 in an F32 file.

### Use

```sh
build/speech-tts <model.gguf> --voice-name ono_anna --language ja -o out.wav "明日の東京は晴れです。"
```

`speech-worker <model.gguf>` runs it behind the worker protocol (above).

### Accuracy

`reference/qwen3-tts/dump.py` runs the official implementation with greedy decoding and saves the tensors
of every stage; the check tools compare against them.

| Check | Result |
|---|---|
| Codec decoder, whole utterance, CPU, F32 (`codec-check`) | 114 dB SNR against the official decoder |
| Codec decoder, one frame at a time against whole, CPU | 133 dB SNR |
| Codec decoder on Metal | error at -63 dB of the voice |
| Talker and code predictor, F32, teacher forcing (`talker-check`) | argmax matches on every frame; greedy decode gives the same 54 frames |
| Tokenizer (`tokenizer-check`) | matches the model's `tokenizer.json` on 19 texts |

The tokenizer follows the pre-tokenizer of the `tokenizer.json` that ships with the model. The official
package loads it through transformers 4.57.3 with `fix_mistral_regex=True`, which swaps in Mistral's
pattern; the two differ on Latin words in mixed case, contractions and `/`.

### Speed

Q8_0 weights, Japanese sentences, after the shaders are compiled:

| Model | Device | First audio | Real-time factor | VRAM |
|---|---|---|---|---|
| 0.6B | Apple M5, Metal | 0.04 s | 0.38 | |
| 1.7B | Apple M5, Metal | 0.07 s | 0.48 | |
| 0.6B | RTX 2080, Vulkan | 0.07 s | 0.31 | 1.6 GB |
| 1.7B | RTX 2080, Vulkan | 0.08 s | 0.36 | 2.7 GB |

## Irodori-TTS

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) by Aratako: a DiT that makes the 32-dimensional latent of
a 48 kHz codec (Semantic-DACVAE-Japanese-32dim) for a whole sentence, its length set beforehand by a
duration predictor. Implemented, for v4.1-Small-MF (4 MeanFlow steps) and v4.1-Small (Euler steps with the
runtime's guidance, text 3.0 and speaker 5.0 while t ≥ 0.5):

- the official text normalization, with NFKC from Unicode 13.0 as the official runtime's Python has it,
  which keeps the 56 emoji the model reads as directions (🤭 a giggle, 😮‍💨 a sigh, 👂 a whisper and the
  rest of the runtime's `ALLOWED_ANNOTATION_EMOJIS`),
- the SentencePiece Unigram tokenizer with byte fallback, and ModernBERT-ja with its projector,
- the reference's loudness normalization and the codec encoder, in windows of 100 frames,
- the speaker encoder, the duration predictor with the runtime's `seconds` and `duration_scale`, the DiT and
  both samplers,
- the tail cut where the latent goes flat, and the codec decoder, a first window of 12 frames (0.48 s) and
  then 48 at a time, each window giving the samples of decoding the whole latent at once.

Not implemented: captions (VoiceDesign), speaker-inversion embeddings and SilentCipher's watermark. The noise comes
from speech.cpp's own generator, so a seed gives other audio than the same seed in the official runtime. A reference
at another rate than 48 kHz is resampled with the library's filter, not with the runtime's torchaudio defaults (Audio
at another rate, above).

### Models

A synthesis needs one file per model, the codec inside it:
[sakasegawa/Irodori-TTS-v4.1-Small-MF-GGUF](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-MF-GGUF)
holds `irodori-tts-v4.1-small-mf-f16.gguf` and
[sakasegawa/Irodori-TTS-v4.1-Small-GGUF](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-GGUF)
`irodori-tts-v4.1-small-f16.gguf`, and their cards list the SHA-256. The repositories get these files with the
release of 0.7.0; until then they hold the model and the separate codec of earlier releases, which this one refuses.
To convert them yourself from the official checkpoints and codec, which `reference/irodori-tts/pins.py` pins by
revision:

```sh
cd reference/irodori-tts
uv run python convert.py mf ../../models --type f16       # irodori-tts-v4.1-small-mf-f16.gguf, 1.9 GB
uv run python convert.py rf ../../models --type f16       # irodori-tts-v4.1-small-f16.gguf, 1.9 GB
```

`--type` also takes `f32` (3.4 GB) and `q8_0` (1.2 GB). The codec stays float32 in every type, as the released
files have it. Qwen3-ASR 1.7B transcribed the 20 sentences of the speed table below with 2.99% CER in F32 and in
F16, and 3.81% in Q8_0, which garbled one phrase.

### Use

```sh
build/speech-tts irodori-tts-v4.1-small-mf-f16.gguf --voice bright=bright-young-woman-10s.voice.gguf \
    -o out.wav "明日の東京は晴れです。" [--device NAME] [--seed n] [--steps n] [--seconds s | --duration-scale x] [--speed x]
```

### Length and speed

The duration predictor sets the length of a sentence before the DiT makes it. A request may change it as the
official runtime's request does, and the C API, the worker and `speech-tts` take the same three options:

- `seconds` fixes the length: the latent has the frames that hold `int(seconds / speed × 48000)` samples, and the
  audio is cut there. The duration predictor does not run. Both `seconds` and `seconds / speed` must lie within 0.5
  to 30 s; the runtime clamps a length outside them with a warning, speech.cpp refuses it.
- `duration_scale` (`durationScale` in the worker) multiplies the predicted frames before they are rounded. It must
  be above 0, and it cannot be given with `seconds`, which the runtime would let override it without a word.
- `speed`, from 0.25 to 4, divides both: the length is `seconds / speed`, or the prediction scaled by
  `duration_scale / speed`. This is what [Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server)
  does with the `speed` of OpenAI's speech API, in the same range, so a request gets the length it would get
  there.

The predicted length alone, at a scale and a speed of 1, is kept within 0.5 to 30 s, as the runtime keeps it. A scale
or a speed that takes it outside them is refused, naming the scale, or the speed when the scale is 1, since the
caller asked for a length the model does not make (docs/adr/0014).

The audio may end before the length where the latent goes flat, as in the runtime. The frames are the
official runtime's for every combination the dumps cover (`irodori-condition-check`, `irodori-synthesis-check`).

### Accuracy

`reference/irodori-tts/dump.py` runs the official implementation on the CPU in float32 with fixed noise and
saves every stage; the check tools compare each stage, given the dump's own inputs, with it. Apple M5:

| Check | CPU, F32 | Metal, F32 | Vulkan, F16 model and F32 codec |
|---|---|---|---|
| Normalization and tokenizer, 164 texts with the 56 direction emoji (`irodori-text-check`) | all equal | all equal | all equal on the 51 texts without them |
| Text condition (`irodori-text-check`) | 118 to 123 dB SNR | 65 to 123 dB | 65 to 74 dB |
| Reference latent (`irodori-codec-check`) | 99 dB | 40 dB | 33 dB |
| Speaker condition (`irodori-condition-check`) | 111 dB | 51 dB | 51 dB |
| Length, predicted, scaled, fixed and at a speed (`irodori-condition-check`) | the official frames on every dump | the same | the same on the dumps without options |
| DiT steps, MF and RF (`irodori-dit-check`) | 95 dB or more | 46 dB or more | 63 dB or more (MF) |
| Sampled latent, MF / RF 40 steps | 86 to 122 dB / 109 to 111 dB | 36 to 67 dB / 59 to 67 dB | 66 dB (MF, 27 frames) |
| Decoded audio (`irodori-codec-check`) | 119 dB | 68 dB | 68 dB |
| Decoding in windows against at once | equal | equal | 89 dB (encoder), equal (decoder) |
| Whole synthesis from the dump's noise (`irodori-synthesis-check`) | 75 to 110 dB, the same length | 22 to 61 dB, the same length | 58 dB (MF, 27 frames), the same length |

The Metal and Vulkan columns were measured on an Apple M5 and an RTX 2080 with driver 591.86.

Metal's matrix kernel rounds both its inputs to half precision (`kernel_mul_mm_f32_f32` keeps its tiles as
`half`), which is the gap between the CPU and Metal; MeanFlow's four large steps carry it into the latent,
so on Metal the audio is the same speech rather than the same waveform. Vulkan accumulates in float32 as
the port asks. The codec on a GPU is checked against the CPU as well: on Metal its error lies 68 dB below the
voice and on Vulkan 68 dB, and the quietest tenth of the 20 ms frames stays as quiet as on the CPU (-78 and
-76 dBFS against -76). audio.cpp v0.8.2's Irodori-TTS adds a distorted copy of the voice 14 dB below it on
Metal and raises the quiet parts to -60 dBFS.

A reference at another rate, made from the dumps' 48 kHz reference with torchaudio's `kaiser_best`, gives on the CPU
the latent the official codec makes of the same audio resampled to 48 kHz by torchaudio with `kaiser_best`, to 99 dB.
That latent lies 30.6 dB from the 48 kHz reference's at 44.1 kHz and 10.7 dB at 24 kHz, which has nothing above
11.4 kHz; the official runtime, which resamples with torchaudio's defaults, is 29.5 and 12.5 dB from its own.

### Speed

The 20 sentences of speech-bench's prompts/speak-ja-JP.json through `speech-worker` in the voice file above,
one request at a time, after the worker is ready:

| Model | Device | Median first audio | p90 first audio | Real-time factor | Memory |
|---|---|---|---|---|---|
| v4.1-Small-MF F16, 4 steps | Apple M5, Metal | 0.23 s | 0.51 s | 0.17 | 2.2 GB |
| v4.1-Small-MF Q8_0, 4 steps | Apple M5, Metal | 0.25 s | 0.51 s | 0.18 | 1.5 GB |
| v4.1-Small F16, 16 steps | Apple M5, Metal | 1.12 s | 3.29 s | 0.34 | 2.2 GB |
| v4.1-Small F16, 40 steps | Apple M5, Metal | 2.64 s | 8.04 s | 0.64 | 2.2 GB |
| v4.1-Small-MF F16, 4 steps | RTX 2080, Vulkan | 0.13 s | 0.23 s | 0.10 | 2.1 GB |
| v4.1-Small-MF Q8_0, 4 steps | RTX 2080, Vulkan | 0.13 s | 0.22 s | 0.07 | 1.5 GB |
| v4.1-Small F16, 16 steps | RTX 2080, Vulkan | 0.49 s | 1.13 s | 0.14 | 2.2 GB |

audio.cpp v0.8.2 took 1.22 s (M5) and 0.80 s (RTX 2080) to the median first audio with v4.1-Small at 16
steps, answering with the whole sentence. Memory is the worker's peak memory footprint on the M5 and the
rise of the GPU's memory on the RTX 2080, with the F32 codec. The first audio comes after the text, the
whole sampler and the codec's first window, so it grows with the sentence.

## FastConformer

Speech recognition with NVIDIA NeMo's [FastConformer](https://arxiv.org/abs/2305.05084) models: a log-mel
frontend, a subsampling by 8 with depthwise convolutions, conformer layers with relative positional attention,
and a decoder that turns the encoder's frames into tokens. Implemented, each with the decoding NeMo's `transcribe()`
uses for it by default, for
[nvidia/parakeet-tdt_ctc-0.6b-ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) (Japanese) and
[nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (25 European languages, with
punctuation and capitals), with their TDT decoder, and
[reazon-research/reazonspeech-nemo-v2](https://huggingface.co/reazon-research/reazonspeech-nemo-v2) (Japanese, with
punctuation, for recordings of many minutes), with its RNN-T decoder:

- the frontend as NeMo runs it in evaluation (pre-emphasis, a centred STFT, the checkpoint's mel filters, 80 for
  parakeet-ja and ReazonSpeech and 128 for parakeet-v3, the log and the normalization of each mel bin over the
  recording), on the host in double precision,
- the subsampling, the 24 conformer layers and the convolution modules, on ggml, with the options in which the
  checkpoints differ read from the GGUF file: parakeet-v3 does not scale the subsampling's output and has no biases
  in its conformer layers, and ReazonSpeech attends locally,
- the attention: the parakeet models' relative positional attention over the whole recording, and ReazonSpeech's
  local attention, Longformer's as NeMo computes it, over the 128 frames (10.24 s) on either side of each frame
  and one global token, the first frame, that every frame attends to and that attends to every frame,
- the TDT decoder of the parakeet models: the prediction network (an embedding and two LSTM layers) and the joint
  on ggml, one step per emitted token, and NeMo's greedy decoding on the host, with the model's durations (0 to 4
  frames) and at most 10 tokens on one frame, as configured in the checkpoint,
- ReazonSpeech's RNN-T decoder: the same prediction network and a joint without durations, and the beam search its
  checkpoint configures, NeMo's alignment-length synchronous search (`alsd`) with a beam of 4, the best finished
  hypothesis chosen by its score per label, on the host,
- the SentencePiece pieces turned into text as NeMo's decoding writes it, with the space before each of the
  vocabulary's punctuation marks removed.

A recording goes through the encoder whole, however long, as `transcribe()` runs it: speech.cpp does not cut it
into segments. With local attention, ReazonSpeech's time and memory grow with the length of the recording; the
parakeet models attend over all of it, and theirs grow with its square
([ADR 0013](docs/adr/0013-local-attention-recognizes-long-audio-whole-as-transcribe-does.md)). The
reazonspeech package that ReazonSpeech's card recommends pads the audio with 0.5 s of silence on either side before
it calls `transcribe()`; speech.cpp does not, so its text is NeMo's for the audio as given.

The parakeet-ja checkpoint's CTC head is not converted: NeMo decodes with TDT by default, and the two write a
different text on some utterances ([ADR 0012](docs/adr/0012-the-recognizer-decodes-with-the-models-default-decoder.md)).
For the same reason ReazonSpeech decodes with its beam search alone, not with the greedy decoding NeMo could also run
on it: of the eight FLEURS utterances and the 65 s input of the checks below, NeMo's greedy decoding writes another
text for four: each lacks some of the commas and full stops the beam search writes, and one also has another word. Why recognition goes through this port is in
[ADR 0009](docs/adr/0009-speech-recognition-runs-through-a-fastconformer-port.md), and how it reaches the C API, the
worker, the server and `speech-asr` in [ADR 0011](docs/adr/0011-speech-recognition-is-a-task-of-every-entry-point.md).

### Models

Recognition needs one file per model: `parakeet-tdt_ctc-0.6b-ja-f16.gguf` from
[sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF](https://huggingface.co/sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF),
`parakeet-tdt-0.6b-v3-f16.gguf` from
[sakasegawa/parakeet-tdt-0.6b-v3-GGUF](https://huggingface.co/sakasegawa/parakeet-tdt-0.6b-v3-GGUF) or
`reazonspeech-nemo-v2-f16.gguf` from
[sakasegawa/reazonspeech-nemo-v2-GGUF](https://huggingface.co/sakasegawa/reazonspeech-nemo-v2-GGUF), whose cards
list their SHA-256. The repositories get these files in layout 1 with the release of 0.7.0; until then they hold
the files of earlier releases, which this one refuses. To convert them yourself, `reference/fastconformer/` pins NeMo
3.0.0 with PyTorch 2.10.0 and each checkpoint by revision, size and SHA-256:

```sh
cd reference/fastconformer
uv run python convert.py parakeet-tdt_ctc-0.6b-ja ../../models --type f16   # parakeet-tdt_ctc-0.6b-ja-f16.gguf, 1.2 GB
uv run python convert.py parakeet-tdt-0.6b-v3 ../../models --type f16       # parakeet-tdt-0.6b-v3-f16.gguf, 1.3 GB
uv run python convert.py reazonspeech-nemo-v2 ../../models --type f16       # reazonspeech-nemo-v2-f16.gguf, 1.2 GB
```

`--type f32` writes the same at 2.5 GB. The converter refuses a checkpoint with an option the C++ does not run
(another subsampling, attention or decoding, a prompt, a language tag to strip, a tokenizer piece it cannot write)
rather than write a file that would recognize differently from NeMo. GGUF files converted before 0.7.0 have no
layout and are refused; convert them again. The parakeet weights are NVIDIA's, under CC-BY-4.0, and ReazonSpeech's
are reazon-research's, under the Apache License 2.0.

### Use

```sh
speech-asr parakeet-tdt_ctc-0.6b-ja-f16.gguf utterance.wav     # the text on stdout
speech-asr parakeet-tdt-0.6b-v3-f16.gguf utterance.wav
speech-asr reazonspeech-nemo-v2-f16.gguf meeting.wav            # a recording of minutes, whole
speech-worker parakeet-tdt_ctc-0.6b-ja-f16.gguf                 # a recognition worker (The worker, above)
speech-server parakeet-tdt-0.6b-v3-f16.gguf                     # POST /v1/audio/transcriptions
```

The models recognize 16 kHz mono audio, to which the library resamples audio at another rate. parakeet-tdt_ctc-0.6b-ja
and reazonspeech-nemo-v2 recognize `ja`, and parakeet-tdt-0.6b-v3 `bg`, `cs`, `da`, `de`, `el`, `en`, `es`, `et`,
`fi`, `fr`, `hr`, `hu`, `it`, `lt`, `lv`, `mt`, `nl`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sv` and `uk`, the languages
of its model card. None has an input for a language: parakeet-v3 finds the language of the audio itself, as NeMo's
`transcribe()` runs it, without a prompt. A request's language is therefore only checked against the model's
(`languageSelectable` false) and changes nothing in the text. For a long recording, prefer reazonspeech-nemo-v2: the
parakeet models' memory grows with the square of the length. A request of the C API that sets `timestamps` also gets
the text's tokens and segments with their times in seconds, from the frames the decoding emitted them on, and the
segments end where the model's file says a sentence ends (Accuracy, below).

### Accuracy

`reference/fastconformer/dump.py` runs the official model on the CPU in float32 and saves every stage; the
checks compare each stage, given the dump's own inputs, with it:

```sh
cd reference/fastconformer
uv run python dump.py parakeet-tdt_ctc-0.6b-ja out <16 kHz mono WAVE files>
uv run python dump.py parakeet-tdt-0.6b-v3 out <16 kHz mono WAVE files>
uv run python dump.py reazonspeech-nemo-v2 out <16 kHz mono WAVE files>
uv run python dump.py --times reazonspeech-nemo-v2 out   # NeMo's times alone, added to the dumps already there
cd ../..
build/fastconformer-frontend-check <model.gguf> reference/fastconformer/out
build/fastconformer-encoder-check <model.gguf> reference/fastconformer/out [gpu|cpu|device name]
build/fastconformer-transducer-check <model.gguf> reference/fastconformer/out [gpu|cpu|device name]
build/fastconformer-times-check <model.gguf> reference/fastconformer/out [gpu|cpu|device name]
```

`dump.py` writes the dumps of each model to `out/<model>/<file name>/`, and each check reads those of the model
it is given, by the GGUF file's `general.name`.

On three utterances of FLEURS ja_jp's test split (12677001980660723842, 6.36 s; 13903496305700695803, 10.50 s;
2630315561484880103, 25.50 s), on an Apple M5:

| Check | CPU, F32 | CPU, F16 | Metal, F32 or F16 |
|---|---|---|---|
| Features (`fastconformer-frontend-check`) | 117 to 127 dB SNR | the same | the same (on the host) |
| Subsampling (`fastconformer-encoder-check`) | 123 dB | 60 to 61 dB | 68 to 70 dB |
| Encoder output, after 24 layers | 114 to 118 dB | 55 to 56 dB | 63 to 64 dB |
| Prediction network on the dump's labels (`fastconformer-transducer-check`) | 130 to 133 dB | 57 to 60 dB | 132 to 134 dB with F32, 66 to 70 dB with F16 |
| Joint log-probabilities on the dump's frames and prediction outputs | 140 dB | 76 dB | 85 to 87 dB |
| Greedy tokens and text from the dump's encoder output | equal | equal | equal |
| Text from the audio, every stage ours | equal on all three | equal on all three | equal on all three |

For parakeet-tdt-0.6b-v3, on twelve utterances of FLEURS' test split, three each of en_us, de_de, fr_fr and
es_419 (5.64 to 23.40 s, one of each language over 20 s), on an Apple M5:

| Check | CPU, F32 | CPU, F16 | Metal, F32 or F16 |
|---|---|---|---|
| Features | 94 to 125 dB SNR | the same | the same (on the host) |
| Subsampling | 122 dB | 59 to 61 dB | 69 to 70 dB |
| Encoder output, after 24 layers | 106 to 114 dB | 31 to 55 dB | 52 to 61 dB |
| Prediction network on the dump's labels | 130 to 133 dB | 56 to 60 dB | 130 to 133 dB with F32, 66 to 69 dB with F16 |
| Joint log-probabilities on the dump's frames and prediction outputs | 141 to 142 dB | 77 dB | 88 to 89 dB |
| Greedy tokens and text from the dump's encoder output | equal | equal | equal |
| Text from the audio, every stage ours | equal on all twelve | equal on all twelve | equal on all twelve |

The lowest features, 94 dB on fr_fr 10043298898524273336, are the dump's own float32: the same steps in float64 in
PyTorch differ from it as much. parakeet-v3's conformer layers carry values of 250 to 500, where parakeet-ja's stay
near 100, so half precision costs more: most on the CPU with F16 weights, whose dot product on ARM also sums in half
precision, down to 31 dB on de_de 10229344228128634115, which still gives NeMo's text.

Metal gives the same numbers with F32 and F16 weights for the encoder and the joint, since its matrix kernel
rounds both its inputs to half precision either way; the prediction network multiplies a single vector, which
Metal does in float32.

For reazonspeech-nemo-v2, on eight utterances of FLEURS ja_jp's test split (the three above, 6183819757443715774,
8.94 s; 14931648021649736041, 10.86 s; 16124271561776664380, 14.52 s; 17416616907086415885, 17.88 s;
9518252661993015549, 28.20 s) and two long inputs, the first 4 and the first 21 utterances of the split in the order
of its `test.tsv` joined end to end (64.80 s and 311.22 s), on an Apple M5:

| Check | CPU, F32 | CPU, F16 | Metal, F32 | Metal, F16 |
|---|---|---|---|---|
| Features | 113 to 128 dB SNR | the same | the same (on the host) | the same |
| Subsampling | 122 to 123 dB | 61 to 62 dB | 70 dB | 70 dB |
| Encoder output, after 24 layers of local attention | 104 to 119 dB, 95.9 dB on the 311 s input | 46 to 56 dB, 29.7 dB on the 311 s input | 55 to 71 dB, 37.7 dB on the 65 s input | the same |
| Prediction network on the dump's labels, along the beam's hypotheses | 125 to 128 dB | 61 to 64 dB | 125 to 128 dB | 69 to 73 dB |
| Joint log-probabilities on the dump's frames and prediction outputs | 124 to 126 dB | 71 to 72 dB | 83 to 88 dB | 83 to 88 dB |
| Beam search's tokens and text from the dump's encoder output | equal | equal | equal | equal |
| Text from the audio, every stage ours | equal on all ten | equal on all ten | equal on all ten | equal on all ten |

The dumps record every evaluation of the beam search, 464 to 2,234 joint evaluations an utterance and 18,272 for the
311 s input, and the check replays the search on the dump's encoder output. In half precision the error of the
encoder grows in a few near-silent frames between the joined utterances, from 70 dB at the first layer to 42 dB at
the thirteenth in the 65 s input, and the text is NeMo's all the same; on the CPU in float32 a local attention whose
band missed one frame on one side would give 27 to 50 dB, far below what the arithmetic explains.
`fastconformer-encoder-check` therefore asks for 90 dB on the CPU with F32 weights and 25 dB where half precision
enters.

The 25 dumps' audio at 44.1 and 48 kHz, made with torchaudio's `kaiser_best`, gives with F32 weights on the CPU and on
Metal the text NeMo gives for the same file brought to 16 kHz by torchaudio with `kaiser_best`, on all 50. That text is
the dump's own on 21 of 25 at either rate; NeMo's `transcribe()` of the files, which resamples them with soxr, gives
the dump's text on 22 and speech.cpp's on 22. The others differ in a hyphen, a comma, a full stop and a few words of
the 311 s input: making the files and resampling them back takes away the band above 7.6 kHz, which the 16 kHz audio
has, and soxr takes away a slightly different one.

`dump.py` also saves the times NeMo's `transcribe(timestamps=True)` gives: the frame each token was emitted on and,
for TDT, the duration predicted with it, and the spans of the tokens and of the segments in frames and in seconds.
`fastconformer-times-check` decodes the dump's encoder output and compares the frames, the durations, the tokens'
spans in frames and in seconds and the segments NeMo's separators give, at the ends of words as NeMo ends them, with
them: on every dump of the three models, on the CPU with F32 weights and on Metal with F16, all are NeMo's exactly.
For the Japanese models, whose text has no spaces and so no end of a word before its last token, it also cuts
segments as the recognizer does with their files' `fastconformer.segment.breaks`, after `。`, `？`, `！`, `?` and `!`
wherever they stand, and checks that each segment ends there, at a
separator that ends a word, or with the last token. NeMo's beam search records with each token
the step of its search, the frame plus the tokens before it, so that its times for ReazonSpeech run past the end of
the audio, to 386.16 s for the 311.22 s input; speech.cpp keeps the frame, and the check compares the step it gives.
From the audio, every stage ours, the frames are NeMo's but for one token of ReazonSpeech's 9518252661993015549 on
Metal with F16, one frame early: the two alignments of its tokens differ by 0.007 in log-probability on the CPU in
float32, and half precision reverses them. None of the checkpoints sets NeMo's separators, which are `.`, `!` and `?`
by default, and the vocabularies of the Japanese models hold `。` and the ASCII `?` and `!`, not `！` or `？`.

### Speed

`speech-asr` on an Apple M5 with F16 weights on Metal, after loading: 0.07 s for the 6.36 s utterance, 0.11 s for
the 10.50 s one and 0.28 s for the 25.50 s one. Of the last, the encoder takes 0.20 to 0.26 s, the TDT decoding
0.07 to 0.08 s (107 steps of the prediction network, about 0.6 ms each on the GPU) and the frontend 13 to 15 ms. On
the CPU with F32 weights and ggml's default four threads it takes 6.3 s, 0.2 to 0.3 s of it the decoding.

parakeet-tdt-0.6b-v3 on the same M5 with F16 weights on Metal, after loading: 0.08 s for 5.64 s of audio, 0.12 to
0.17 s for 10.20 to 12.84 s and 0.24 to 0.32 s for 20.76 to 23.40 s, a real-time factor of 0.012 over the twelve
utterances. Of the 23.40 s one, the encoder takes 0.20 s, the TDT decoding 0.08 s (128 steps of the prediction
network, and a joint over 8,198 outputs where parakeet-ja's has 3,078) and the frontend 16 ms. On the CPU with F32
weights it takes 9.4 s.

reazonspeech-nemo-v2 on the same M5 with F16 weights on Metal, after loading: 0.17 s for the 6.36 s utterance, 0.27 s
for the 10.50 s one, 0.55 s for the 25.50 s one and 0.63 s for the 28.20 s one, 1.4 s for the 64.80 s input and
4.9 s for the 311.22 s one, a real-time factor of 0.016 to 0.027. The beam search takes about half of it: each of its
steps, one graph of the new predictions and the joint of up to 4 hypotheses on the GPU, takes about 0.7 ms, and the
311 s input takes some 4,860 steps (timed apart in `fastconformer-transducer-check`: the encoder 3.7 s, the beam
search 3.3 s, the frontend 0.14 s). The process's peak memory footprint on the CPU with F16 weights, which
holds every buffer, is 1.35 GB for 6.36 s, 1.55 GB for 64.80 s and 2.35 GB for 311.22 s, growing with the length;
parakeet-tdt_ctc-0.6b-ja's is 1.30, 1.49 and 4.14 GB, growing with its square, and the 311 s input takes it 62 s on
the CPU where ReazonSpeech takes 26 s.

## GGUF files

Every model is one GGUF file, its codec included, which `reference/<model>/convert.py` writes from the checkpoint
that `reference/<model>/pins.py` pins by revision (docs/adr/0015). The file says the version of its layout in
`speech.layout`, 1 for every family, and in `speech.requires` the first release whose reader takes it, 0.7.0. A
reader takes the layout it knows; a newer one is refused with a message that names `speech.requires`, and a file
without `speech.layout`, converted for a release before 0.7.0, is refused as such. Every key below is required in
its family's layout unless the table says when it is present, and has exactly the type listed: a key missing or of
another type is refused with a message that names it, and so is a string that names a kind other than the ones
listed. The tensors are exactly the ones the keys call for: a tensor missing or one not called for is refused. A key
whose meaning the tables leave empty means what the official configuration's field of the same name means.

### Keys of every model file

| Key | Type | Meaning | Source |
|---|---|---|---|
| `general.architecture` | string | the family: `qwen3-tts`, `irodori-tts` or `fastconformer` | the converter |
| `general.name` | string | the model's name | the pinned repository's name (`Qwen3-TTS-12Hz-0.6B-CustomVoice`, `Irodori-TTS-v4.1-Small-MF`, `parakeet-tdt-0.6b-v3`) |
| `general.license` | string | SPDX identifier | the model card |
| `general.source.url` | string | `https://huggingface.co/<repository>/tree/<revision>` | the pin |
| `speech.layout` | u32 | 1, the version of the family's layout | the converter |
| `speech.requires` | string | `0.7.0`, the first release whose reader takes this layout | the converter's table of layouts |
| `speech.task` | string | `synthesis` or `recognition`; must be the family's | the converter |
| `speech.sample_rate` | u32 | the rate of the audio made or recognized | Qwen3-TTS: `speech_tokenizer/config.json` `output_sample_rate`; Irodori-TTS: the DACVAE's `sample_rate`; FastConformer: the featurizer's `sample_rate` |
| `speech.languages` | [string] | BCP 47 tags, sorted | Qwen3-TTS: the names of `codec_language_id` through the converter's tag table, dialects left out; the others: the model card |
| `speech.language_use` | string | `steers` or `checked`; must be what the family does | `steers` for qwen3-tts, `checked` for the others |
| `speech.voices` | [string] | the built-in voices' names; present for qwen3-tts alone | `talker_config.spk_id`'s names, sorted |
| `speech.voice_languages` | [string] | each voice's language, aligned with `speech.voices` | the model card's "Native Language" column through the tag table (Dylan's and Eric's dialects are `zh`) |
| `speech.voice_genders` | [string] | `female` or `male`, aligned | the model card's "Voice Description" column |
| `speech.voice_descriptions` | [string] | the description, aligned | the model card's "Voice Description" column |

### qwen3-tts

From the checkpoint's `config.json` (`talker_config`, its `code_predictor_config`, and the top level),
`generation_config.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json` and `speech_tokenizer/config.json`
(`decoder_config`), and the official code at the commit the converter's environment pins (QwenLM/Qwen3-TTS@022e286).

| Key | Type | Meaning | Source |
|---|---|---|---|
| `qwen3-tts.talker.hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim` | u32 | | `talker_config` |
| `qwen3-tts.talker.rms_norm_eps`, `rope_theta` | f32 | | `talker_config` |
| `qwen3-tts.talker.vocab_size` | u32 | the codec ids, audio codes and control tokens | `talker_config.vocab_size` |
| `qwen3-tts.talker.num_code_groups` | u32 | codes per frame | `talker_config.num_code_groups` |
| `qwen3-tts.talker.max_position_embeddings` | u32 | the positions of the talker's cache, prompt and frames together | `talker_config.max_position_embeddings` |
| `qwen3-tts.talker.codec_bos_id`, `codec_eos_token_id` (the end of speech), `codec_pad_id`, `codec_think_id`, `codec_nothink_id`, `codec_think_bos_id`, `codec_think_eos_id` | u32 | | `talker_config` |
| `qwen3-tts.talker.suppressed_tokens` | u32 | the ids at the end of the vocabulary that sampling never picks, the end of speech excepted (1024) | the official `generate()`'s `suppress_tokens`, `range(vocab_size - 1024, vocab_size)` |
| `qwen3-tts.code_predictor.hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size` | u32 | | `code_predictor_config` |
| `qwen3-tts.code_predictor.rms_norm_eps`, `rope_theta` | f32 | | `code_predictor_config` |
| `qwen3-tts.text.tts_bos_token_id`, `tts_eos_token_id`, `tts_pad_token_id`, `im_start_token_id`, `im_end_token_id`, `assistant_token_id` | u32 | | `config.json` |
| `qwen3-tts.text.newline_token_id` | u32 | the token of "\n" in the chat template (198) | `vocab.json`'s id of `Ċ`, the byte-level form of "\n" |
| `qwen3-tts.language_ids` | [i32] | the codec id of each of `speech.languages`, aligned | `talker_config.codec_language_id` |
| `qwen3-tts.speaker_ids` | [i32] | the codec id of each of `speech.voices`, aligned | `talker_config.spk_id` |
| `qwen3-tts.dialect_ids` | [i32] | the codec id of the dialect each voice speaks, or −1, aligned | `talker_config.spk_is_dialect` through `codec_language_id` |
| `qwen3-tts.dialect_language` | string | the language in which, as with `auto`, a voice with a dialect speaks it (`zh`) | the official prompt's `language.lower() in ["chinese", "auto"]` through the tag table |
| `qwen3-tts.generation.min_frames` | u32 | frames before the end of speech may be sampled (2) | the official `generate()`'s `min_new_tokens` |
| `qwen3-tts.generation.max_frames` | u32 | the model's limit in frames (8192) | `generation_config.json` `max_new_tokens` |
| `qwen3-tts.generation.talker.do_sample` | bool | | `generation_config.json` `do_sample` |
| `qwen3-tts.generation.talker.temperature`, `top_p`, `repetition_penalty` | f32 | | `generation_config.json` |
| `qwen3-tts.generation.talker.top_k` | u32 | | `generation_config.json` |
| `qwen3-tts.generation.code_predictor.do_sample` | bool | | `subtalker_dosample` |
| `qwen3-tts.generation.code_predictor.temperature`, `top_p` | f32 | | `subtalker_temperature`, `subtalker_top_p` |
| `qwen3-tts.generation.code_predictor.top_k` | u32 | | `subtalker_top_k` |
| `qwen3-tts.generation.code_predictor.repetition_penalty` | f32 | | `code_predictor_config.repetition_penalty`, which the code predictor's `generate()` uses since the official code passes none |
| `qwen3-tts.tokenizer.tokens` | [string] | the BPE vocabulary in id order | `vocab.json` and `tokenizer_config.json` `added_tokens_decoder` |
| `qwen3-tts.tokenizer.merges` | [string] | merges in rank order | `merges.txt` |
| `qwen3-tts.codec.num_quantizers` | u32 | must equal `talker.num_code_groups` | `decoder_config.num_quantizers` |
| `qwen3-tts.codec.latent_dim`, `codebook_dim`, `hidden_size`, `num_attention_heads`, `head_dim`, `num_hidden_layers`, `sliding_window` | u32 | | `decoder_config` |
| `qwen3-tts.codec.num_key_value_heads` | u32 | must equal the heads, the one form the C++ runs | `decoder_config.num_key_value_heads` |
| `qwen3-tts.codec.rms_norm_eps`, `rope_theta` | f32 | | `decoder_config` |
| `qwen3-tts.codec.upsample_rates` | [i32] | the decoder blocks' strides | `decoder_config.upsample_rates` |
| `qwen3-tts.codec.upsampling_ratios` | [i32] | the upsampling stages' strides | `decoder_config.upsampling_ratios` |

A frame lasts the product of `upsample_rates` and `upsampling_ratios` (1920) samples at `speech.sample_rate`, 0.08 s.
The text a request takes at most is `talker.max_position_embeddings` − `generation.max_frames` − 11, the rows its
prompt adds to the text (24565 tokens for both sizes).

Tensors, with T = `talker.num_hidden_layers`, C = `code_predictor.num_hidden_layers`, G = `talker.num_code_groups`,
Q = `codec.num_quantizers`, L = `codec.num_hidden_layers`, U = the length of `codec.upsampling_ratios`, B = the
length of `codec.upsample_rates`:

- `talker.text_embd`, `talker.text_proj.fc1.{weight,bias}`, `talker.text_proj.fc2.{weight,bias}`, `talker.codec_embd`,
  `talker.codec_head`, `talker.norm`;
- `talker.blk.{0..T-1}.` and `cp.blk.{0..C-1}.` each with `attn_norm`, `ffn_norm`, `attn_q`, `attn_k`, `attn_v`,
  `attn_o`, `attn_q_norm`, `attn_k_norm`, `ffn_gate`, `ffn_up`, `ffn_down`;
- `cp.norm`, `cp.codec_embd.{0..G-2}`, `cp.head.{0..G-2}`, and `cp.in_proj.{weight,bias}` when
  `code_predictor.hidden_size` differs from `talker.hidden_size` (the 1.7B model), as the official model makes
  `small_to_mtp_projection` a Linear exactly then;
- `codec.vq.first.codebook.0`, `codec.vq.first.out_proj`, `codec.vq.rest.codebook.{0..Q-2}`, `codec.vq.rest.out_proj`;
- `codec.pre_conv.{weight,bias}`, `codec.tf.in_proj.{weight,bias}`, `codec.tf.out_proj.{weight,bias}`, `codec.tf.norm`,
  and `codec.tf.blk.{0..L-1}.` with `attn_norm`, `ffn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_o`, `attn_scale`,
  `ffn_gate`, `ffn_up`, `ffn_down`, `ffn_scale`;
- `codec.up.{0..U-1}.` with `tconv.{weight,bias}`, `dwconv.{weight,bias}`, `norm.{weight,bias}`, `pw1.{weight,bias}`,
  `pw2.{weight,bias}`, `gamma`;
- `codec.dec.in_conv.{weight,bias}`, `codec.dec.blk.{0..B-1}.` with `snake.{alpha,inv_beta}`, `tconv.{weight,bias}`
  and `res.{0,1,2}.` with `snake1.{alpha,inv_beta}`, `conv1.{weight,bias}`, `snake2.{alpha,inv_beta}`,
  `conv2.{weight,bias}`; `codec.dec.out_snake.{alpha,inv_beta}`, `codec.dec.out_conv.{weight,bias}`.

The three residual units per decoder block (dilations 1, 3 and 9) are the architecture's, as the official module
builds them, and stay in the C++.

### irodori-tts

From the checkpoint's `model.safetensors` metadata (`config_json`, `text_encoder_config_json`), its `tokenizer/` and
model card, the DACVAE codec (Aratako/Semantic-DACVAE-Japanese-32dim at its pin), the official runtime at the commit
the converter's environment pins (`SamplingRequest`'s defaults in `irodori_tts/inference_runtime.py`) and
Irodori-TTS-Server@61012c760f22f7b4a6c21c5c5f8f9e148120b6f9 for the speed.

| Key | Type | Meaning | Source |
|---|---|---|---|
| `irodori-tts.flow` | string | `meanflow` or `rf_velocity` | `config_json` `flow_parameterization` (`rf_velocity` when absent, the config's default) |
| `irodori-tts.latent_dim` | u32 | the codec latent's channels, the DiT's and the speaker encoder's input | `config_json` `latent_dim`; the converter checks the codec's `codebook_dim` equals it |
| `irodori-tts.norm_eps` | f32 | | `config_json` `norm_eps` |
| `irodori-tts.rope_theta` | f32 | the RoPE base of the speaker encoder and the DiT (10000) | the official `precompute_freqs_cis()` default, which both keep |
| `irodori-tts.text.hidden_size`, `num_heads`, `num_layers` | u32 | ModernBERT-ja | `text_encoder_config_json` `hidden_size`, `num_attention_heads`, `num_hidden_layers` |
| `irodori-tts.text.norm_eps` | f32 | | `norm_eps` |
| `irodori-tts.text.window` | u32 | the tokens a local layer reaches on either side | `local_attention / 2` |
| `irodori-tts.text.layer_global` | [i32] | 1 for a layer with global attention | `layer_types`, `full_attention` |
| `irodori-tts.text.rope_theta_global`, `rope_theta_local` | f32 | | `rope_parameters.full_attention.rope_theta`, `rope_parameters.sliding_attention.rope_theta` |
| `irodori-tts.text.dim` | u32 | the text condition's channels | `config_json` `text_dim` |
| `irodori-tts.text.max_tokens` | u32 | the longest text (256) | `config_json` `max_text_len` |
| `irodori-tts.speaker.dim`, `num_layers`, `num_heads`, `patch_size` | u32 | | `config_json` `speaker_dim`, `speaker_layers`, `speaker_heads`, `speaker_patch_size` |
| `irodori-tts.duration.num_layers` | u32 | | `duration_layers` |
| `irodori-tts.dit.dim`, `num_layers`, `num_heads`, `timestep_dim` | u32 | | `model_dim`, `num_layers`, `num_heads`, `timestep_embed_dim` |
| `irodori-tts.sampler.default_steps` | u32 | the default of the sampler's steps (4 or 40) | the runtime's `num_steps` default, 4 for MeanFlow and 40 otherwise |
| `irodori-tts.sampler.cfg_text` | f32 | present when `flow` is `rf_velocity` (3.0) | `SamplingRequest.cfg_scale_text` |
| `irodori-tts.sampler.cfg_speaker` | f32 | the same (5.0) | `SamplingRequest.cfg_scale_speaker` |
| `irodori-tts.sampler.cfg_min_t`, `cfg_max_t` | f32 | the same (0.5 and 1.0) | `SamplingRequest.cfg_min_t`, `cfg_max_t` |
| `irodori-tts.length.min_seconds`, `max_seconds` | f32 | the shortest and the longest speech (0.5 and 30) | `SamplingRequest.min_seconds`, `max_seconds` |
| `irodori-tts.length.min_speed`, `max_speed` | f32 | the speed's bounds (0.25 and 4) | Irodori-TTS-Server's speed bounds, OpenAI's |
| `irodori-tts.reference.max_seconds` | f32 | the longest reference recording (120) | `config_json` `ref_max_seconds` |
| `irodori-tts.reference.lufs` | f32 | the loudness a reference is brought to (−16) | `SamplingRequest.ref_normalize_db` |
| `irodori-tts.tail.window` | u32 | the frames over which the latent's tail is found flat (20) | `SamplingRequest.tail_window_size` |
| `irodori-tts.tail.std_threshold`, `mean_threshold` | f32 | (0.05 and 0.1) | `SamplingRequest.tail_std_threshold`, `tail_mean_threshold` |
| `irodori-tts.tokenizer.tokens` | [string] | the Unigram pieces | `tokenizer/tokenizer.json` `model.vocab` |
| `irodori-tts.tokenizer.scores` | [f64] | their scores | the same |
| `irodori-tts.tokenizer.added_ids` | [i32] | | `added_tokens` |
| `irodori-tts.tokenizer.bos_id` | u32 | | `tokenizer_config.json` `bos_token` |
| `irodori-tts.tokenizer.unknown_id` | u32 | | `model.unk_id` |
| `irodori-tts.codec.hop_length` | u32 | samples per latent frame | the DACVAE's `hop_length`; the converter checks it is the product of the encoder's rates |
| `irodori-tts.codec.encoder_rates`, `decoder_rates` | [i32] | | the DACVAE's `encoder_rates`, `decoder_rates` |
| `irodori-tts.codec.sha256` | string | the codec's identity, which voice files carry | the SHA-256 the converter computes over the official codec's tensors (below) |

The codec's hash is SHA-256 over the tensors of `weights.pth` as `DACVAE.load()` gives them, before the weight
normalization is folded: for each tensor in ascending order of its name, the name in UTF-8, a 0 byte, the number of
dimensions as a little-endian u32, each dimension as a little-endian u64, and the values as little-endian float32 in
row-major order. Every conversion of the same official codec carries the same hash, whatever type it stores:
Semantic-DACVAE-Japanese-32dim at its pin is `67cb1a241c3b75d7c5f46c966fca711e7962422da98f7f32ee2863d39dbe794d`.

Tensors, with X = `text.num_layers`, S = `speaker.num_layers`, D = `duration.num_layers`, N = `dit.num_layers`, E and
F the lengths of `codec.encoder_rates` and `codec.decoder_rates`:

- `text.embd`, `text.embd_norm`, `text.blk.{0..X-1}.` with `attn_q`, `attn_k`, `attn_v`, `attn_out`, `ffn_norm`,
  `ffn_act`, `ffn_gate`, `ffn_down`, and `attn_norm` for every layer but the first, which ModernBERT does not
  normalize; `text.final_norm`, `text.proj.{weight,bias}`, `text.proj.res_norm`, `text.proj.res_up.{weight,bias}`,
  `text.proj.res_down.{weight,bias}`, `text.norm`;
- `speaker.in_proj.{weight,bias}`, `speaker.blk.{0..S-1}.` with `attn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_o`,
  `attn_gate`, `q_norm`, `k_norm`, `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_down`; `speaker.norm`;
- `duration.in_proj.{weight,bias}`, `duration.blk.{0..D-1}.` with `norm`, `mod.{weight,bias}`,
  `caption_mod.{weight,bias}`, `ffn_gate`, `ffn_up`, `ffn_down`; `duration.out_norm`, `duration.out_proj.{weight,bias}`,
  `duration.null_caption`;
- `dit.cond.{0,1,2}`, and `dit.delta_cond.{0,1,2}` when `flow` is `meanflow`; `dit.in_proj.{weight,bias}`,
  `dit.blk.{0..N-1}.` with `attn_q`, `attn_k`, `attn_v`, `attn_o`, `attn_gate`, `attn_k_text`, `attn_v_text`,
  `attn_k_speaker`, `attn_v_speaker`, `q_norm`, `k_norm`, `ffn_gate`, `ffn_up`, `ffn_down`, and for `attn_ada` and
  `ffn_ada` each of `shift`, `scale`, `gate` with `down`, `up.weight`, `up.bias`; `dit.out_norm`,
  `dit.out_proj.{weight,bias}`;
- `codec.enc.conv_in.{weight,bias}`, `codec.enc.blk.{0..E-1}.` with `res.{0,1,2}.` (`snake1.{alpha,inv_alpha}`,
  `conv1.{weight,bias}`, `snake2.{alpha,inv_alpha}`, `conv2.{weight,bias}`), `snake.{alpha,inv_alpha}`,
  `down.first`, `down.second`, `down.bias`; `codec.enc.snake.{alpha,inv_alpha}`, `codec.enc.conv_out.{weight,bias}`,
  `codec.bottleneck.mean.{weight,bias}`;
- `codec.dec.in_proj.{weight,bias}`, `codec.dec.conv_in.{weight,bias}`, `codec.dec.blk.{0..F-1}.` with
  `snake.{alpha,inv_alpha}`, `up.{weight,bias}` and `res.{0,1,2}.` as in the encoder;
  `codec.dec.out_snake.{alpha,inv_alpha}`, `codec.dec.conv_out.{weight,bias}`.

Constants that stay in the C++, since they are not the model's: the decoder's windows of 12 and 48 frames, which are
how speech.cpp streams a latent the official runtime decodes whole; the encoder's and decoder's margins of 8 and 10
frames, which follow from the codec's architecture; and SentencePiece's penalty for an unknown piece.

### fastconformer

From the `.nemo` checkpoint as NeMo 3.0.0 restores it (`model.cfg`, the featurizer, the encoder, the decoding and the
SentencePiece model), NeMo's own constants where it has them, and the model card for the languages and the license.

| Key | Type | Meaning | Source |
|---|---|---|---|
| `fastconformer.frontend.n_fft` | u32 | | featurizer `n_fft` |
| `fastconformer.frontend.hop_length` | u32 | samples per mel frame | featurizer `hop_length` |
| `fastconformer.frontend.n_mels` | u32 | the mel bins, also the subsampling's input width | featurizer `nfilt` |
| `fastconformer.frontend.preemphasis` | f32 | | featurizer `preemph` |
| `fastconformer.frontend.log_guard` | f32 | | featurizer `log_zero_guard_value` |
| `fastconformer.frontend.std_guard` | f32 | (1e-5) | `CONSTANT` in NeMo's `features.py` |
| `fastconformer.encoder.d_model`, `num_layers`, `num_heads`, `conv_kernel` | u32 | | `cfg.encoder.d_model`, `n_layers`, `n_heads`, `conv_kernel_size` |
| `fastconformer.encoder.subsampling_factor` | u32 | a power of two | `cfg.encoder.subsampling_factor` |
| `fastconformer.encoder.norm_eps` | f32 | | the layers' `norm_out.eps` |
| `fastconformer.encoder.pos_base` | f32 | (10000) | `INF_VAL` of NeMo's `multi_head_attention.py` |
| `fastconformer.encoder.xscale` | f32 | | `encoder.xscale`, or 1 when it is None |
| `fastconformer.encoder.ff_factor` | f32 | | the layers' `fc_factor` |
| `fastconformer.encoder.use_bias` | bool | whether the linear layers and pointwise convolutions have biases | the layers' `use_bias` |
| `fastconformer.encoder.attention` | string | `rel_pos` or `rel_pos_local_attn` | `encoder.self_attention_model` |
| `fastconformer.encoder.attention_context` | u32 | present for `rel_pos_local_attn`: frames on either side | `encoder.att_context_size[0]` |
| `fastconformer.encoder.global_tokens` | u32 | present for `rel_pos_local_attn` | `encoder.global_tokens` |
| `fastconformer.decoder.kind` | string | `tdt` or `rnnt` | the decoding `transcribe()` runs (`GreedyBatchedTDTInfer` or `BeamRNNTInfer`) |
| `fastconformer.decoder.blank_id` | u32 | | `decoding.blank_id` |
| `fastconformer.decoder.prediction_layers` | u32 | the prediction network's LSTM layers | `decoder.pred_rnn_layers` |
| `fastconformer.decoder.tdt.durations` | [i32] | present for `tdt` | `decoding.cfg.durations` |
| `fastconformer.decoder.tdt.max_symbols` | u32 | present for `tdt`: tokens on one frame at most | the decoding's `max_symbols` |
| `fastconformer.decoder.rnnt.beam_size` | u32 | present for `rnnt` | the beam search's `beam_size` |
| `fastconformer.decoder.rnnt.score_norm` | bool | present for `rnnt` | `score_norm` |
| `fastconformer.decoder.rnnt.max_target_ratio` | f32 | present for `rnnt` | `alsd_max_target_length` |
| `fastconformer.segment.separators` | [string] | the marks that end a segment where a word ends, as NeMo ends one | the decoding's `segment_seperators`, or NeMo's default `.`, `?`, `!` when the checkpoint sets none (none of the three does) |
| `fastconformer.segment.breaks` | [string] | the marks that end a segment wherever they stand: `。`, `？`, `！`, `?`, `!` for a model whose languages are written without spaces, which NeMo's word ends never cut, and none otherwise | speech.cpp's own, from the model's languages |
| `fastconformer.tokenizer.tokens` | [string] | the SentencePiece pieces in id order | the tokenizer's model proto |
| `fastconformer.tokenizer.unknown_id` | u32 | | `unk_id()` |
| `fastconformer.tokenizer.unknown_surface` | string | | `trainer_spec.unk_surface` |
| `fastconformer.tokenizer.strip_leading_space` | bool | whether the first "▁" of the text is dropped | `normalizer_spec.add_dummy_prefix or remove_extra_whitespaces` |
| `fastconformer.tokenizer.punctuation` | [string] | the marks before which the decoding removes a space | `decoding.supported_punctuation` |

An encoder frame lasts `frontend.hop_length × encoder.subsampling_factor / speech.sample_rate` (160 × 8 / 16000 =
0.08 s).

Tensors, with L = `encoder.num_layers`, K = log2(`encoder.subsampling_factor`) and P = `decoder.prediction_layers`:

- `frontend.window`, `frontend.filterbank`;
- `sub.conv.0.{weight,bias}`, `sub.conv.{1..K-1}.` with `dw.{weight,bias}` and `pw.{weight,bias}`,
  `sub.out.{weight,bias}`;
- `blk.{0..L-1}.` with `ff1_norm.{weight,bias}`, `ff2_norm.{weight,bias}`, `attn_norm.{weight,bias}`,
  `conv_norm.{weight,bias}`, `out_norm.{weight,bias}`, `attn_pos.weight`, `attn_pos_bias_u`, `attn_pos_bias_v`,
  `conv_dw.{weight,bias}`, and `ff1_up`, `ff1_down`, `ff2_up`, `ff2_down`, `attn_q`, `attn_k`, `attn_v`, `attn_out`,
  `conv_pw1_a`, `conv_pw1_gate` and `conv_pw2`, each a `.weight` and, when `encoder.use_bias` is true, a `.bias`;
- `pred.embed.weight`, `pred.lstm.{0..P-1}.` with `ih.weight`, `hh.weight`, `bias`;
- `joint.enc.{weight,bias}`, `joint.pred.{weight,bias}`, `joint.out.{weight,bias}`; `joint.out` has
  `decoder.blank_id` + 1 outputs, and as many more as `decoder.tdt.durations` has entries for `tdt`.

### Voice files

A voice file of Irodori-TTS is a GGUF file of its own, layout 1:

| Key | Type | Meaning |
|---|---|---|
| `general.architecture` | string | `irodori-tts-voice` |
| `speech.layout` | u32 | 1 |
| `speech.requires` | string | `0.7.0` |
| `irodori-tts-voice.codec_sha256` | string | the `irodori-tts.codec.sha256` of the model that encoded it; a model with another hash refuses it |
| `irodori-tts-voice.reference_seconds` | f32 | the reference recording's length |
| `irodori-tts-voice.reference_sample_rate` | u32 | the reference recording's rate, before it was resampled |
| `irodori-tts-voice.device_kind` | string | `cpu`, `gpu` or `igpu`: the kind of device that encoded it |

and one tensor, `latent`, F32 with ne = [`latent_dim`, frames]. Voice files made before 0.7.0 have no `speech.layout`
and are refused; they are made again from their WAVE files.

## License

MIT, see [LICENSE](LICENSE). The model weights are their authors': Qwen3-TTS is the Qwen team's, under the
Apache License 2.0. Irodori-TTS v4.1-Small and v4.1-Small-MF are Aratako's, under the MIT License with the
ethical restrictions of their model cards (no voice cloning without consent, no deepfakes or
misinformation). Semantic-DACVAE-Japanese-32dim is Aratako's and MIT on its card; it derives from Meta's
facebook/dacvae-watermarked, which is under the Apache License 2.0. That card's text also names the SAM
License, a sentence left from the README of facebookresearch/dacvae, which Meta corrected to Apache-2.0 on
2025-12-19; the repository's LICENSE has been Apache-2.0 from its first commit.
