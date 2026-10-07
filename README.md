# speech.cpp

Speech synthesis and speech recognition in C++ on [ggml](https://github.com/ggml-org/ggml), as a library with a C
API (`include/speech.h`) and one executable on it, `speech`, whose subcommands speak text, recognize speech, serve
OpenAI's audio API over HTTP and run the worker process that [ASIST](https://github.com/nyosegawa/asist) starts. It
targets Metal, Vulkan and CUDA; it is checked on Metal, on Vulkan (NVIDIA) and on the CPU. Every stage of a port is
checked against the official implementation.

| Organization | Model line | Models | Family | Task | Converted weights |
|---|---|---|---|---|---|
| Qwen | Qwen3-TTS-12Hz | 0.6B and 1.7B CustomVoice | `qwen3-tts` | speech synthesis with the named speakers, streamed frame by frame | [sakasegawa/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF), [sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF) |
| Aratako | Irodori-TTS | v4.1-Small-MF and v4.1-Small | `irodori-tts` | Japanese speech synthesis in the voice of a reference recording, a sentence at a time, streamed as the codec decodes it | [sakasegawa/Irodori-TTS-v4.1-Small-MF-GGUF](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-MF-GGUF), [sakasegawa/Irodori-TTS-v4.1-Small-GGUF](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-GGUF) |
| nvidia | parakeet-tdt_ctc | 0.6b-ja | `fastconformer` | Japanese speech recognition with its TDT decoder, a recording at a time | [sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF](https://huggingface.co/sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF) |
| nvidia | parakeet-tdt | 0.6b-v3 | `fastconformer` | speech recognition of 25 European languages, which the model tells apart itself, with its TDT decoder | [sakasegawa/parakeet-tdt-0.6b-v3-GGUF](https://huggingface.co/sakasegawa/parakeet-tdt-0.6b-v3-GGUF) |
| reazon-research | reazonspeech-nemo | v2 | `fastconformer` | Japanese speech recognition of recordings of many minutes, with its RNN-T decoder's beam search | [sakasegawa/reazonspeech-nemo-v2-GGUF](https://huggingface.co/sakasegawa/reazonspeech-nemo-v2-GGUF) |
| Qwen | Qwen3-ASR | 0.6B and 1.7B | `qwen3-asr` | speech recognition of 30 languages, found by the model or forced, with a prompt of context, by an encoder and a Qwen3 decoder | [sakasegawa/Qwen3-ASR-0.6B-GGUF](https://huggingface.co/sakasegawa/Qwen3-ASR-0.6B-GGUF), [sakasegawa/Qwen3-ASR-1.7B-GGUF](https://huggingface.co/sakasegawa/Qwen3-ASR-1.7B-GGUF) |

The organization and the model line are the model file's `general.organization` and `general.basename`. The family
is its `general.architecture`, which the model information calls `architecture`: the code that runs the file, which
several model lines can share, as `fastconformer` runs both parakeet lines and ReazonSpeech. Each family's section below
says what it implements.

## Binaries

[Releases](https://github.com/nyosegawa/speech.cpp/releases) carry one archive per platform,
`speech-<version>-<platform>.zip`, for macOS arm64 with Metal (`macos-arm64-metal`), Windows x64 with Vulkan
(`windows-x64-vulkan`) and Linux x64 with Vulkan and with the CPU alone (`linux-x64-vulkan`, `linux-x64-cpu`), and
their SHA-256 sums in `SHA256SUMS`. An archive holds:

| File | What it is |
|---|---|
| `speech`, or `speech.exe` | the one executable, with the library and ggml linked in (The `speech` command line, below) |
| `libspeech.3.dylib` with the link `libspeech.dylib`, `libspeech.so.3` with the link `libspeech.so`, or `speech.dll` with its import library `speech.lib` | the shared library, which exports the C API and nothing else; 3 is the C API's major version |
| `speech.h` | the C API |

CI builds and checks the same archive on every pull request and keeps it as the run's artifact. A release is the tag
`v<version>` of the number in the file `VERSION`, which CI checks before it publishes; the library reports the same
number through `speech_version()`, `speech --version` prints it, and the worker's `ready` message carries it. Versions
follow [Semantic Versioning](https://semver.org): while they are 0.x, a release whose
change a caller must adapt to (the worker protocol, the C API, the GGUF layout, a voice file's form, the command
line's arguments) raises the minor version, and any other release the patch. The Vulkan build needs no particular
driver version; on the first run the GPU driver compiles its shaders, which takes seconds and is cached by the driver
until it is updated. The Metal build compiles its kernels on its first run as well (16 s for an Irodori-TTS worker on
an Apple M5, 1.5 s on the runs after).

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
(`cmake -B build`) runs `speech` on the CPU without `--device`.

The build makes the library twice from the same sources: statically into `build/speech`, so that the executable is
one file with ggml inside, and as the shared library `libspeech` (`libspeech.3.dylib`, `libspeech.so.3`,
`speech.dll`), which exports the functions of `speech.h` and nothing else. It also builds the checks, which
need the weights and the reference dumps to run.

On Metal, the library turns off Metal 4's tensor API for its own ggml: it sets `GGML_METAL_TENSOR_DISABLE` while
ggml first lists its devices, when ggml reads it, and then restores it. ggml uses that API on the M5 and later chips,
and its matrix kernel in ggml v0.25.3 writes past its output when the output has 64 modulo 128 columns; on an M5 this
turned a codec window of 64 frames into noise ([#13](https://github.com/nyosegawa/speech.cpp/issues/13)). Without it Metal comes closer to the CPU, and on
the M5 Irodori-TTS takes a fifth to a third longer to its first audio while Qwen3-TTS keeps its speed.

## The `speech` command line

`speech` is one executable with a subcommand for each program on the library, each of which reaches the models through
the C API alone (docs/adr/0017):

| Subcommand | What it does |
|---|---|
| `speech tts` | speaks text into a WAVE file or to stdout |
| `speech asr` | writes the text of WAVE files |
| `speech voice` | makes an Irodori-TTS voice file from a reference recording |
| `speech info` | prints what a model file says of its model, without loading it |
| `speech devices` | lists the devices a model can run on |
| `speech serve` | serves a model over HTTP with OpenAI's audio API (The HTTP server, below) |
| `speech worker` | serves a model over JSON Lines on stdin and stdout (The worker protocol 2, below) |

```sh
# One sentence to a file, with a Qwen3-TTS speaker
speech tts Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf --voice ono_anna --seed 42 -o out.wav "明日の東京は晴れです。"

# A text file, one sentence per line, into one WAVE file, in the voice of a reference recording
speech tts Irodori-TTS-848M-MF-v4.1-F16.gguf --add-voice bright=bright-young-woman-10s.voice.gguf --voice bright \
    -o story.wav < story.txt

# Straight into a player, which starts as the first audio arrives
echo "こんにちは。" | speech tts Irodori-TTS-848M-MF-v4.1-F16.gguf \
    --add-voice bright=bright-young-woman-10s.voice.gguf --voice bright -o - | ffplay -nodisp -autoexit -

# An Irodori-TTS voice file from a reference recording (Irodori-TTS voices, below)
speech voice Irodori-TTS-848M-MF-v4.1-F16.gguf bright-young-woman-10s.wav bright-young-woman-10s.voice.gguf

# The text of recordings, and their segments with their times as JSON Lines
speech asr parakeet-tdt_ctc-0.6B-ja-F16.gguf meeting.wav
speech asr parakeet-tdt_ctc-0.6B-ja-F16.gguf --timestamps --format json one.wav two.wav > texts.jsonl

# The text of a recording in Japanese, told the names it holds
speech asr Qwen3-ASR-1.7B-Q8_0.gguf --language ja --prompt "Claude Code、渋谷" meeting.wav

# What a model file holds, without loading it
speech info Irodori-TTS-848M-MF-v4.1-F16.gguf
```

A model is one GGUF file, its codec included (GGUF files, below).

One parser reads every command line. Every subcommand that loads a model takes `--device NAME` (`auto`, the default,
for the first GPU or the CPU on a machine without one; `gpu`; `cpu`; or a name `speech devices` lists) and `--threads
N`, the C API's load parameters. `speech tts` and `speech asr` take every request option of the C API's vocabulary
(Options, below) as a flag of its name in kebab-case, read by the option's type: `--voice`, `--language`, `--seed`,
`--speed`, `--seconds`, `--duration-scale`, `--steps`, `--max-seconds`, `--timestamps`, which, a boolean, is a flag
without a value, and `--prompt`. A flag's value follows it or an `=` (`--seed 7`, `--seed=7`), and a number is read
whole: `--steps 4x` is a usage error. The model refuses an option it does not take, as in the C API: Qwen3-TTS answers
`--speed 1.5` with `speech: unsupported (speed): ...`. An argument that begins with `-`, such as a text, follows `--`.
`speech <subcommand> --help` lists a subcommand's flags, and `speech --version` prints the release and the C API's
version (`speech.cpp 0.7.0, C API 3.0`).

| Exit | Meaning |
|---|---|
| 0 | done |
| 1 | a failure, printed on stderr as `speech: <code> (<option>): <message>`: the code is the library's category (`speech_status_name()`, Errors below), the option the input at fault, left out with its parentheses when there is none |
| 2 | a command line that cannot be run, printed with a pointer to `speech <subcommand> --help` |
| 3 | a request stopped at the most the model makes (`model_limit`): for `speech tts` the longest speech, and the WAVE file is complete; for `speech asr` the most tokens of a text, and every file's text is written |

A subcommand's stdout carries its output alone: whatever ggml, a system library or the GPU driver prints to stdout goes
to stderr. On Windows the command line is read as UTF-8, and stdin and stdout are binary.

### `speech tts`

```
speech tts MODEL -o FILE|- [options] [TEXT]
  --voice NAME --language TAG --seed N --speed X --seconds S --duration-scale X --steps N --max-seconds S
  --add-voice NAME=FILE       add a voice from a voice file or a WAVE file before speaking; repeatable
  --device NAME --threads N
  -v                          report the model, and each request's seed and stop reason
```

Speaks TEXT, or without it each non-empty line of stdin as one request, into one 16-bit mono WAVE at the model's rate;
`-o -` writes it to stdout. `--seed` applies to every request, so a line gives the audio of a worker's request with the
same seed; without it each request draws its own, which `-v` reports. `--add-voice` adds a voice to a model that takes
voice files (Irodori-TTS, which has none of its own), and `--voice` chooses the voice to speak with, which a synthesis
model requires. A request that stops at `--max-seconds`, or at the longest the model makes, which Qwen3-TTS's file
gives (655 s), is reported on stderr whatever `-v` says. The model loads without a warm-up, so a GPU compiles its
kernels during the first request.

The WAVE is written as the audio is made, so a player reading stdout starts before the rest is made. Into a regular
file, `> out.wav` included, the RIFF and data sizes are set once the audio is complete. Anywhere else, a pipe or a file
appended to with `>>`, the header cannot be written again in place, so the sizes stay `0xFFFFFFFF`, as ffmpeg writes
them to a pipe, and players and ffmpeg read to the end of the stream. A run that fails removes the WAVE file it was
writing, so a file it leaves is always complete. Text on stdin is UTF-8; a byte order mark and CRLF line endings are
accepted.

It reports on stderr where the time went, which makes it a tool for measuring as well: the load, and for each request
its seconds of audio, the time to its first audio and to its end, and the real-time factor, with the sums when stdin
gave several lines.

### `speech asr`

```
speech asr MODEL [options] AUDIO.wav...
  --language TAG --timestamps --prompt TEXT
  --format text|json          text (the default) or one JSON object per file and line
  --device NAME --threads N -v
```

Recognizes each WAVE file in the order given: 16-, 24- or 32-bit PCM or 32-bit float at any rate, its channels
averaged, which the library resamples to the model's rate (16 kHz for FastConformer and Qwen3-ASR; Audio at another
rate, below). Audio in another format is converted first (`ffmpeg -i in.mp3 out.wav`). `text` writes one line per
file; with `--timestamps` it writes one line per segment instead, `FILE<TAB>START<TAB>END<TAB>TEXT`, the times in
seconds with three decimals. `json` writes `{"file":…,"text":…,"stop":…}` for each file, with `"languages"` where the
model names the languages it heard (Qwen3-ASR) and `"segments"` and `"tokens"` when `--timestamps` is given, in the
form of the worker's `end`. It reports on stderr the load and, for each file, its seconds of audio, the time to its text
and the real-time factor, and a recognition that stopped at the most tokens the model writes (Qwen3-ASR's 4096), after
which it exits with 3 once every file's text is written. A failure names the file, and the lines of the files before it
are already on stdout.

### `speech voice`

```
speech voice MODEL REFERENCE.wav VOICE.gguf [--device NAME] [--threads N] [-v]
```

Makes a voice file from a reference recording at any rate, reading only the codec's encoder from MODEL (Irodori-TTS
voices, below). `--device` defaults to `cpu` here, the one device whose latent is the official encoder's to 99 dB
(docs/adr/0002).

### `speech info`

```
speech info MODEL [--json] [--meta]
```

Prints the model's information without loading it: its name and the rest of its identity (organization, model line,
size label, finetune and version, license, source and weight type), its architecture and layout, task and rate,
languages, voices, each option with its type, default, range or choices and whether it steers, the longest text and the
sizes. `--json` prints the model information as JSON instead (Model information as JSON, below), the object that the
worker's `ready` and the server's `/v1/models` carry. `--meta` adds every metadata entry of the GGUF file: as
`key = value` in text, an array of more than eight items shortened with its length given, and whole as a `meta` object
in JSON.

### `speech devices`

```
speech devices [--json]
```

Lists the devices a model can run on, the CPU and the GPUs without the accelerators ggml runs beside the CPU, with their
kind (`cpu`, `gpu` or `igpu`), description and memory. `--json` prints one object, for a program that chooses a device
before it starts a worker:

```json
{"devices":[{"name":"Vulkan0","description":"NVIDIA GeForce RTX 2080","kind":"gpu","memory_total":8589934592,"memory_free":7516192768}]}
```

## Layout

- `include/speech.h` is the C API, the one way into the library.
- `src/` is the library: `speech.cpp`, `info.cpp` and `request.cpp` implement the C API over one engine per family
  (`src/<family>-engine.cpp`, which declares the options the family takes), `src/families/<family>/` runs one
  architecture of model, whichever weights it is given, and `src/common/` holds what the families share.
- `tools/` holds `speech`, the one executable on the library: its subcommands for the command line (`tools/cli/`), the
  worker (`tools/worker/`) and the HTTP server (`tools/server/`), and what they share (`tools/common/`): the parser of
  the command line, the JSON reader and the request options. Beside them, `worker_smoke.py`,
  `worker_recognition_smoke.py`, `server_smoke.py` and `speech_cli_smoke.py` drive each entry point as a caller does.
- `vendor/cpp-httplib/` holds cpp-httplib's header and license, which `speech serve` uses.
- `checks/` holds a check per ported stage that compares it with the official implementation, and
  `speech-api-check`, which runs the C API through the shared library with a synthesis model and, with
  `transcribe`, with a recognition model in F32 or F16 and the dumps of `reference/fastconformer/` or
  `reference/qwen3-asr/`, whose texts it compares byte for byte. Beside them, `qwen3-asr-timing` and
  `llama-server-timing.py` time Qwen3-ASR and llama.cpp's server on the same audio (Qwen3-ASR, Speed, below).
- `reference/<model>/` pins the official implementation in a uv environment and the checkpoints by revision,
  converts the weights to one GGUF file per model (GGUF files, below) and dumps the tensors the checks compare with;
  `reference/resample/` dumps torchaudio's resampling, which `resample-check` compares the library's with, and
  `reference/unicode/` writes the Unicode tables of `src/common/` and the normalizations of Python 3.10 and of the
  tokenizers library, which `unicode-check` compares the library's with (Unicode normalization, below).

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

## Unicode normalization

The library normalizes text as each model's reference does, with the tables of the version of Unicode the reference
has: Irodori-TTS takes NFKC with those of Unicode 13.0, as its runtime's Python 3.10 does, and Qwen3-TTS and Qwen3-ASR
bring the text they tokenize to NFC with those of Unicode 9.0, as the normalizer of their tokenizers does in the
tokenizers library, whose unicode-normalization-alignments crate has 9.0. Text that is not in NFC, such as Japanese
copied from a macOS file name, whose voiced marks stand apart, gets the tokens of its NFC. The two versions differ only
on characters assigned after 9.0, and one set of tables holds both. `unicode-check` compares the library's NFC and
NFKC in both versions with the tokenizers library's and Python 3.10's on 3626 texts, which hold every code point that
a normalization changes or orders, every mark beside one of each combining class, every pair of a canonical
decomposition with and without a mark between them, every Hangul syllable and jamo, and random sequences of them; all
are equal.

```sh
cd reference/unicode
uv run python gen_unicode.py ../../src/common/unicode-data.inc
uv run python normalization_cases.py out
cd ../..
build/unicode-check reference/unicode/out/normalization-cases.tsv
```

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
if (check(speech_model_load("Irodori-TTS-848M-MF-v4.1-F16.gguf", NULL, &model)) &&
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
if (check(speech_model_load("parakeet-tdt_ctc-0.6B-ja-F16.gguf", NULL, &model)) &&
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
  compiles its kernels before the first request (off by default; `speech worker` and `speech serve` turn it on). A device asked for by `gpu` or by name that is not there or does not start is an error, and no other
  device takes its place.
- **Model information.** `speech_model_info_open()` reads what a model file says of its model from its metadata,
  without its weights and without a device: its name and the rest of its identity, from the GGUF specification's
  general keys (`speech_model_info_organization()`, `_basename()`, `_size_label()`, `_finetune()`, `_version()`,
  `_license()`, `_source()` and `_weight_type()`; GGUF files, below), its architecture and layout, task and sample
  rate, languages, voices, whether it takes voice files and the codec they must carry, the longest text, the options
  it takes with their types, defaults, ranges and choices (Options, below), its sizes, and every metadata entry as
  JSON.
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
  `speech_transcribe()` recognizes the whole audio at once. A request that the checks of the whole request refuse is
  left as it was, to be fixed and run again; one that has started its work runs once. `speech_request_set_progress()` gives
  a callback that hears how far a request has come while it passes no audio: Irodori-TTS's sampler steps and a
  recognition's stages and decoding. `speech_request_cancel()` stops one request, from any thread, at the next audio,
  step or stage, and a request cancelled before it runs returns at once; either returns `SPEECH_CANCELLED`.
- **Results.** `speech_request_result()` gives what a request that returned `SPEECH_OK` or `SPEECH_CANCELLED` did:
  why it stopped (`complete`, `max_seconds`, `model_limit` or `cancelled`; a recognition stops at `model_limit` when it
  reaches the most tokens the model writes, Qwen3-ASR's 4096, with the text written up to it), the seed of a synthesis
  (the request's, or one the library drew from 0 to 2^53 - 1, with which the same request repeats its audio on the same
  device), the samples it passed, and the text of a recognition with, when the request set `timestamps`, its segments
  and tokens with their times in seconds (FastConformer, below), and the languages it heard as BCP 47 tags of the
  model's languages (`speech_result_language_count()`, `speech_result_language()`): the language Qwen3-ASR writes
  before its text or the one the request forced on it, one for each run of parts of the same language where it
  recognizes long audio in parts, and none for a model that writes none, such as FastConformer, for audio without
  speech, for a name that is none of the model's languages and for a cancelled request (Qwen3-ASR, below).
- **Errors.** A function that can fail returns a `speech_status`, a negative one for an error, whose category says
  what kind of failure it is; `speech_status_name()` gives its name, `speech_last_error()` the message and
  `speech_last_error_option()` the input it concerns (an option's name, `text`, `audio`, `device`, `threads`, `name`,
  `path`, or for `speech_voice_make()` the parameter's name), on the same thread. No C++ exception crosses the API,
  and the library checks what it is given before ggml sees it.

| Status | Name | Meaning |
|---|---|---|
| `SPEECH_OK` | `ok` | the call did what it was asked |
| `SPEECH_CANCELLED` | `cancelled` | `speech_request_cancel()` or a callback stopped the request |
| `SPEECH_ERROR_INVALID_ARGUMENT` | `invalid_argument` | the caller's mistake: a NULL pointer, an empty text, a value of the wrong type, a required option left out, a request run again after its work |
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
  `speech_api_version_minor()` for a caller that loads the library at run time, give the API's version, 3.1: the
  major rises when a declaration changes in a way an existing caller notices, and the minor when a function, an
  option or an enum value is added. A program built against major version M and minor version m runs against a
  library of the same major version and a minor version of m or more. The shared library's SOVERSION is the major
  version (`libspeech.3.dylib`, `libspeech.so.3`).

Link `libspeech` (on Windows, define `SPEECH_SHARED` and link `speech.lib`), or, within this CMake project,
the target `speech` (shared) or `speech-static`.

### Options

What a request may ask of a model is one vocabulary of options (`speech_option`), each with a fixed name in
snake_case (`speech_option_name()`, `speech_option_from_name()`) and one type. Every entry point uses the same name: a
member of the worker's messages and of the server's speech request, and a flag in kebab-case on the command line
(`--duration-scale`). Each family declares the options it
takes in one table in its engine (`src/<family>-engine.cpp`); ranges and defaults that the model defines come from its
GGUF file (the keys are named in parentheses). The setters, the model information and its JSON read that table and
nothing else.

| Option | Type | Neutral | Qwen3-TTS | Irodori-TTS | FastConformer | Qwen3-ASR |
|---|---|---|---|---|---|---|
| `voice` | string | none | required; one of the speakers (`speech.voices`); steers | required; one of the voices added since loading; steers | not taken | not taken |
| `language` | string | `auto` | default `auto`; one of `general.languages`; steers | default `auto`; one of `general.languages` (`ja`); checked | default `auto`; one of `general.languages`; checked | default `auto`; one of `general.languages`; steers |
| `seed` | int | none | 0 to 2^53 - 1; drawn when not set | 0 to 2^53 - 1; drawn when not set | not taken | not taken |
| `speed` | float | 1 | not taken | 0.25 to 4 (`irodori-tts.length.min_speed`, `max_speed`), default 1 | not taken | not taken |
| `seconds` | float | none | not taken | 0.5 to 30 (`irodori-tts.length.min_seconds`, `max_seconds`), no default | not taken | not taken |
| `duration_scale` | float | 1 | not taken | above 0, default 1 | not taken | not taken |
| `steps` | int | none | not taken | 1 to 2147483647, default `irodori-tts.sampler.default_steps` (4 for MeanFlow, 40 for RF) | not taken | not taken |
| `max_seconds` | float | none | above 0 to the model's limit, `qwen3-tts.generation.max_frames` frames (8192 × 0.08 s = 655.36 s); no default | not taken | not taken | not taken |
| `timestamps` | bool | false | not taken | not taken | default false | not taken |
| `prompt` | string | `""` | not taken | not taken | not taken | default `""`, any text; steers |

A value at an option's neutral value is accepted by every model; any other value of an option a model does not take
is `unsupported`, and an option marked "none" has no neutral value. A string option's value must be one of its
choices (`voice` compared with case; `language` also takes `auto` and a region or script of a choice, compared
without case); a number outside the range is `out_of_range`. "Checked" means the language is compared with the
model's languages and then not used: Irodori-TTS and the Japanese recognizers have one language, and
parakeet-tdt-0.6b-v3 finds the language of the audio itself. Qwen3-ASR is told a language that steers it, or finds it
itself with `auto`. `prompt` is what a recognition is told of the audio before it hears it, the names and terms it may
hold (Qwen3-ASR, below).

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
  `audio`. Qwen3-ASR pads audio under 0.5 s with zeros and takes any.
- Qwen3-ASR, a `prompt` whose tokens, with those of the longest part of the audio (1200 s at most, 13 tokens a second)
  and the 4096 the model may write, are more than its decoder's 65536 positions: `out_of_range`, option `prompt`.

### Model information as JSON

`speech_model_info_json()` writes one object, its members in this order; a member that does not apply is left out
rather than null. `speech info --json` prints it, the worker's `ready` and `info` carry it as `model`, and the HTTP
server's model object as `speech`, so every program that shows a model shows the same:

```json
{
  "name": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
  "organization": "Qwen",
  "basename": "Qwen3-TTS-12Hz",
  "size_label": "0.6B",
  "finetune": "CustomVoice",
  "license": "Apache-2.0",
  "source": {"repository": "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "revision": "85e237c12c027371202489a0ec509ded67b5e4b5"},
  "weight_type": "Q8_0",
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
  "file_bytes": 1213534464,
  "weight_bytes": 1208175044,
  "device": "MTL0",
  "threads": 0
}
```

- `organization`, `basename`, `size_label`, `finetune`, `version`, `license`, `source` and `weight_type` are the
  model's identity, from the GGUF specification's general keys of its file (GGUF files, below). `finetune` and
  `version` are left out for a model whose name has none; `source` is the repository the file was converted from and
  the revision converted; `weight_type` is the type that holds most of the weights, `F32`, `F16` or `Q8_0`.
- `architecture` is the family: the code that runs the file, which several model lines share, as `fastconformer` runs
  parakeet and ReazonSpeech. `layout` is the version of the family's layout that the file has.
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
- The example is `Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf` loaded on Metal, with two of its nine voices shown.

## The worker protocol 2

`speech worker` is a process that another program starts to speak texts or to recognize speech, as ASIST does, and a
program on the C API like any other. It speaks [JSON Lines](https://jsonlines.org): one JSON object per line on stdin
and on stdout, UTF-8, and nothing else on stdout. Every log goes to stderr, and so does anything ggml, a system library
or the GPU driver prints to stdout; a caller treats a line on stdout that is not a JSON object as a defect of the worker
and fails, rather than skipping it. The worker serves one model, one request or peek at a time, in the order they
become complete; `info` and `count_tokens`, which only read the model's information, are answered as they arrive,
also while a request runs.

```
speech worker MODEL [--add-voice NAME=FILE]... [--device NAME] [--threads N] [--no-warmup]
```

```sh
speech worker Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf
speech worker Irodori-TTS-848M-MF-v4.1-F16.gguf \
    --add-voice bright=bright-young-woman-10s.voice.gguf --add-voice calm=calm-reference.wav
speech worker parakeet-tdt_ctc-0.6B-ja-F16.gguf
```

### Lines

Every line in either direction is one JSON object with a string member `type`. JSON is read whole: nested objects,
arrays, booleans and numbers. A member set to `null` counts as left out. A member that the message's type does not
have is refused, a name that is not an option included.

The values of option members follow the option's type (Options, above): a JSON string; an integer written without a
fraction or an exponent; any JSON number; `true` or `false`.

### Start

```
out {"type":"ready","protocol":2,"version":"0.7.0","model":{...model information...}}
out {"type":"fatal","error":{"code":"model_file","option":null,"message":"..."}}
```

`ready` comes once the model is loaded, warmed up unless `--no-warmup` so that a GPU has compiled its kernels before
the first request, and has the voices of `--add-voice`; its `model` is the model information as JSON (above), the
object `speech info --json` prints, with the device, the threads and the voices added. `fatal` comes instead when the
worker cannot start, and the worker exits with 1, or with 2 for a command line it cannot run. `protocol` is raised when
a caller must change to keep working: a message or member removed, renamed or given another meaning. A member or
message added, an option added to the vocabulary, or a member added to the model information does not raise it.

### Requests

| `type` | Members | Task | Answer |
|---|---|---|---|
| `synthesize` | `id`, `text`, option members | synthesis | `progress` and `chunk`s, then one terminal message |
| `chunk` | `id`, `seq`, `pcm` | recognition | none of its own; it adds audio to request `id` |
| `transcribe` | `id`, `sample_rate`, option members | recognition | `progress`, then one terminal message |
| `peek` | `id`, `sample_rate`, option members | recognition | one `partial`; the request stays open |
| `add_voice` | `id`, `name`, `path` | synthesis models that take voice files | one terminal message |
| `info` | `id` | any | one terminal message, at once |
| `count_tokens` | `id`, `text` | synthesis | one terminal message, at once |
| `cancel` | `id` | any | none of its own |

- `id` is a non-empty string, unique among the requests that have not had their terminal message.
- A recognition request is its `chunk` lines, `seq` 0, 1, 2 … in order, then its `transcribe` line, which makes it
  complete; the chunks of several requests may interleave. `sample_rate` is the rate of the chunks' audio, any rate;
  the library resamples it.
- `pcm`, in both directions, is base64 of 16-bit little-endian mono samples. Audio from the worker is at the model's
  `sample_rate`, each sample `round(clamp(x, −1, 1) × 32767)`; audio to the worker is read as `x / 32768`.
- `path` of `add_voice` is a voice file or a WAVE file (Irodori-TTS voices, below).
- `info` and `count_tokens` are answered as they arrive rather than in turn, so a caller can count the tokens of the
  next text while a synthesis runs. `info` gives the voices added so far: one sent before an `add_voice` has had its
  `end` may not list that voice yet.
- `peek`, for live captions, asks for the text of a recognition request that is still collecting chunks. It takes the
  members `transcribe` takes, and they apply to that peek alone: it recognizes the chunks that came before it as a
  `transcribe` with those members would, answers one `partial`, and leaves the request collecting, so that more chunks
  may follow and its `transcribe` still gets the request's one terminal message. A peek waits its turn as a request
  does, sends no `progress`, and is dropped without an answer when its request ends before its turn. A peek of an id
  that is not collecting chunks (unknown, already answered, or after its `transcribe`) is answered with a `partial`
  that carries an `error` and leaves that request as it was.
- A peek recognizes the whole audio received so far, so its cost grows with the audio: parakeet-tdt_ctc-0.6b-ja in F16
  on Metal took 0.07 s for 6.4 s of audio and 0.11 s for 10.5 s. The parakeet models' memory also grows with the
  square of the audio's length (docs/adr/0013).
- `cancel` of a request that is collecting chunks or waiting ends it with `cancelled` at once. One that is running
  stops at the next point where its work can stop, between two chunks, two of Irodori-TTS's sampler steps or two
  stages of a recognition, and ends with `cancelled`; `add_voice` cannot be stopped once it runs and ends with its
  answer. A cancel of an id that no request in flight has, because its request was answered or never sent, is ignored
  and changes nothing later.
- Once a recognition request has had its `error` or its `cancelled` while it collected chunks, its later `chunk` lines
  are dropped up to and including its `transcribe` line, which frees its id, or until a chunk 0 under its id starts a
  new request; a caller that cancels need not send the `transcribe`.

### Answers

```
out {"type":"chunk","id":"a","seq":0,"pcm":"..."}
out {"type":"progress","id":"r","done":0.42}
out {"type":"partial","id":"r","text":"...","stop":"complete","segments":[...],"tokens":[...]}
out {"type":"partial","id":"x","error":{"code":"invalid_argument","option":"id","message":"..."}}
out {"type":"end","id":"a","seed":1234,"samples":96000,"stop":"complete"}
out {"type":"end","id":"r","text":"...","stop":"complete","segments":[{"start":0.0,"end":2.48,"text":"..."}],"tokens":[{"start":0.0,"end":0.16,"text":"..."}]}
out {"type":"end","id":"q","text":"...","stop":"complete","languages":["ja"]}
out {"type":"end","id":"v"}
out {"type":"end","id":"i","model":{...model information...}}
out {"type":"end","id":"c","tokens":14}
out {"type":"error","id":"a","error":{"code":"out_of_range","option":"speed","message":"..."}}
out {"type":"cancelled","id":"a"}
```

- Every request gets exactly one terminal message, `end`, `error` or `cancelled`, and nothing for its id after it.
  `partial` is never terminal: it has the members of a recognition's `end`, `text` and `stop`, with `languages` and
  with `segments` and `tokens` as `end` has them, or an `error`, which ends the peek and not its request.
- `end` of a synthesis has `seed` (the request's or the one the library drew), `samples` (the number sent in its
  chunks) and `stop` (`complete`, `max_seconds` or `model_limit`). `end` of a recognition has `text` and `stop`
  (`complete`, or `model_limit` when it reached the most tokens the model writes, its text written up to there);
  `languages`, the BCP 47 tags of the languages the model heard in the order of the audio, where the result has any
  (Qwen3-ASR, below), and left out where it has none; and `segments` and `tokens` when the request set `timestamps`,
  in the form the C API gives them (FastConformer, below).
  `end` of `info` has `model`; `end` of `add_voice` has nothing more. `end` of `count_tokens` has `tokens`, the number
  of the model's tokens the text takes as a synthesis counts it against `max_text_tokens`
  (`speech_model_info_text_tokens()`), so that a caller can split a long text before it sends it.
- `error` has `code` (a `speech_status_name()`), `option` (the input it concerns: an option's name, `text`, `audio`,
  `id`, `type`, `seq`, `pcm`, `name`, `path`, `sample_rate` or another member's name, or null) and `message`.
- `progress` comes for a running request that is passing no chunks, when the library reports progress and at least
  1 s has passed since the request's last message; `done` is the library's fraction, 0 to 1. Irodori-TTS reports its
  sampler's steps, and a recognition its stages and its decoding. A caller that takes a silent worker for a hung one
  allows for the longest single step: the encoder of a long recording runs as one (3.7 s for 311 s of audio with
  reazonspeech-nemo-v2 in F16 on an Apple M5).
- A line that names no request it could belong to (not a JSON object, no string `id`, a `cancel` with a member it does
  not have, or a line that starts a request under an `id` already in flight) is answered with an `error` without `id`.
- A request of the other task is `unsupported` with the option `type`: a `synthesize` to a recognition model, and a
  `chunk` or `transcribe` to a synthesis model, whose request this error answers and whose later lines are dropped. A
  member the type does not have is `invalid_argument` with that member as `option`. The values of option members are
  checked by the library's setters when the request runs, so their errors are the C API's.

When stdin closes, the worker answers the requests it has, answers each request still collecting chunks with an
`error`, since no `transcribe` can follow, and exits with 0.

`languages` in the model information lists the languages as BCP 47 tags. A request's `language` must be one of them or
a region or script of one (`ja`, `ja-JP`, `zh-Hant`), or `auto`, which leaves the choice to the model; any other
language is `out_of_range`.

- **Qwen3-TTS** passes audio frame by frame (`incremental` true, 24 kHz). Its voices are the model's speakers, and it
  speaks `de`, `en`, `es`, `fr`, `it`, `ja`, `ko`, `pt`, `ru` and `zh`; the language steers it, going into its prompt.
  Two speakers speak a Chinese dialect, `dylan` (Beijing) and `eric` (Sichuan), when the language is `zh` or left to
  the model, as in the official implementation.
- **Irodori-TTS** makes a sentence at once and passes it as the codec decodes it (`incremental` false, 48 kHz), so a
  request should be one sentence; a text longer than the model's 256 tokens is refused. Its voices are those of
  `--add-voice` and `add_voice`. It speaks `ja`, and the language is only checked.
- **FastConformer** recognizes a request's audio at once. The parakeet models' encoders attend over the whole of it, so
  a request to them should be one utterance: on an Apple M5, the 25.5 s FLEURS utterance takes 0.22 s on Metal, and
  its memory grows with the square of the length. reazonspeech-nemo-v2 attends locally and takes a recording of
  minutes in one request. A cancel takes effect before the encoder starts, once it has run, or between two steps of
  the decoding.
- **Qwen3-ASR** recognizes a request's audio at once, up to 1200 s, and longer audio in parts of up to 1200 s, each
  alone (Qwen3-ASR, below). Its encoder attends within windows of 8 s, so its time grows with the length of the audio
  and with the text it writes. A forced `language` steers it, and a `prompt` tells it the names and terms the audio may
  hold; its `end` gives in `languages` the language it heard, or the forced one. A cancel takes effect between two of
  the encoder's graphs of up to four windows, two blocks of 512 rows of the decoder's prefill or two tokens.

A session with Irodori-TTS and one with a recognizer, the second peeking at its request while it collects chunks:

```
in  {"type":"synthesize","id":"1","text":"明日の東京は晴れです。","voice":"bright","seed":42}
out {"type":"chunk","id":"1","seq":0,"pcm":"..."}
out {"type":"chunk","id":"1","seq":1,"pcm":"..."}
out {"type":"end","id":"1","seed":42,"samples":134400,"stop":"complete"}

in  {"type":"chunk","id":"r","seq":0,"pcm":"..."}
in  {"type":"peek","id":"r","sample_rate":16000}
out {"type":"partial","id":"r","text":"群島や湖では","stop":"complete"}
in  {"type":"chunk","id":"r","seq":1,"pcm":"..."}
in  {"type":"transcribe","id":"r","sample_rate":16000}
out {"type":"end","id":"r","text":"群島や湖では必ずしもヨットは必要ありません。","stop":"complete"}
```

### Irodori-TTS voices

Irodori-TTS has no voices of its own; it speaks in the voice of a reference. A voice is either:

- a reference WAVE file: at most 120 s, 16-, 24- or 32-bit PCM or 32-bit float at any rate, the channels
  averaged and resampled to 48 kHz. `speech_voice_add()`, and so `--add-voice` and the worker's `add_voice`,
  normalizes its loudness and encodes it with the codec, as the official runtime does for every request.
- a voice file, which `speech voice` or `speech_voice_make()` writes from a reference WAVE file, reading only the
  codec's encoder from the model file: the reference's codec latent in a GGUF file of its own (Voice files, below) that
  carries the hash of the codec's tensors. It works with every model file of the same codec, v4.1-Small-MF and
  v4.1-Small in any type, and a model of another codec refuses it. Voice files made before 0.7.0 have no layout and are
  refused; make them again from their WAVE files.

```sh
speech voice Irodori-TTS-848M-MF-v4.1-F16.gguf bright-young-woman-10s.wav bright-young-woman-10s.voice.gguf
```

For the 10.7 s reference bright-young-woman-10s.wav, the voice file is 35 KB against the WAVE file's 1 MB,
and loads in 0.016 s on an Apple M5 (Metal) and 0.025 s on an RTX 2080 (Vulkan), against 0.72 s and 0.43 s
to encode the WAVE file (5.05 s on the M5's CPU). Made on the CPU, `speech voice`'s default device, its latent is the
official encoder's to 99 dB SNR; on Metal it is 40 dB and on Vulkan 33 dB (see Accuracy below). ASIST
carries voice files (docs/adr/0002).

## The HTTP server

`speech serve` serves one model over HTTP with OpenAI's audio API, so that a web app, a Python script or curl can
speak a text or recognize speech without starting the worker. It loads the model as the worker does, listens once the
model is ready, and logs to stderr.

```
speech serve MODEL [--host 127.0.0.1] [--port 8080] [--cors-origin ORIGIN|*]... [--add-voice NAME=FILE]...
                   [--device NAME] [--threads N] [--no-warmup]
```

```sh
speech serve Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf
speech serve Irodori-TTS-848M-MF-v4.1-F16.gguf \
    --add-voice bright=bright-young-woman-10s.voice.gguf --port 8080 --cors-origin http://localhost:5173
speech serve parakeet-tdt_ctc-0.6B-ja-F16.gguf
```

| Option | Meaning |
|---|---|
| `--host ADDRESS` | the address to listen on, 127.0.0.1 unless given; the server has no authentication and no TLS, so a server reachable from other machines belongs behind a proxy that adds them |
| `--port N` | the port, 8080 unless given |
| `--cors-origin ORIGIN` | an origin a web page may call the server from, such as `http://localhost:5173`, repeated for more, or `*` for any. Without it the server sends no CORS headers. Preflight requests are answered, and `X-Sample-Rate`, `X-Speech-Seed` and `X-Speech-Stop` are exposed |
| `--add-voice NAME=FILE`, `--device`, `--threads`, `--no-warmup` | as for the worker (above) |

The endpoints:

- `GET /health` answers `{"status":"ok"}`.
- `GET /v1/models` lists the loaded model as OpenAI's model object (`id` the model's name, `object`, `created`,
  `owned_by` `"speech.cpp"`) with `version`, the release, and `speech`, the model information (Model information as
  JSON, above). `GET /v1/models/{id}` gives it alone, and another id is a 404 (`model_not_found`).
- `POST /v1/audio/speech` speaks a text with a synthesis model, as
  [OpenAI's create speech](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create) does.
- `POST /v1/audio/transcriptions` recognizes the speech in a WAV file with a recognition model, as
  [OpenAI's create transcription](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create)
  does (below).

The endpoint of the other task answers a 404 whose message names the right one.

A speech request is a JSON object:

| Member | Meaning |
|---|---|
| `input` | the text, required |
| `model` | the loaded model's `id` from `/v1/models`, or left out. Any other model is a 404 (`model_not_found`) |
| `response_format` | `wav` (the default) or `pcm`. OpenAI's default is `mp3`, which speech.cpp does not encode; `mp3`, `opus`, `aac` and `flac` are refused |
| `stream_format` | `audio` (the default) or `sse`, which needs `pcm` |
| `voice`, `language`, `seed`, `speed`, `seconds`, `duration_scale`, `steps`, `max_seconds`, `timestamps`, `prompt` | every option of the vocabulary by its name (Options, above), of the option's type, which the model checks: `voice` is one of the model's voices, a Qwen3-TTS speaker or a voice of `--add-voice`, and required; `speed` and `voice` are OpenAI's, the others speech.cpp's own. A request without `seed` gets one drawn from 0 to 2^53 - 1 |

A member speech.cpp does not take, OpenAI's `instructions` among them, is refused rather than ignored; a member set to
`null` counts as left out.

The response:

- `wav` is the whole file, 16-bit mono at the model's rate, sent once the speech is made, with `X-Sample-Rate`,
  `X-Speech-Seed`, the seed it was made with, and `X-Speech-Stop`, why it stopped (`complete`, `max_seconds` or
  `model_limit`).
- `pcm` is raw 16-bit little-endian mono at the model's rate, `Content-Type: audio/pcm`, streamed with chunked
  transfer as the model makes it, with `X-Sample-Rate` and `X-Speech-Seed`; its headers leave before the speech ends,
  so it cannot carry the stop reason. The first bytes leave with the worker's first chunk: on an Apple M5 under heavy
  load from other work, the same requests alternated between the server and the worker gave a median first byte of
  0.106 s against the worker's 0.102 s for Qwen3-TTS 0.6B Q8_0, and 0.64 s against 0.60 s for Irodori-TTS
  v4.1-Small-MF F16, and the same audio, byte for byte, for the same seed.
- `stream_format: "sse"` sends the same PCM as server-sent events, each a `data:` line:
  `{"type":"speech.audio.delta","audio":"<base64 PCM>"}` for each chunk, then
  `{"type":"speech.audio.done","seed":42,"samples":134400,"stop":"complete"}`. OpenAI's done event carries the usage in
  tokens, which speech.cpp does not count, so it gives the seed, the samples and the stop reason instead.

The same request with the seed of `X-Speech-Seed` gives the same audio on the same device.

Errors have OpenAI's shape, `{"error":{"message":...,"type":...,"param":...,"code":...}}`. A failure of the library
becomes an error by its category alone (Errors, above), the same for the same mistake whenever it happens; `param` is
the input at fault, with `text` written as `input` and `audio` as `file`:

| Category | HTTP | `type` | `code` |
|---|---|---|---|
| `invalid_argument` | 400 | `invalid_request_error` | `invalid_value` |
| `unsupported` | 400 | `invalid_request_error` | `unsupported_parameter` |
| `out_of_range` | 400 | `invalid_request_error` | `unsupported_value` |
| `model_file`, `device`, `out_of_memory`, `io`, `internal` | 500 | `server_error` | the category's name |

So an unknown voice is a 400 `unsupported_value` with `param` `voice`, a `speed` on Qwen3-TTS a 400
`unsupported_parameter`, and a text longer than the model takes a 400 `unsupported_value` with `param` `input`. The
server's own refusals keep their codes: `unknown_parameter` for a member it does not have, `invalid_type` for a value of
another type, `missing_required_parameter`, `unsupported_value` for a format it does not give, and 404
`model_not_found`; a body that is not JSON is a 400 without a code. A `pcm` or SSE stream begins once the library has
begun the request's work, so a request it refuses gets its error status; an error after that ends a `pcm` stream
without its last chunk, which the client reads as a broken transfer, and an SSE stream with
`{"type":"error","error":{...}}`.

The model serves one request at a time, in the order they arrive; the others wait. A client that disconnects
while it waits is dropped, and one that disconnects while its request runs stops it at the next chunk (Qwen3-TTS),
sampler step or codec window (Irodori-TTS) or stage (a recognition), as a cancel of the worker does, so the next
request starts then.

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
| `file` | the audio, required: a WAV file, 16-, 24- or 32-bit PCM or 32-bit float at any rate, its channels averaged and resampled by the library to the model's `sample_rate` (16000 for FastConformer and Qwen3-ASR). Any other file is refused with a 400 (`param` `file`) rather than guessed at; convert it first (`ffmpeg -i in.mp3 out.wav`) |
| `model` | the loaded model's `id`, or left out; any other model is a 404 (`model_not_found`) |
| `language` | the option `language`: a BCP 47 tag of one of the model's languages, or `auto` (the default) |
| `prompt` | the option `prompt`: what the model is told of the audio before it hears it, for a model that takes it (Qwen3-ASR); OpenAI's own member, which it describes as text to guide the model's style or continue a previous segment |
| `response_format` | `json` (the default), which answers `{"text":"..."}`; `text`, which answers the text alone as `text/plain`; or `verbose_json`, which answers `{"task":"transcribe","language":…,"duration":…,"text":…,"segments":[{"id":0,"start":…,"end":…,"text":…}]}`, the duration being the file's in seconds and `language` the language the model heard (below). With a model that gives times (FastConformer) it sets the option `timestamps` and carries the segments; with one that gives none (Qwen3-ASR) it carries no segments, which OpenAI's schema does not require. `srt`, `vtt` and `diarized_json` are refused: speech.cpp gives neither subtitles nor speakers |
| `timestamp_granularities[]` | `segment`, with `verbose_json`, OpenAI's default, which a model that gives no times refuses (`unsupported_parameter`, `param` `timestamps`); `word` is refused |

Every answer carries `X-Speech-Stop`, why the recognition ended: `complete`, or `model_limit` when it reached the most
tokens the model writes, with the text written up to there. OpenAI's `verbose_json` requires `language` and describes
it as the language of the input audio, its example being Whisper's `english`; speech.cpp gives the BCP 47 tag of the
language, the form its `language` member takes (`ja`), from the result (The C API, above): the language Qwen3-ASR heard
or the forced one, and for audio over 1200 s whose parts it heard in several, their tags joined with commas in the
order of the audio (`ja,en`), as qwen-asr's `transcribe()` joins their names. Where the result has no language, for
FastConformer, which writes none, and for audio without speech, `language` is left out rather than made up, as are the
members of OpenAI's segment that the recognizers have no value for (`seek`, `tokens`, `temperature`, `avg_logprob`,
`compression_ratio`, `no_speech_prob`) and the usage in tokens or seconds that OpenAI's answers carry. OpenAI's other
members (`temperature`, `stream`, `include[]` and the rest) are refused with a 400 rather than ignored, and so is a
member given twice. Audio the library cannot take is a 400 by its category with `param` `file`: no samples or fewer
than the model needs, or a rate it cannot resample from; a language the model does not recognize is a 400
`unsupported_value` with `param` `language`, and a `prompt` to a model that takes none a 400 `unsupported_parameter`.
The upload may be up to 25 MB, OpenAI's limit; the model recognizes the whole file at once, so a file should be one
utterance for FastConformer's parakeet models, where Qwen3-ASR takes up to 1200 s at once (below). The request waits its
turn like a speech request, and a client that goes away cancels it.

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@utterance.wav -F response_format=text
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@meeting.wav -F response_format=verbose_json
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@meeting.wav -F language=ja -F prompt="Claude Code、渋谷"
```

## Files on Hugging Face

A converted file is named under GGUF's naming convention (ggml's `docs/gguf.md`) from its general keys (GGUF files,
below): `<basename>-<size label>-<finetune>-<version>-<type>.gguf` without the parts the model has none of, as gguf-py's
`naming_convention()` writes it. Where the model's name gives no size, the size label is the parameters of the file's
tensors, counted and rounded as gguf-py counts them. Each Hugging Face repository of converted weights, one per upstream
repository and named after it with `-GGUF`, holds the file of the type released, its codec inside it; the converter
writes the other types as well:

| Upstream repository | Converted repository | Released file | Other types the converter writes |
|---|---|---|---|
| [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | [sakasegawa/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF) | `Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf` | `Qwen3-TTS-12Hz-0.6B-CustomVoice-F16.gguf`, `Qwen3-TTS-12Hz-0.6B-CustomVoice-F32.gguf` |
| [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | [sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF) | `Qwen3-TTS-12Hz-1.7B-CustomVoice-Q8_0.gguf` | `Qwen3-TTS-12Hz-1.7B-CustomVoice-F16.gguf`, `Qwen3-TTS-12Hz-1.7B-CustomVoice-F32.gguf` |
| [Aratako/Irodori-TTS-v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF) | [sakasegawa/Irodori-TTS-v4.1-Small-MF-GGUF](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-MF-GGUF) | `Irodori-TTS-848M-MF-v4.1-F16.gguf` | `Irodori-TTS-848M-MF-v4.1-Q8_0.gguf`, `Irodori-TTS-848M-MF-v4.1-F32.gguf` |
| [Aratako/Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) | [sakasegawa/Irodori-TTS-v4.1-Small-GGUF](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-GGUF) | `Irodori-TTS-841M-v4.1-F16.gguf` | `Irodori-TTS-841M-v4.1-Q8_0.gguf`, `Irodori-TTS-841M-v4.1-F32.gguf` |
| [nvidia/parakeet-tdt_ctc-0.6b-ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) | [sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF](https://huggingface.co/sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF) | `parakeet-tdt_ctc-0.6B-ja-F16.gguf` | `parakeet-tdt_ctc-0.6B-ja-F32.gguf` |
| [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | [sakasegawa/parakeet-tdt-0.6b-v3-GGUF](https://huggingface.co/sakasegawa/parakeet-tdt-0.6b-v3-GGUF) | `parakeet-tdt-0.6B-v3-F16.gguf` | `parakeet-tdt-0.6B-v3-F32.gguf` |
| [reazon-research/reazonspeech-nemo-v2](https://huggingface.co/reazon-research/reazonspeech-nemo-v2) | [sakasegawa/reazonspeech-nemo-v2-GGUF](https://huggingface.co/sakasegawa/reazonspeech-nemo-v2-GGUF) | `reazonspeech-nemo-619M-v2-F16.gguf` | `reazonspeech-nemo-619M-v2-F32.gguf` |
| [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | [sakasegawa/Qwen3-ASR-0.6B-GGUF](https://huggingface.co/sakasegawa/Qwen3-ASR-0.6B-GGUF) | `Qwen3-ASR-0.6B-Q8_0.gguf` | `Qwen3-ASR-0.6B-F16.gguf`, `Qwen3-ASR-0.6B-F32.gguf` |
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | [sakasegawa/Qwen3-ASR-1.7B-GGUF](https://huggingface.co/sakasegawa/Qwen3-ASR-1.7B-GGUF) | `Qwen3-ASR-1.7B-Q8_0.gguf` | `Qwen3-ASR-1.7B-F16.gguf`, `Qwen3-ASR-1.7B-F32.gguf` |

Irodori-TTS's names give Small, a word, where the convention's size label is a number, so their size label is counted:
848M for v4.1-Small-MF, whose DiT has MeanFlow's 7.2M parameters more, and 841M for v4.1-Small, each with its codec.
reazonspeech-nemo-v2's name gives no size, and its label is 619M.

Beside each GGUF file, a repository holds a file of the same name with `.json` added: the output of
`speech info --json` for it, made at release. A program can show a model's identity, voices, languages, options and
sizes, and choose which file to download, without downloading the model file; once it loads the model, the worker's
`ready` carries the same object with the device, the threads and the voices added, and a program that pinned the JSON
compares the two.

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
than holding the longest from the start, so a short sentence does not take the memory of the longest, and the talker
reads its prompt 512 rows at a time, each against the rows before it, so that the memory of a long text grows with its
length rather than its square: a prompt of 5137 rows takes 0.18 GB to compute, where reading it whole took 1.8 GB.
A text of more than 24565 tokens, what the talker's 32768 positions leave beside the longest speech and its prompt,
is refused.

### Models

A synthesis needs one file per model, the codec inside it:
[sakasegawa/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF)
holds `Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf` and
[sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF)
`Qwen3-TTS-12Hz-1.7B-CustomVoice-Q8_0.gguf`. The talker and the separate codec of earlier releases, which this one
refuses, remain in the repositories' history. To convert them yourself from the official checkpoints, which
`reference/qwen3-tts/pins.py` pins by revision:

```sh
cd reference/qwen3-tts
uv run python convert.py 0.6b ../../models --type q8_0    # Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf, 1.2 GB
uv run python convert.py 1.7b ../../models --type q8_0    # Qwen3-TTS-12Hz-1.7B-CustomVoice-Q8_0.gguf, 2.3 GB
```

`--type` also takes `f16` and `f32`, whose files are 4.1 GB for 0.6B and 8.1 GB for 1.7B. The codec's large weights
are float16 in a Q8_0 or F16 file, as the released files have them, and float32 in an F32 file.

### Use

```sh
build/speech tts <model.gguf> --voice ono_anna --language ja -o out.wav "明日の東京は晴れです。"
```

`speech worker <model.gguf>` runs it behind the worker protocol (above), and `speech serve <model.gguf>` behind
OpenAI's speech API.

### Accuracy

`reference/qwen3-tts/dump.py` runs the official implementation with greedy decoding and saves the tensors
of every stage; the check tools compare against them.

| Check | Result |
|---|---|
| Codec decoder, whole utterance, CPU, F32 (`codec-check`) | 114 dB SNR against the official decoder |
| Codec decoder, one frame at a time against whole, CPU | 133 dB SNR |
| Codec decoder on Metal | error at -63 dB of the voice |
| Talker and code predictor, F32, teacher forcing (`talker-check`) | argmax matches on every frame; greedy decode gives the same 54 frames |
| Tokenizer (`tokenizer-check`) | encodes 27 texts, 8 of which NFC changes, and decodes 1191 sequences of ids as the model's `tokenizer.json` does |
| Talker's prompt in blocks of 512 against one block, F32 (`qwen3-decoder-check`) | the same keys, values and logits on the CPU (5137 rows) and on Metal with flash attention (3665 rows); with the two products forced on Metal, 2.3e-3 at most for a cached row and 1.5e-3 for the logits, with the same argmax (3665 rows) |

The tokenizer follows the pre-tokenizer of the `tokenizer.json` that ships with the model. The official
package loads it through transformers 4.57.3 with `fix_mistral_regex=True`, which swaps in Mistral's
pattern; the two differ on Latin words in mixed case, contractions and `/`. Before it splits a text, the tokenizer
brings it to NFC, as the normalizer of `tokenizer.json` does (Unicode normalization, above).
`reference/qwen3-tts/tokenizer_cases.py` writes the cases from the model's own tokenizer:

```sh
cd reference/qwen3-tts
uv run python tokenizer_cases.py <Qwen3-TTS checkpoint dir> out/tokenizer-cases.tsv out/decode-cases.tsv
cd ../..
build/tokenizer-check <model.gguf> reference/qwen3-tts/out/tokenizer-cases.tsv reference/qwen3-tts/out/decode-cases.tsv
```

### Speed

Q8_0 weights, Japanese sentences, after the shaders are compiled:

| Model | Device | First audio | Real-time factor | VRAM |
|---|---|---|---|---|
| 0.6B | Apple M5, Metal | 0.04 s | 0.31 | |
| 1.7B | Apple M5, Metal | 0.07 s | 0.43 | |
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
holds `Irodori-TTS-848M-MF-v4.1-F16.gguf` and
[sakasegawa/Irodori-TTS-v4.1-Small-GGUF](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-GGUF)
`Irodori-TTS-841M-v4.1-F16.gguf`, and their cards list the SHA-256. The model and the separate codec of earlier
releases, which this one refuses, remain in the repositories' history. To convert them yourself from the official
checkpoints and codec, which `reference/irodori-tts/pins.py` pins by revision:

```sh
cd reference/irodori-tts
uv run python convert.py mf ../../models --type f16       # Irodori-TTS-848M-MF-v4.1-F16.gguf, 1.9 GB
uv run python convert.py rf ../../models --type f16       # Irodori-TTS-841M-v4.1-F16.gguf, 1.9 GB
```

`--type` also takes `f32` (3.4 GB) and `q8_0` (1.2 GB). The codec stays float32 in every type, as the released
files have it. Qwen3-ASR 1.7B transcribed the 20 sentences of the speed table below with 2.99% CER in F32 and in
F16, and 3.81% in Q8_0, which garbled one phrase.

### Use

```sh
build/speech tts Irodori-TTS-848M-MF-v4.1-F16.gguf --add-voice bright=bright-young-woman-10s.voice.gguf --voice bright \
    -o out.wav "明日の東京は晴れです。" [--device NAME] [--seed N] [--steps N] [--seconds S | --duration-scale X] [--speed X]
```

### Length and speed

The duration predictor sets the length of a sentence before the DiT makes it. A request may change it as the
official runtime's request does, and the C API, the worker, the server and `speech tts` take the same three options:

- `seconds` fixes the length: the latent has the frames that hold `int(seconds / speed × 48000)` samples, and the
  audio is cut there. The duration predictor does not run. Both `seconds` and `seconds / speed` must lie within 0.5
  to 30 s; the runtime clamps a length outside them with a warning, speech.cpp refuses it.
- `duration_scale` multiplies the predicted frames before they are rounded. It must
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

The 20 sentences of speech-bench's prompts/speak-ja-JP.json through the worker in the voice file above,
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
worker, the server and the command line in [ADR 0011](docs/adr/0011-speech-recognition-is-a-task-of-every-entry-point.md).

### Models

Recognition needs one file per model: `parakeet-tdt_ctc-0.6B-ja-F16.gguf` from
[sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF](https://huggingface.co/sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF),
`parakeet-tdt-0.6B-v3-F16.gguf` from
[sakasegawa/parakeet-tdt-0.6b-v3-GGUF](https://huggingface.co/sakasegawa/parakeet-tdt-0.6b-v3-GGUF) or
`reazonspeech-nemo-619M-v2-F16.gguf` from
[sakasegawa/reazonspeech-nemo-v2-GGUF](https://huggingface.co/sakasegawa/reazonspeech-nemo-v2-GGUF), whose cards
list their SHA-256. The files of earlier releases, which this one refuses, remain in the repositories' history. To
convert them yourself, `reference/fastconformer/` pins NeMo 3.0.0 with PyTorch 2.10.0 and each checkpoint by
revision, size and SHA-256:

```sh
cd reference/fastconformer
uv run python convert.py parakeet-tdt_ctc-0.6b-ja ../../models --type f16   # parakeet-tdt_ctc-0.6B-ja-F16.gguf, 1.2 GB
uv run python convert.py parakeet-tdt-0.6b-v3 ../../models --type f16       # parakeet-tdt-0.6B-v3-F16.gguf, 1.3 GB
uv run python convert.py reazonspeech-nemo-v2 ../../models --type f16       # reazonspeech-nemo-619M-v2-F16.gguf, 1.2 GB
```

`--type f32` writes the same at 2.5 GB. The converter refuses a checkpoint with an option the C++ does not run
(another subsampling, attention or decoding, a prompt, a language tag to strip, a tokenizer piece it cannot write)
rather than write a file that would recognize differently from NeMo. GGUF files converted before 0.7.0 have no
layout and are refused; convert them again. The parakeet weights are NVIDIA's, under CC-BY-4.0, and ReazonSpeech's
are reazon-research's, under the Apache License 2.0.

### Use

```sh
speech asr parakeet-tdt_ctc-0.6B-ja-F16.gguf utterance.wav        # the text on stdout
speech asr parakeet-tdt-0.6B-v3-F16.gguf --timestamps utterance.wav
speech asr reazonspeech-nemo-619M-v2-F16.gguf meeting.wav         # a recording of minutes, whole
speech worker parakeet-tdt_ctc-0.6B-ja-F16.gguf                    # a recognition worker (The worker protocol 2, above)
speech serve parakeet-tdt-0.6B-v3-F16.gguf                         # POST /v1/audio/transcriptions
```

The models recognize 16 kHz mono audio, to which the library resamples audio at another rate. parakeet-tdt_ctc-0.6b-ja
and reazonspeech-nemo-v2 recognize `ja`, and parakeet-tdt-0.6b-v3 `bg`, `cs`, `da`, `de`, `el`, `en`, `es`, `et`,
`fi`, `fr`, `hr`, `hu`, `it`, `lt`, `lv`, `mt`, `nl`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sv` and `uk`, the languages
of its model card. None has an input for a language: parakeet-v3 finds the language of the audio itself, as NeMo's
`transcribe()` runs it, without a prompt. A request's language is therefore only checked against the model's (the
option does not steer) and changes nothing in the text. For a long recording, prefer reazonspeech-nemo-v2: the
parakeet models' memory grows with the square of the length. A request that sets `timestamps` also gets the text's
tokens and segments with their times in seconds, from the frames the decoding emitted them on, and the segments end
where the model's file says a sentence ends (Accuracy, below).

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

`speech asr` on an Apple M5 with F16 weights on Metal, after loading: 0.07 s for the 6.36 s utterance, 0.11 s for
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

## Qwen3-ASR

Speech recognition with Qwen's [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), 0.6B and 1.7B, in 30
languages: a Whisper-style log-mel frontend, an encoder that attends within windows of 8 s, a projector, and a Qwen3
decoder that writes the language it hears and then the text. The reference is transformers 5.18's own implementation
with its windowed encoder, and qwen-asr 0.0.6's code for what transformers leaves out (docs/adr/0018). Implemented:

- the frontend: qwen-asr's normalization of the audio (a peak above 1 brought to 1), an utterance under 0.5 s padded
  with zeros to it, and Whisper's log-mel as transformers' feature extractor computes it, 128 bins of a centred STFT of
  400 points, on the host in double precision,
- the encoder, on ggml: three 3 × 3 convolutions of stride 2 over each chunk of 1 s, the last padded with zeros to a
  whole chunk, a sinusoid of each token's position within its chunk, 18 (0.6B) or 24 (1.7B) layers that attend within
  windows of 104 tokens (8 s), up to four windows a graph on a GPU and one on the CPU, and the projector to the
  decoder's width,
- the prompt as qwen-asr writes it: the checkpoint's chat template with a system turn that holds the request's
  `prompt`, empty without one, the audio's tokens in the user turn, and for a forced `language` the prefill
  `language <Name><asr_text>`, which steers the model; the text on either side of the audio is tokenized with the
  pre-tokenizer of the checkpoint's tokenizer, split first at its added tokens, and the text between them brought to
  NFC as the tokenizers library brings it (Unicode normalization, above),
- the decoder: the Qwen3 stack it shares with Qwen3-TTS's talker, its output tied to the token embeddings, a
  key/value cache in F16 that grows with the request, the prompt read 512 rows at a time, and attention through
  ggml's flash attention on a GPU whose backend computes it and through two matrix products on the CPU, chosen when the
  model loads (docs/adr/0019),
- greedy decoding until an end token or 4096 new tokens, the limit of the model's `generate()`, at which the result
  says `model_limit`,
- the output decoded without its special tokens and parsed as qwen-asr's `parse_asr_output()` parses it: its repetition
  fix, the text after `<asr_text>`, "" for the `language None` of audio without speech, and the language the model
  wrote before `<asr_text>`, which the result gives as its tag (Use, below),
- audio over 1200 s cut as qwen-asr cuts it (Long audio, below).

Not implemented: timestamps, which need Qwen3-ForcedAligner-0.6B, a second model (a request takes `timestamps` only as
false); qwen-asr's streaming.

### Models

Recognition needs one file per model: `Qwen3-ASR-0.6B-Q8_0.gguf` from
[sakasegawa/Qwen3-ASR-0.6B-GGUF](https://huggingface.co/sakasegawa/Qwen3-ASR-0.6B-GGUF) or `Qwen3-ASR-1.7B-Q8_0.gguf`
from [sakasegawa/Qwen3-ASR-1.7B-GGUF](https://huggingface.co/sakasegawa/Qwen3-ASR-1.7B-GGUF).
`reference/qwen3-asr/` pins transformers 5.18.0, qwen-asr 0.0.6 and PyTorch 2.10.0, and each checkpoint by
revision, size and SHA-256, and converts it:

```sh
cd reference/qwen3-asr
uv run python convert.py Qwen3-ASR-0.6B ../../models --type q8_0   # Qwen3-ASR-0.6B-Q8_0.gguf, 0.84 GB
uv run python convert.py Qwen3-ASR-1.7B ../../models --type q8_0   # Qwen3-ASR-1.7B-Q8_0.gguf, 2.18 GB
```

`--type f16` writes 1.57 and 4.08 GB, and `--type f32` 3.14 and 8.16 GB. The token embeddings, which are also the
output matrix, are stored once, where the checkpoint holds them twice. The weights are the Qwen team's, under the
Apache License 2.0.

### Use

```sh
speech asr Qwen3-ASR-1.7B-Q8_0.gguf utterance.wav                       # the language left to the model
speech asr Qwen3-ASR-1.7B-Q8_0.gguf --language ja utterance.wav         # Japanese, forced
speech asr Qwen3-ASR-1.7B-Q8_0.gguf --prompt "Claude Code、CI、渋谷" meeting.wav
speech worker Qwen3-ASR-1.7B-Q8_0.gguf
speech serve Qwen3-ASR-0.6B-Q8_0.gguf                                    # POST /v1/audio/transcriptions, with "prompt"
```

The models recognize 16 kHz mono audio, to which the library resamples audio at another rate, in `ar`, `cs`, `da`,
`de`, `el`, `en`, `es`, `fa`, `fi`, `fil`, `fr`, `hi`, `hu`, `id`, `it`, `ja`, `ko`, `mk`, `ms`, `nl`, `pl`, `pt`,
`ro`, `ru`, `sv`, `th`, `tr`, `vi`, `yue` and `zh`, and 22 Chinese dialects under `zh` or `auto`. Cantonese and
Filipino have no two-letter code, and take the three letters of ISO 639 (`yue`, `fil`). A forced language steers the
model, which then writes the text alone; `auto` lets it write the language it hears first. The result gives that
language (`speech_result_language()`, the worker's and `speech asr --format json`'s `languages`, the server's
`verbose_json` `language`) as the tag of `general.languages` whose name in `qwen3-asr.language_names` it is (GGUF
files, below), or the forced language's tag, as qwen-asr gives the forced language. It gives none for audio without
speech, where the model writes `language None`, or where it writes nothing after a forced language, which qwen-asr
also takes for no language; and none for a name that is not one of the 30, which a warning in the log reports, where
qwen-asr passes the name on. On every dump the model wrote one of the 30 names. The `prompt` is what the model is told
of the audio before it hears it, the names and terms it may hold, which it was trained to use as background and not to
follow as instructions.

### Long audio

The model takes at most 1200 s at once. Longer audio is cut as qwen-asr's `transcribe()` cuts it: at 1200 s from the
last cut, moved to the quietest 0.1 s within 5 s on either side and to its quietest sample, a part shorter than 0.5 s
padded with zeros, each part recognized alone, the texts joined without a separator, and the languages merged as
qwen-asr's `merge_languages()` merges their names: in the order of the audio, one for each run of parts of the same
language, and none for a part without one, so that a recording whose parts the model heard in Japanese and then in
English gives `ja` and `en`. The memory is that of the longest part (Speed, below). A part that reaches the 4096
tokens stops there, and the request says `model_limit` with the text of every part. A recording of over a few minutes
may well reach them: on the first 1203 s of 1338 s of FLEURS ja_jp joined, the 0.6B model repeats three sentences from
the first minute on until it reaches 4096 tokens, as the official implementation does. A caller that wants the whole
text of a long recording cuts it at its pauses into pieces of a few minutes.

### Accuracy

`reference/qwen3-asr/dump.py` runs transformers 5.18's Qwen3-ASR on the CPU in float32 with qwen-asr's prompt and
parse, and saves every stage; the checks compare each stage, given the dump's own inputs, with it:

```sh
cd reference/qwen3-asr
uv run python dump.py Qwen3-ASR-0.6B out
uv run python dump.py Qwen3-ASR-1.7B out
uv run python parse_cases.py out    # outputs with qwen-asr's language and text of each, for qwen3-asr-decoder-check
uv run python split_cases.py out    # synthetic audio with qwen-asr's split of each, for qwen3-asr-split-check
# texts and ids with the checkpoint's own tokenizer, for tokenizer-check; the folder is the one pins.py downloads to
uv run python ../qwen3-tts/tokenizer_cases.py <checkpoint dir> out/tokenizer-cases.tsv out/decode-cases.tsv
cd ../..
build/tokenizer-check <model.gguf> reference/qwen3-asr/out/tokenizer-cases.tsv reference/qwen3-asr/out/decode-cases.tsv
build/qwen3-asr-frontend-check <model.gguf> reference/qwen3-asr/out
build/qwen3-asr-encoder-check <model.gguf> reference/qwen3-asr/out [gpu|cpu|device name]
build/qwen3-asr-decoder-check <model.gguf> reference/qwen3-asr/out [gpu|cpu|device name] [flash|products]
build/qwen3-asr-split-check <model.gguf> reference/qwen3-asr/out [gpu|cpu|device name]
```

The inputs are utterances of FLEURS' test split: ja_jp 12677001980660723842 (6.36 s), 13903496305700695803
(10.50 s) and 2630315561484880103 (25.50 s), en_us 10197164397713068203 (5.76 s) and 2880067776280655708 (23.64 s),
cmn_hans_cn 12933878060487367144 (6.66 s) and 14716260585206763911 (8.64 s), de_de 10009182821551087671 (11.16 s),
and two cuts of the first, 0.9 s from 0.6 s on and its first 0.6 s, near silence; each with four requests: the
language left to the model, forced, and both with a prompt that names the utterance's terms. A 1338.42 s input, the
first 100 utterances of ja_jp joined, is dumped to qwen-asr's split and each part's ids and text with the 0.6B model.
On an Apple M5, for the 0.6B and the 1.7B model:

| Check | CPU, F32 | Metal, F32 or F16 | CPU, Q8_0 | Metal, Q8_0 |
|---|---|---|---|---|
| Tokenizer, 29 texts, 10 of which NFC changes and 2 with added tokens, and 1263 sequences of ids (`tokenizer-check`) | equal | equal | equal | equal |
| Features (`qwen3-asr-frontend-check`) | 123.5 to 143.6 dB SNR | the same (on the host) | the same | the same |
| Projector output from the dump's features (`qwen3-asr-encoder-check`) | 85.6 to 111.0 dB, 94.4 to 112.5 dB | 44.4 to 70.6 dB, 50.2 to 68.6 dB with F32; 49.8 to 66.7 dB, 50.7 to 64.8 dB with F16 | 20.2 to 33.2 dB, 24.3 to 33.0 dB | 20.4 to 38.2 dB, 17.1 to 38.3 dB |
| Prompt ids (`qwen3-asr-decoder-check`) | the dump's, all 80 | the same | the same | the same |
| Logits of the prompt's last four rows from the dump's input | 97.4 to 115.1 dB, 96.1 to 114.8 dB | 51.2 to 68.5 dB, 50.1 to 73.2 dB | 19.3 to 33.9 dB, 18.2 to 32.4 dB | 23.1 to 39.5 dB, 21.8 to 37.0 dB |
| Argmax teacher-forced on the dump's ids (1176 and 1184 steps) | every step | every step | all but 8 and 1 | all but 5 and 1 |
| Greedy ids from the dump's projector output, every later stage ours | the dump's on all 80 requests | the same | the dump's on 37 and 38 of 40 | on 37 and 39 of 40 |
| Text from the audio, every stage ours | the dump's on all 80 | the same | on 37 and 35 of 40 | on 37 and 40 of 40 |
| Decoding of the dump's ids, parse of its raw text, 262 cases of the parse, each with its language | equal | equal | equal | equal |

The prompt's logits come from an F32 cache, which leaves the arithmetic of the weights alone; the recognizer's own
F16 cache puts them 42 to 56 dB from transformers' on the CPU in F32, where every text is the dump's as well. With F16
weights on the CPU, whose dot product sums in half precision, the 0.6B model's logits are 37.3 to 53.4 dB and every
text is the dump's. Where a greedy choice differs from the dump's, the check takes it for the arithmetic's only when
the dump's margin between the two tokens is within our error on their two logits, and so it was each time, at margins
of 0.03 to 1.39: the near-silent input with its language forced, where the model writes another filler; the 25.50 s
input, in a name's middle dot (グレン・クッシング for グレンクッシング), a comma, 熱気道 for 熱挙動 and 安定 for 判定; and
the spaces around a Latin name in the 8.64 s one.

The language the result gives, from the dump's raw text and from the audio with every stage ours, is the one qwen-asr
parsed for every dumped request, the forced requests and the 0.6B model's `language None` for the near-silent input
included (the 1.7B model hears Chinese in it, 嗯。, without a prompt): on all 80 on Metal in F32 and in Q8_0, where it
is the dump's also for the texts that differ, and on the 0.6B model's 40 on the CPU in F32. The 262 cases of
`parse_cases.py` hold what the dumps do not: names in other cases and in the code points outside ASCII that Python's
case mappings turn into ASCII, names that are none of the model's (8, which give none where qwen-asr passes the name
on), `None` in other cases and places, a language line after another line, and the line breaks of `str.splitlines()`.

The Metal column is the decoder's flash attention. With the two products forced (`products`), as a GPU without ggml's
flash attention runs them, the prompt's logits lie 47.9 to 71.3 dB and 48.8 to 66.5 dB from transformers' in F32 and
23.2 to 39.2 dB and 21.9 to 37.0 dB in Q8_0, the teacher-forced and greedy rows are as above, and every text is as
above but one: the 1.7B model in Q8_0 writes は、日本語の語源である。 for the near-silent input with its language forced,
at the step where its greedy decoding from the projector output already takes the other token.

The split is qwen-asr's on five synthetic inputs of 1200 to 3700 s (`split_cases.py`: noise with quiet stretches,
speech-like bursts, silence, a last part of 0.2 s that is padded, and audio of exactly 1200 s, which is not split)
and on the 1338.42 s input, cut at 1202.97 s. There, with the 0.6B model in Q8_0 on Metal, the first part's prompt
and its 4096 ids are the dump's, the model repeating three sentences until the limit, and the second part's prompt is
the dump's and its first 137 ids of 376. With the 0.6B model in F32 on the CPU, every id of both parts, 4096 and 376,
is the dump's, and so is the joined text, as on Metal in F32. Both parts write Japanese, and the result's language is
the dump's, `ja`, on Metal in F32 and Q8_0.

### Speed

On an Apple M5 with Q8_0 weights on Metal, after loading, the language left to the model and, in parentheses, forced:
the median of five rounds (three with the 1.7B model) of `build/qwen3-asr-timing`, alternated with
`checks/llama-server-timing.py`, which asks llama.cpp b11246's `llama-server` (the release ASIST bundles) with
ggml-org's Qwen3-ASR GGUF files in Q8_0 for the same audio as ASIST asks it, a 16-bit WAV with the prompt cache off:

```sh
build/qwen3-asr-timing <model.gguf> gpu 1 <dump folder>...
python3 checks/llama-server-timing.py <llama-server> <model.gguf> <mmproj.gguf> MTL0 1 <dump folder>...
```

| Audio | 0.6B, speech.cpp | 0.6B, llama.cpp | 1.7B, speech.cpp | 1.7B, llama.cpp |
|---|---|---|---|---|
| ja_jp, 6.36 s | 0.19 s (0.17 s) | 0.17 s (0.16 s) | 0.43 s (0.38 s) | 0.47 s (0.44 s) |
| ja_jp, 10.50 s | 0.28 s (0.25 s) | 0.25 s (0.23 s) | 0.64 s (0.59 s) | 0.58 s (0.58 s) |
| de_de, 11.16 s | 0.34 s (0.32 s) | 0.31 s (0.30 s) | 0.78 s (0.74 s) | 0.74 s (0.71 s) |
| en_us, 23.64 s | 0.63 s (0.59 s) | 0.52 s (0.51 s) | 1.41 s (1.36 s) | 1.25 s (1.20 s) |
| ja_jp, 25.50 s | 0.87 s (0.80 s) | 0.78 s (0.75 s) | 1.98 s (1.89 s) | 1.81 s (1.76 s) |

The decoding is as fast as llama.cpp's: 146 to 158 tokens fed back a second with the 0.6B model against llama.cpp's
147 to 157, and 58 to 63 with the 1.7B against 58 to 62, the two within the few percent by which one round differs
from the next. What remains is the encoder and the prompt: of the 25.50 s input, the 0.6B model's encoder takes
0.133 s and the prefill of its 347 rows 0.110 s, where llama.cpp takes 0.169 s for both, and the 1.7B model's 0.163 s
and 0.305 s, where llama.cpp takes 0.295 s. llama.cpp computes their matrix products through Metal 4's tensor API,
which speech.cpp leaves off for a defect of ggml's kernel (docs/adr/0003); without it llama.cpp took 0.250 s and
0.523 s for them. llama.cpp's prompt has no system turn, its log-mel one frame more and its last chunk the tokens of
its padding, so its prompt differs from the official one, and it writes another text than the official on 6 of the
20 requests with the 0.6B model and 4 with the 1.7B: with the 1.7B model 軍港や湖ではカマザタ寿司もヨット for the official
群島や湖では必ずしもヨット on the 6.36 s utterance, and with the 0.6B model 光も for 日陰も on the 10.50 s one.

The 1338.42 s input takes 120 s with the 0.6B model, nearly all of it its first part: an encoder of 5.3 s, a prefill
of 15,654 rows in 15 s, and 4096 tokens in 94 s, 43 a second, as each reads a cache of up to 19,750 positions. The
process's peak memory footprint is 2.03 GB: the memory is that of the longest part.

## GGUF files

Every model is one GGUF file, its codec included, which `reference/<model>/convert.py` writes from the checkpoint
that `reference/<model>/pins.py` pins by revision (docs/adr/0015). The file says the version of its layout in
`speech.layout`, 1 for every family, and in `speech.requires` the first release whose reader takes it, 0.7.0. A
reader takes the layout it knows; a newer one is refused with a message that names `speech.requires`, and a file
without `speech.layout`, converted for a release before 0.7.0, is refused as such. Every key below is required in
its family's layout unless the table says when it is present, and has exactly the type listed: a key missing or of
another type is refused with a message that names it, and so is a string that names a kind other than the ones
listed. The tensors are exactly the ones the keys call for, each of the shape the keys give it and of a type its
converter writes: a tensor missing, one not called for, and one of another shape or type are refused, naming the
tensor and the shape or type expected and found, before any weight is loaded. The types are the ones each
converter's `--type` gives, as its docstring says: the matrices it applies to in Q8_0, F16 or F32 (F16 or F32 for
FastConformer), Qwen3-TTS's codec's large weights and Qwen3-ASR's convolutions in F16 or F32, and the norms, biases,
codebooks and the rest in F32. A width that no key gives, listed with each family's tensors, is taken from one tensor, and every other tensor
of that width is checked against it. A key that sizes the model is refused when it is 0, and so is a value the model
cannot run with: heads that do not divide their width or are of an odd width, which RoPE cannot turn in pairs, an id
outside the vocabulary or table it indexes, or a stride the codec does not take. A key whose meaning the tables
leave empty means what the official configuration's field of the same name means.

The model's identity and languages are in the GGUF specification's own keys (ggml's `docs/gguf.md`, "Standardized
key-value pairs"), under the names it gives them, and the file is named under its naming convention (Files on Hugging
Face, above), so that tools that read GGUF metadata show them. The rest is speech.cpp's own, under `speech.` and the
family's architecture; only speech.cpp runs these files.

### Keys of every model file

| Key | Type | Meaning | Source |
|---|---|---|---|
| `general.architecture` | string | the family, the code that runs the file: `qwen3-tts`, `irodori-tts`, `fastconformer` or `qwen3-asr` | the converter |
| `general.name` | string | the model's name | the pinned repository's name (`Qwen3-TTS-12Hz-0.6B-CustomVoice`, `Irodori-TTS-v4.1-Small-MF`, `parakeet-tdt-0.6b-v3`, `Qwen3-ASR-1.7B`) |
| `general.organization` | string | the organization that publishes the model | the pinned repository's namespace (`Qwen`, `Aratako`, `nvidia`, `reazon-research`) |
| `general.basename` | string | the model line, which the file's name begins with | the repository's name through the converter's table of names (`Qwen3-TTS-12Hz`, `Irodori-TTS`, `parakeet-tdt_ctc`, `parakeet-tdt`, `reazonspeech-nemo`, `Qwen3-ASR`) |
| `general.size_label` | string | the number of parameters with its scale, B or M | the repository's name (`0.6B`, `1.7B`), or, where it gives none, the parameters of the file's tensors as gguf-py's `size_label()` rounds them (`848M`, `841M`, `619M`) |
| `general.finetune` | string | what the model was trained toward beyond its line; present where its name gives it | the repository's name (`CustomVoice`, `MF`, `ja`) |
| `general.version` | string | present where the model's name gives one | the repository's name (`v4.1`, `v3`, `v2`) |
| `general.license` | string | SPDX expression | the model card |
| `general.source.repo_url` | string | the repository converted, `https://huggingface.co/<repository>` | the pin |
| `general.source.url` | string | the revision converted: `<general.source.repo_url>/tree/<revision>`, since the specification has no key of its own for a revision | the pin |
| `general.file_type` | u32 | the type that holds most of the tensors' bytes, as gguf-py's `LlamaFileType` numbers it: 0 (F32), 1 (F16) or 7 (Q8_0); a file whose tensors say otherwise is refused | the converter's `--type` |
| `general.quantization_version` | u32 | the version of ggml's quantized blocks (2); present when the file holds a quantized tensor | gguf-py's `GGML_QUANT_VERSION` |
| `general.languages` | [string] | each language's shortest ISO 639 code, sorted, which requests and the model information give as BCP 47 tags: two letters, or three for a language that has no two-letter code (`yue`, `fil`), where the GGUF specification asks for two letters | Qwen3-TTS: the names of `codec_language_id` through the converter's table of codes, dialects left out; Qwen3-ASR: the tags of transformers' `LANGUAGE_CODE_TO_NAME`; the others: the model card |
| `speech.layout` | u32 | 1, the version of the family's layout | the converter |
| `speech.requires` | string | `0.7.0`, the first release whose reader takes this layout | the converter's table of layouts |
| `speech.task` | string | `synthesis` or `recognition`; must be the family's | the converter |
| `speech.sample_rate` | u32 | the rate of the audio made or recognized | Qwen3-TTS: `speech_tokenizer/config.json` `output_sample_rate`; Irodori-TTS: the DACVAE's `sample_rate`; FastConformer: the featurizer's `sample_rate`; Qwen3-ASR: qwen-asr's `SAMPLE_RATE`, the feature extractor's rate |
| `speech.language_use` | string | `steers` or `checked`; must be what the family does | `steers` for qwen3-tts and qwen3-asr, `checked` for the others |
| `speech.voices` | [string] | the built-in voices' names; present for qwen3-tts alone | `talker_config.spk_id`'s names, sorted |
| `speech.voice_languages` | [string] | each voice's language, aligned with `speech.voices` | the model card's "Native Language" column through the table of codes (Dylan's and Eric's dialects are `zh`) |
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
| `qwen3-tts.language_ids` | [i32] | the codec id of each of `general.languages`, aligned | `talker_config.codec_language_id` |
| `qwen3-tts.speaker_ids` | [i32] | the codec id of each of `speech.voices`, aligned | `talker_config.spk_id` |
| `qwen3-tts.dialect_ids` | [i32] | the codec id of the dialect each voice speaks, or −1, aligned | `talker_config.spk_is_dialect` through `codec_language_id` |
| `qwen3-tts.dialect_language` | string | the language in which, as with `auto`, a voice with a dialect speaks it (`zh`) | the official prompt's `language.lower() in ["chinese", "auto"]` through the table of codes |
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
builds them, and stay in the C++, and so do the widths the official decoder fixes in its code rather than in its
configuration: 3 for `pre_conv`, 1 for a residual unit's second convolution, 7 for every other convolution, and the
fourfold width of a ConvNeXt block. The widths that no key gives are the text embedding's width and rows
(`text_hidden_size` and `text_vocab_size`, from `talker.text_embd`, whose rows must cover `tokenizer.tokens`), the
codebooks' entries (`codebook_size`, from `codec.vq.first.codebook.0`, which must cover the talker's and the code
predictor's codes), the decoder's width (`decoder_dim`, from `codec.dec.in_conv.weight`) and the codec transformer's
feed-forward width (its `intermediate_size`, from `codec.tf.blk.0.ffn_gate`).

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

The widths that no key gives are the feed-forward widths of ModernBERT (from `text.blk.0.ffn_act`), of its projector
(`text.proj.res_up.weight`), of the speaker encoder (`speaker.blk.0.ffn_gate`) and of the DiT (`dit.blk.0.ffn_gate`),
the duration predictor's width (`duration.in_proj.weight`), the rank of the DiT's AdaLN
(`dit.blk.0.attn_ada.shift.down`), and the codec's first, latent and decoder widths (`codec.enc.conv_in.weight`,
`codec.enc.conv_out.weight`, `codec.dec.conv_in.weight`). The codec's strides are even, and each of
`codec.encoder_rates` and `codec.decoder_rates` multiplies to `codec.hop_length`.

Constants that stay in the C++, since they are not the model's: the decoder's windows of 12 and 48 frames, which are
how speech.cpp streams a latent the official runtime decodes whole; the encoder's and decoder's margins of 8 and 10
frames, which follow from the codec's architecture; SentencePiece's penalty for an unknown piece; and the widths
DACVAE fixes in its code rather than in its configuration, 7 for the first and the last convolution and a residual
unit's first, 3 for the encoder's last, and 1 for a residual unit's second and the decoder's input projection.

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

The widths that no key gives are the window's length (from `frontend.window`, at most `frontend.n_fft`), the
subsampling's channels (`sub.conv.0.weight`), the feed-forward width (`blk.0.ff1_up.weight`), and the widths of the
prediction network (`pred.embed.weight`) and the joint (`joint.enc.weight`). The subsampling's convolutions are 3 wide,
NeMo's default, which the C++ pads by 1 on either side; `encoder.d_model` is even, and `decoder.tdt.durations` is
not empty.

### qwen3-asr

From the checkpoint's `config.json` (`thinker_config`, its `audio_config` and `text_config`), `preprocessor_config.json`,
`generation_config.json`, `chat_template.json`, `vocab.json`, `merges.txt` and `tokenizer_config.json`, transformers
5.18's Qwen3-ASR (the feature extractor, the encoder and its language tags) and qwen-asr 0.0.6's `inference/utils.py`
(the limits of the audio, the forced language and the parse of the output), both pinned by the converter's `uv.lock`
(docs/adr/0018).

| Key | Type | Meaning | Source |
|---|---|---|---|
| `qwen3-asr.language_names` | [string] | the name the forced language's prefill writes for each language of `general.languages`, aligned with it (`Cantonese` for `yue`), and the one the model writes when the language is left to it, which the result gives as its tag; ASCII, with no capital but the first letter, the form in which the parse compares a name the model writes | transformers' `LANGUAGE_CODE_TO_NAME`, the names of qwen-asr's `SUPPORTED_LANGUAGES` |
| `qwen3-asr.frontend.n_fft`, `hop_length`, `n_mels` | u32 | | `preprocessor_config.json` `n_fft`, `hop_length`, `feature_size` |
| `qwen3-asr.frontend.log_floor`, `dynamic_range`, `log_offset`, `log_divisor` | f32 | the log10's guard, the range kept below the utterance's maximum, and the shift and scale after it (1e-10, 8, 4, 4) | the feature extractor's code |
| `qwen3-asr.audio.min_samples` | u32 | an utterance shorter is padded with zeros to it (8000, 0.5 s) | qwen-asr's `MIN_ASR_INPUT_SECONDS`, the extractor's `min_length` |
| `qwen3-asr.audio.max_samples` | u32 | longer audio is split (19,200,000, 1200 s) | qwen-asr's `MAX_ASR_INPUT_SECONDS` |
| `qwen3-asr.audio.split_search_samples`, `split_window_samples` | u32 | how far on either side of a cut the split looks for the quietest window, and the window (80,000 and 1,600: 5 s and 0.1 s) | `split_audio_into_chunks()`'s `search_expand_sec` and `min_window_ms` |
| `qwen3-asr.encoder.d_model`, `num_layers`, `num_heads`, `ffn_dim` | u32 | | `audio_config` `d_model`, `encoder_layers`, `encoder_attention_heads`, `encoder_ffn_dim` |
| `qwen3-asr.encoder.chunk_frames` | u32 | the frames of a chunk the convolutions take at once (100) | 2 × `n_window` |
| `qwen3-asr.encoder.window_frames` | u32 | the frames of a window the layers attend within (800) | `n_window_infer` |
| `qwen3-asr.encoder.norm_eps` | f32 | the LayerNorms' epsilon (1e-5) | the encoder's LayerNorms |
| `qwen3-asr.encoder.max_timescale` | f32 | the positions' sinusoids (10000) | `SinusoidsPositionEmbedding` |
| `qwen3-asr.decoder.hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size`, `max_position_embeddings` | u32 | | `text_config` |
| `qwen3-asr.decoder.rms_norm_eps`, `rope_theta` | f32 | | `text_config` |
| `qwen3-asr.prompt.before_context`, `before_audio`, `after_audio` | string | the chat template's text before the context, between it and the audio's tokens, and after them | `chat_template.json` filled as qwen-asr fills it, split at the context and the audio token |
| `qwen3-asr.prompt.audio_token` | string | the added token whose rows the projector's output replaces (`<\|audio_pad\|>`) | the processor's `audio_token` |
| `qwen3-asr.prompt.language_prefix`, `asr_text` | string | the forced language's prefill around the name (`language `, `<asr_text>`), and around the name the model writes in its output, after which its text begins; the prefix in ASCII | qwen-asr's `_LANG_PREFIX` and `_ASR_TEXT_TAG` |
| `qwen3-asr.output.repetition_threshold`, `repetition_max_period` | u32 | the repetition fix of the parse: runs and patterns repeated past the threshold kept once (20, 20) | `detect_and_fix_repetitions()` |
| `qwen3-asr.generation.eos_ids` | [i32] | the tokens that end the decoding | `generation_config.json` `eos_token_id` |
| `qwen3-asr.generation.max_new_tokens` | u32 | the most tokens a recognition writes (4096) | the model's `generate()` and qwen-asr's vLLM backend (docs/adr/0018) |
| `qwen3-asr.tokenizer.tokens`, `merges` | [string] | the BPE tokens in id order, the added ones included, and the merges | `vocab.json`, `tokenizer_config.json`'s `added_tokens_decoder`, `merges.txt` |
| `qwen3-asr.tokenizer.added_ids`, `special_ids` | [i32] | the added tokens, at which a text is split before the BPE, and the special ones among them, which decoding drops | `added_tokens_decoder` |

Tensors, with E = `encoder.num_layers` and D = `decoder.num_hidden_layers`:

- `frontend.window`, `frontend.filterbank`;
- `enc.conv.{1,2,3}.{weight,bias}`, `enc.conv_out.weight`;
- `enc.blk.{0..E-1}.` with `attn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_out`, `ffn_norm`, `ffn_up` and `ffn_down`, each a
  `.weight` and a `.bias`; `enc.norm.{weight,bias}`;
- `proj.{1,2}.{weight,bias}`, the projector;
- `dec.token_embd`, which is also the output matrix, the checkpoint's being tied to it;
- `dec.blk.{0..D-1}.` with `attn_norm`, `ffn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_o`, `attn_q_norm`, `attn_k_norm`,
  `ffn_gate`, `ffn_up` and `ffn_down`; `dec.norm`.

The width no key gives is the convolutions' channels, from `enc.conv.1.weight` (480). The convolutions are 3 × 3 with
a stride of 2 and a padding of 1, which the official module fixes in its code and the converter checks. The parse's
mark of audio without speech, `language none` in any case before `<asr_text>`, is the text of qwen-asr's parse, which
the converter finds in its code, and no key holds it.

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

and one tensor, `latent`, F32 with ne = [`latent_dim`, frames], `latent_dim` being the model's. A voice file of another
codec is refused as the caller's mistake before its latent is checked. Voice files made before 0.7.0 have no
`speech.layout` and are refused; they are made again from their WAVE files.

## License

MIT, see [LICENSE](LICENSE). The model weights are their authors': Qwen3-TTS and Qwen3-ASR are the Qwen team's,
under the Apache License 2.0. Irodori-TTS v4.1-Small and v4.1-Small-MF are Aratako's, under the MIT License with the
ethical restrictions of their model cards (no voice cloning without consent, no deepfakes or
misinformation). Semantic-DACVAE-Japanese-32dim is Aratako's and MIT on its card; it derives from Meta's
facebook/dacvae-watermarked, which is under the Apache License 2.0. That card's text also names the SAM
License, a sentence left from the README of facebookresearch/dacvae, which Meta corrected to Apache-2.0 on
2025-12-19; the repository's LICENSE has been Apache-2.0 from its first commit.
