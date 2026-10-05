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
| FastConformer | NVIDIA's parakeet-tdt_ctc-0.6b-ja and parakeet-tdt-0.6b-v3 | speech recognition with their TDT decoder, an utterance at a time: Japanese, and 25 European languages the model tells apart itself | [sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF](https://huggingface.co/sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF), [sakasegawa/parakeet-tdt-0.6b-v3-GGUF](https://huggingface.co/sakasegawa/parakeet-tdt-0.6b-v3-GGUF) |

## Binaries

[Releases](https://github.com/nyosegawa/speech.cpp/releases) carry, for macOS arm64 (Metal), Windows x64
(Vulkan) and Linux x64 (Vulkan, and the CPU alone), `speech-worker-<version>-<platform>.zip` with the worker
alone, which is what ASIST bundles, and `speech-cpp-tools-<version>-<platform>.zip` with the command-line
tools `speech-tts` and `speech-asr`, the HTTP server `speech-server` and the shared library with its header
(`libspeech.dylib`, `libspeech.so`, or `speech.dll` with its import library `speech.lib`, and `speech.h`), with
their SHA-256 sums. A release is the tag `v<version>` of the number in the file `VERSION`, which CI checks before it
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
tool is one file with ggml inside, and as the shared library `libspeech` (`libspeech.dylib`, `libspeech.so`,
`speech.dll`), which exports the functions of `speech.h` and nothing else. It also builds the checks, which
need the weights and the reference dumps to run.

On Metal, every tool turns off Metal 4's tensor API before it starts a device. ggml uses that API on the
M5 and later chips, and its matrix kernel in ggml v0.25.3 writes past its output when the output has 64
modulo 128 columns; on an M5 this turned a codec window of 64 frames into noise
([#13](https://github.com/nyosegawa/speech.cpp/issues/13)). Without it Metal comes closer to the CPU, and on
the M5 Irodori-TTS takes a fifth to a third longer to its first audio while Qwen3-TTS keeps its speed.

## Command line

`speech-tts` speaks text with any model speech.cpp runs, through the C API, into a WAVE file (16-bit mono PCM
at the model's rate) or to stdout. It loads the model once and speaks the text given, or, without one, every
line of stdin as a request of its own, blank lines skipped, all into the one WAVE in the order of the lines.
The first line takes the seed and each later line the next one, as the worker's requests do, so a line gives
the same audio as the same request to the worker.

```sh
# One sentence to a file, with a Qwen3-TTS speaker
speech-tts qwen3-tts-0.6b-customvoice-q8_0.gguf qwen3-tts-codec-12hz-f16.gguf \
    --voice-name ono_anna --seed 42 -o out.wav "明日の東京は晴れです。"

# A text file, one sentence per line, into one WAVE file, in the voice of a reference recording
speech-tts irodori-tts-v4.1-small-mf-f16.gguf semantic-dacvae-japanese-32dim-f32.gguf \
    --voice bright=bright-young-woman-10s.voice.gguf -o story.wav < story.txt

# Straight into a player, which starts as the first audio arrives
echo "こんにちは。" | speech-tts irodori-tts-v4.1-small-mf-f16.gguf semantic-dacvae-japanese-32dim-f32.gguf \
    --voice bright=bright-young-woman-10s.voice.gguf -o - | ffplay -nodisp -autoexit -

# An Irodori-TTS voice file from a 48 kHz reference recording (see Irodori-TTS voices below)
speech-tts make-voice irodori-tts-v4.1-small-mf-f16.gguf semantic-dacvae-japanese-32dim-f32.gguf \
    bright-young-woman-10s.wav bright-young-woman-10s.voice.gguf --device cpu
```

The options have the names and meanings of the worker's and of the C API's fields:

| Option | For | Meaning |
|---|---|---|
| `-o FILE` | both | the WAVE file to write, or `-` for stdout. Needed |
| `--device NAME`, `gpu`, `cpu` | both | the device as `speech-tts --devices` lists it, the first GPU (the default) or the CPU |
| `--seed n` | both | the seed of the first text; each later line takes the next one. Without it the seed is random |
| `--voice-name NAME` | both | the voice to speak with: a Qwen3-TTS speaker or the name of a `--voice`. Needed unless the model has a single voice; an error lists the model's voices |
| `--language TAG` | both | a BCP 47 tag of one of the model's languages, or `auto` (the default) |
| `--ctx n` | Qwen3-TTS | the talker's context in positions (2048) |
| `--voice NAME=FILE` | Irodori-TTS | a voice, repeated for more: a reference WAVE file or a voice file. At least one is needed |
| `--steps n` | Irodori-TTS | the sampler's steps: 4 for v4.1-Small-MF and 40 for v4.1-Small unless given |
| `--speed x` | Irodori-TTS | the speaking rate, 0.25 to 4 (1) |
| `--seconds s` | Irodori-TTS | the length of each text's speech, 0.5 to 30 s after `--speed`, instead of the predicted one |
| `--duration-scale x` | Irodori-TTS | the factor of the predicted length (1); not together with `--seconds` |
| `-v` | both | also report the release, the model's voices, languages and steps, and each text's seed |

A text that begins with `-` follows `--`. A model refuses what it cannot do, as in the C API: Qwen3-TTS
answers `--speed`, `--seconds` and `--duration-scale` with an error.

`speech-tts` reports on stderr where the time went, which makes it a tool for measuring as well: the load
(which includes compiling the GPU's kernels), and for each text its seconds of audio, the time to its first
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
at the model's rate (16 kHz for FastConformer), its channels averaged; a file at another rate is refused rather than
resampled, so convert it first (`ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav`).

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
- `src/` is the library: `speech.cpp` implements the C API over one engine per family,
  `src/families/<family>/` runs one architecture of model, whichever weights it is given, and `src/common/`
  holds what the families share.
- `tools/` holds the programs built on the library: the worker (`tools/worker/`), the command-line tools
  `speech-tts` and `speech-asr` (`tools/cli/`) and the HTTP server `speech-server` (`tools/server/`), which
  serves a model over HTTP with OpenAI's audio API.
- `vendor/cpp-httplib/` holds cpp-httplib's header and license, which the server alone uses.
- `checks/` holds a check per ported stage that compares it with the official implementation, and
  `speech-api-check`, which runs the C API through the shared library with a synthesis model and, with
  `transcribe`, with a recognition model and the dumps of `reference/fastconformer/`.
- `reference/<model>/` pins the official implementation in a uv environment, converts its weights to GGUF
  and dumps the tensors the checks compare with.

## The C API

`include/speech.h` declares everything; this is the shape of a program that speaks one sentence:

```c
#include "speech.h"

static int on_audio(const float * samples, size_t n, void * user_data) {
    /* n mono float samples at speech_model_sample_rate(); n is 0 between Irodori-TTS's sampler steps.
       Returning nonzero stops the request. */
    return 0;
}

speech_model_params params = speech_model_default_params();
params.model_path = "irodori-tts-v4.1-small-mf-f16.gguf";
params.codec_path = "semantic-dacvae-japanese-32dim-f32.gguf";
speech_voice_source voice = {"bright", "bright-young-woman-10s.voice.gguf"};
params.voices = &voice;
params.n_voices = 1;

speech_model * model;
if (speech_model_load(&params, &model) != SPEECH_OK) {
    fprintf(stderr, "%s\n", speech_last_error());
    return 1;
}
speech_request request = speech_request_default();
request.text = "明日の東京は晴れです。";
request.voice = "bright";
request.seed = 42;
speech_status status = speech_synthesize(model, &request, on_audio, NULL);
speech_model_free(model);
```

and of one that recognizes the speech in 16 kHz mono samples:

```c
static int on_text(const char * text, void * user_data) {
    /* The text, UTF-8, valid during the call. Returning nonzero stops the request. */
    printf("%s\n", text);
    return 0;
}

speech_model_params params = speech_model_default_params();
params.model_path = "parakeet-tdt_ctc-0.6b-ja-f16.gguf";
speech_model * model;
if (speech_model_load(&params, &model) != SPEECH_OK) {
    fprintf(stderr, "%s\n", speech_last_error());
    return 1;
}
speech_transcription_request request = speech_transcription_request_default();
request.samples = samples;          /* const float *, mono */
request.n_samples = n_samples;
request.sample_rate = 16000;        /* speech_model_sample_rate(model) */
speech_status status = speech_transcribe(model, &request, on_text, NULL);
speech_model_free(model);
```

- **Devices.** `speech_device_count()` and `speech_device_get()` list what the library can run on, with the
  memory of each; `params.device` takes a device's name, `"cpu"`, or `"gpu"`/NULL for the first GPU.
- **Models.** `speech_model_load()` chooses the family from `general.architecture` of the model's GGUF and
  takes the codec of Qwen3-TTS and Irodori-TTS, Qwen3-TTS's context and Irodori-TTS's voices (WAVE or voice
  files) and steps. A family refuses, naming the field, a codec it needs and lacks and any field it does not take
  (a codec, voices, steps or a context other than 2048 for FastConformer, voices for Qwen3-TTS). The
  `speech_model_*` getters describe the loaded model: its name, architecture, task (`speech_model_task()`,
  synthesis or recognition), sample rate, how it streams, its voices and languages, whether the language reaches
  the model, its steps and its backend. A recognition model has no voices or steps and streams
  `SPEECH_STREAMING_NONE`, and its sample rate is the rate its audio must have.
- **Requests.** `speech_synthesize()` speaks a text in a voice, in a language or `auto`, from a seed, and
  passes the audio to the callback as it is made. It returns `SPEECH_OK`, `SPEECH_STOPPED` when the callback
  or `speech_cancel()` stopped it, or `SPEECH_ERROR`. A request starts from `speech_request_default()`, so
  that fields a later version adds keep their defaults. Its `speed`, `seconds` and `duration_scale` set the
  speaking rate and the length where the model can follow them (Irodori-TTS, below); a model that cannot
  refuses them with an error rather than ignoring them, as Qwen3-TTS does with any speed but 1 and any length.
- **Recognition.** `speech_transcribe()` recognizes the speech in a `speech_transcription_request`, started from
  `speech_transcription_request_default()`: mono float samples, their rate and a language (or `auto`), and passes
  the text to the callback, once for FastConformer. The rate must be the model's (`speech_model_sample_rate()`):
  the library does not resample, since a resampler's filter changes what the model hears, and refuses another
  rate with an error that names both. `speech_synthesize()` on a recognition model and `speech_transcribe()` on a
  synthesis model are errors. `speech_cancel()` stops a recognition before FastConformer's encoder starts or
  once it has run (docs/adr/0011).
- **Voice files.** `speech_make_voice()` writes an Irodori-TTS voice file from a reference WAVE file.
- **Errors.** A function that can fail returns `SPEECH_ERROR`, and `speech_last_error()` gives the message
  on the same thread. No C++ exception crosses the API.
- **Ownership.** Every string the library returns is its own: a device's strings live as long as the
  process, a model's until `speech_model_free()`, the error message until the next call on the thread.
  Nothing the caller passes is kept after the call.
- **Threads.** A model serves one request at a time; concurrent `speech_synthesize()` or `speech_transcribe()`
  calls on it wait for each other. `speech_cancel()` and the getters may be called from any thread. Separate models are
  independent.
- **Versions.** `speech_version()` gives the release the library was built from (`"0.4.0"`).
  `SPEECH_API_VERSION` and `speech_api_version()` give the version of the API, raised when a change is one an
  existing caller notices; a function added to the API does not raise it. It is 2 since `speech_request`
  gained `speed`, `seconds` and `duration_scale`: a program built against version 1 passes a smaller
  `speech_request`, so it must be rebuilt and start its requests from `speech_request_default()`. Recognition
  added functions, a struct and an enum value and changed nothing an existing caller uses, so the version
  stayed 2.

Link `libspeech` (on Windows, define `SPEECH_SHARED` and link `speech.lib`), or, within this CMake project,
the target `speech` (shared) or `speech-static`.

## The worker

`speech-worker` is a process that another program starts to speak texts or to recognize speech, as ASIST does, and
a program on the C API like any other. It speaks [JSON Lines](https://jsonlines.org): it reads one JSON message
per line on stdin and answers with one JSON object per line on stdout, and runs the family that
`general.architecture` of the model GGUF names. Its `ready` message says the model's task, `"synthesis"` or
`"recognition"`, and the messages it takes follow from the task (below). Nothing else is written to stdout: every log goes to stderr, and so does anything
ggml, a system library or the GPU driver prints to stdout. A caller treats a line on stdout that is not a JSON
object as a defect of the worker and fails, rather than skipping it.

```sh
speech-worker qwen3-tts-0.6b-customvoice-q8_0.gguf qwen3-tts-codec-12hz-f16.gguf
speech-worker irodori-tts-v4.1-small-mf-f16.gguf semantic-dacvae-japanese-32dim-f32.gguf \
    --voice bright=bright-young-woman-10s.voice.gguf --voice calm=calm-reference.wav
speech-worker parakeet-tdt_ctc-0.6b-ja-f16.gguf
```

A synthesis model is given with its codec; a recognition model has none.

| Option | For | Meaning |
|---|---|---|
| `--device NAME`, `gpu`, `cpu` | all | the device as `--devices` names it (`MTL0`, `Vulkan1`), the first GPU (the default) or the CPU |
| `--seed n` | synthesis | the seed of the first request; each later request takes the next one. Without it the seed is random |
| `--ctx n` | Qwen3-TTS | the talker's context in positions (2048, about 160 s of speech) |
| `--voice NAME=FILE` | Irodori-TTS | a voice, repeated for more: a reference WAVE file or a voice file (below). At least one is needed |
| `--steps n` | Irodori-TTS | the sampler's steps: 4 for v4.1-Small-MF and 40 for v4.1-Small unless given |

`speech-worker --devices` prints the devices it can run on, with their memory, and exits.

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

`speed`, `seconds` and `durationScale` are JSON numbers, the `speed`, `seconds` and `duration_scale` of the C
API's request. `speed` is the speaking rate against the model's own (1); `seconds` fixes the length of the
speech, and 0 or no `seconds` leaves it to the model; `durationScale` scales the length the model predicts
(1). A model that cannot follow one of them answers the request with an `error` instead of ignoring it, and so
does a value that is not a number or lies out of its range. Irodori-TTS takes all three (see Length and speed
below). Qwen3-TTS refuses any `speed` but 1 and any `seconds` or `durationScale`: its official implementation
has no control of either (docs/adr/0007).

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

A request is answered once, by its `text` or by one `error`. The audio must be at the model's `sampleRate`, 16000
for FastConformer; another rate is answered with an error, since the worker does not resample (docs/adr/0011). A
chunk whose `seq` is not the next one, whose `pcm` is not base64 or holds an odd number of bytes, and an `end`
without a whole-number `sampleRate` are answered at once with an `error`, and the request's other lines are then
dropped. Requests are recognized one at a time in the order of their ends. A request to speak sent to a
recognition worker is an error, and so is a chunk or an end sent to a synthesis worker. FastConformer recognizes an
utterance at once, its encoder attending over the whole of it, so a request should be one utterance: on an Apple
M5, the 25.5 s FLEURS utterance takes 0.22 s on Metal and its memory grows with the square of the length. A cancel
takes effect before the encoder starts or once it has run.

### Irodori-TTS voices

Irodori-TTS has no voices of its own; it speaks in the voice of a reference. A voice is either:

- a reference WAVE file: 48 kHz (other rates are refused), at most 120 s, 16-, 24- or 32-bit PCM or 32-bit
  float, the channels averaged. The worker normalizes its loudness and encodes it with the codec when it
  starts, as the official runtime does for every request.
- a voice file, which `speech-tts make-voice` or `speech_make_voice()` writes from a reference WAVE file: the reference's codec
  latent in a GGUF that names the codec it was made with (a voice file of another codec is refused).

```sh
speech-tts make-voice irodori-tts-v4.1-small-mf-f16.gguf semantic-dacvae-japanese-32dim-f32.gguf \
    bright-young-woman-10s.wav bright-young-woman-10s.voice.gguf --device cpu
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
speech-server qwen3-tts-0.6b-customvoice-q8_0.gguf qwen3-tts-codec-12hz-f16.gguf
speech-server irodori-tts-v4.1-small-mf-f16.gguf semantic-dacvae-japanese-32dim-f32.gguf \
    --voice bright=bright-young-woman-10s.voice.gguf --port 8080 --cors-origin http://localhost:5173
speech-server parakeet-tdt_ctc-0.6b-ja-f16.gguf
```

| Option | Meaning |
|---|---|
| `--host ADDRESS` | the address to listen on, 127.0.0.1 unless given; the server has no authentication and no TLS, so a server reachable from other machines belongs behind a proxy that adds them |
| `--port n` | the port, 8080 unless given |
| `--cors-origin ORIGIN` | an origin a web page may call the server from, such as `http://localhost:5173`, repeated for more, or `*` for any. Without it the server sends no CORS headers. Preflight requests are answered |
| `--device`, `--ctx`, `--voice NAME=FILE`, `--steps` | as for the worker (above) |

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
| `seed` | speech.cpp's own: the seed, an integer from 0 to 2^64 - 1. Without it the seed is random |
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
| `file` | the audio, required: a WAV file, 16-, 24- or 32-bit PCM or 32-bit float at the model's `sample_rate` (16000 for FastConformer), its channels averaged. Any other file is refused with a 400 (`param` `file`) rather than guessed at; convert it first (`ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav`). Audio at another rate is a 400 as well, since the library does not resample |
| `model` | the loaded model's `id`, or left out; any other model is a 404 (`model_not_found`) |
| `language` | a BCP 47 tag of one of the model's languages, or `auto` (the default) |
| `response_format` | `json` (the default), which answers `{"text":"..."}`, or `text`, which answers the text alone as `text/plain`. `srt`, `vtt`, `verbose_json` and `diarized_json` are refused: speech.cpp gives neither timestamps nor speakers |

OpenAI's other members (`prompt`, `temperature`, `timestamp_granularities[]`, `stream`, `include[]` and the rest)
are refused with a 400 rather than ignored, and so is a member given twice. OpenAI's json answer also carries the
usage in tokens or seconds, which speech.cpp does not count. Audio the model cannot take (no samples, another rate,
a language it does not recognize) is a 400 with the library's message. The upload may be up to 25 MB, OpenAI's
limit; the model recognizes the whole file at once, so a file should be one utterance (see the worker above). The
request waits its turn like a speech request, and a client that goes away cancels it.

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
refused (docs/adr/0007).

### Models

A synthesis needs one talker and the codec, which
[sakasegawa/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF)
(`qwen3-tts-0.6b-customvoice-q8_0.gguf`) and
[sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF)
(`qwen3-tts-1.7b-customvoice-q8_0.gguf`) each hold with `qwen3-tts-codec-12hz-f16.gguf`. To convert them
yourself from the official checkpoints:

```sh
cd reference/qwen3-tts
uv run python convert.py <Qwen3-TTS-12Hz-1.7B-CustomVoice dir> ../../models/gguf --type q8_0 --codec-type f16
```

### Use

```sh
build/speech-tts <talker.gguf> <codec.gguf> --voice-name ono_anna --language ja -o out.wav "明日の東京は晴れです。"
```

`speech-worker <talker.gguf> <codec.gguf>` runs it behind the worker protocol (above).

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

Not implemented: captions (VoiceDesign), speaker-inversion embeddings, SilentCipher's watermark, and
resampling a reference that is not at 48 kHz. The noise comes from speech.cpp's own
generator, so a seed gives other audio than the same seed in the official runtime.

### Models

A synthesis needs one model and the codec, which
[sakasegawa/Irodori-TTS-v4.1-Small-MF-GGUF](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-MF-GGUF)
(`irodori-tts-v4.1-small-mf-f16.gguf`) and
[sakasegawa/Irodori-TTS-v4.1-Small-GGUF](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-GGUF) (`irodori-tts-v4.1-small-f16.gguf`)
each hold with `semantic-dacvae-japanese-32dim-f32.gguf`; their cards list the SHA-256. To convert them yourself
from the pinned official checkpoints:

```sh
cd reference/irodori-tts
uv run python convert.py mf ../../models --type f16       # irodori-tts-v4.1-small-mf-f16.gguf, 1.5 GB
uv run python convert.py rf ../../models --type f16       # irodori-tts-v4.1-small-f16.gguf, 1.5 GB
uv run python convert_codec.py ../../models --type f32    # semantic-dacvae-japanese-32dim-f32.gguf, 371 MB
```

`--type` also takes `f32` and `q8_0` (0.8 GB). Qwen3-ASR 1.7B transcribed the 20 sentences of the speed
table below with 2.99% CER in F32 and in F16, and 3.81% in Q8_0, which garbled one phrase.

### Use

```sh
build/speech-tts irodori-tts-v4.1-small-mf-f16.gguf semantic-dacvae-japanese-32dim-f32.gguf \
    --voice bright=bright-young-woman-10s.voice.gguf -o out.wav "明日の東京は晴れです。" [--device NAME] [--seed n] \
    [--steps n] [--seconds s | --duration-scale x] [--speed x]
```

### Length and speed

The duration predictor sets the length of a sentence before the DiT makes it. A request may change it as the
official runtime's request does, and the C API, the worker and `speech-tts` take the same three options:

- `seconds` fixes the length: the latent has the frames that hold `int(seconds × 48000)` samples, and the
  audio is cut there. The duration predictor does not run. It must lie within 0.5 to 30 s; the runtime
  clamps a length outside them with a warning, speech.cpp refuses it.
- `duration_scale` (`durationScale` in the worker) multiplies the predicted frames before they are rounded and
  kept within 0.5 to 30 s, as in the runtime. It must be above 0, and it cannot be given with `seconds`,
  which the runtime would let override it without a word.
- `speed`, from 0.25 to 4, divides both: the length is `seconds / speed`, or the prediction scaled by
  `duration_scale / speed`. This is what [Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server)
  does with the `speed` of OpenAI's speech API, in the same range, so a request gets the length it would get
  there.

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
and a decoder that turns the encoder's frames into tokens. Implemented, for
[nvidia/parakeet-tdt_ctc-0.6b-ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) (Japanese) and
[nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (25 European languages, with
punctuation and capitals) with their TDT decoder, the one NeMo's `transcribe()` uses:

- the frontend as NeMo runs it in evaluation (pre-emphasis, a centred STFT, the checkpoint's mel filters, 80 for
  parakeet-ja and 128 for parakeet-v3, the log and the normalization of each mel bin over the utterance), on the
  host in double precision,
- the subsampling, the 24 conformer layers with relative positional attention over the whole utterance, and
  the convolution modules, on ggml, with the options in which the two checkpoints differ (parakeet-v3 does not
  scale the subsampling's output and has no biases in its conformer layers) read from the GGUF file,
- the TDT decoder: the prediction network (an embedding and two LSTM layers) and the joint on ggml, one step
  per emitted token, and NeMo's greedy decoding on the host, with the model's durations (0 to 4 frames) and at
  most 10 tokens on one frame, as configured in the checkpoint,
- the SentencePiece pieces turned into text as NeMo's decoding writes it, with the space before each of the
  vocabulary's punctuation marks removed.

The checkpoint's CTC head is not converted: NeMo decodes with TDT by default, and the two write a different text
on some utterances ([ADR 0012](docs/adr/0012-the-recognizer-decodes-with-the-models-default-decoder.md)). Not
implemented yet: reazon-research's reazonspeech-nemo-v2 (Japanese, with local attention and an RNN-T head). Why recognition goes
through this port is in [ADR 0009](docs/adr/0009-speech-recognition-runs-through-a-fastconformer-port.md), and how
it reaches the C API, the worker, the server and `speech-asr` in
[ADR 0011](docs/adr/0011-speech-recognition-is-a-task-of-every-entry-point.md).

### Models

Recognition needs one file, `parakeet-tdt_ctc-0.6b-ja-f16.gguf` from
[sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF](https://huggingface.co/sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF) or
`parakeet-tdt-0.6b-v3-f16.gguf` from
[sakasegawa/parakeet-tdt-0.6b-v3-GGUF](https://huggingface.co/sakasegawa/parakeet-tdt-0.6b-v3-GGUF), whose cards list their SHA-256. To
convert them yourself, `reference/fastconformer/` pins NeMo 3.0.0 with PyTorch 2.10.0 and each checkpoint by
revision, size and SHA-256:

```sh
cd reference/fastconformer
uv run python convert.py parakeet-tdt_ctc-0.6b-ja ../../models --type f16   # parakeet-tdt_ctc-0.6b-ja-f16.gguf, 1.2 GB
uv run python convert.py parakeet-tdt-0.6b-v3 ../../models --type f16       # parakeet-tdt-0.6b-v3-f16.gguf, 1.3 GB
```

`--type f32` writes the same at 2.5 GB. The converter refuses a checkpoint with an option the C++ does not run
(another subsampling or attention, a prompt, a language tag to strip, a tokenizer piece it cannot write) rather
than write a file that would recognize differently from NeMo. GGUF files converted for speech.cpp 0.5.0, which
carry the CTC head instead of the TDT decoder, and for 0.6.0, which lack `fastconformer.use_bias`, are refused;
convert them again. The weights are NVIDIA's, under CC-BY-4.0.

### Use

```sh
speech-asr parakeet-tdt_ctc-0.6b-ja-f16.gguf utterance.wav     # the text on stdout
speech-asr parakeet-tdt-0.6b-v3-f16.gguf utterance.wav
speech-worker parakeet-tdt_ctc-0.6b-ja-f16.gguf                 # a recognition worker (The worker, above)
speech-server parakeet-tdt-0.6b-v3-f16.gguf                     # POST /v1/audio/transcriptions
```

Both models take 16 kHz mono audio. parakeet-tdt_ctc-0.6b-ja recognizes `ja`, and parakeet-tdt-0.6b-v3 `bg`,
`cs`, `da`, `de`, `el`, `en`, `es`, `et`, `fi`, `fr`, `hr`, `hu`, `it`, `lt`, `lv`, `mt`, `nl`, `pl`, `pt`, `ro`,
`ru`, `sk`, `sl`, `sv` and `uk`, the languages of its model card. Neither has an input for a language: parakeet-v3
finds the language of the audio itself, as NeMo's `transcribe()` runs it, without a prompt. A request's language is
therefore only checked against the model's (`languageSelectable` false) and changes nothing in the text.

### Accuracy

`reference/fastconformer/dump.py` runs the official model on the CPU in float32 and saves every stage; the
checks compare each stage, given the dump's own inputs, with it:

```sh
cd reference/fastconformer
uv run python dump.py parakeet-tdt_ctc-0.6b-ja out <16 kHz mono WAVE files>
uv run python dump.py parakeet-tdt-0.6b-v3 out <16 kHz mono WAVE files>
cd ../..
build/fastconformer-frontend-check <model.gguf> reference/fastconformer/out
build/fastconformer-encoder-check <model.gguf> reference/fastconformer/out [gpu|cpu|device name]
build/fastconformer-tdt-check <model.gguf> reference/fastconformer/out [gpu|cpu|device name]
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
| Prediction network on the dump's labels (`fastconformer-tdt-check`) | 130 to 133 dB | 57 to 60 dB | 132 to 134 dB with F32, 66 to 70 dB with F16 |
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

## License

MIT, see [LICENSE](LICENSE). The model weights are their authors': Qwen3-TTS is the Qwen team's, under the
Apache License 2.0. Irodori-TTS v4.1-Small and v4.1-Small-MF are Aratako's, under the MIT License with the
ethical restrictions of their model cards (no voice cloning without consent, no deepfakes or
misinformation). Semantic-DACVAE-Japanese-32dim is Aratako's and MIT on its card; it derives from Meta's
facebook/dacvae-watermarked, which is under the Apache License 2.0. That card's text also names the SAM
License, a sentence left from the README of facebookresearch/dacvae, which Meta corrected to Apache-2.0 on
2025-12-19; the repository's LICENSE has been Apache-2.0 from its first commit.
