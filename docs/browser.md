# Browser

This page describes speech.cpp's WebAssembly build, which runs the C API inside a web page, so that the audio never
leaves the browser. It holds Silero VAD, which finds where someone speaks in a recording.

## What it is

Each [release](https://github.com/nyosegawa/speech.cpp/releases) has it in `speech-<version>-wasm.zip`, whose SHA-256
is in the release's `SHA256SUMS`, or you [build it](#build-it):

| File | What it is |
|---|---|
| `libspeech.js`, `libspeech.wasm` | the library: the C API compiled to WebAssembly, an ES module for a Worker, on one thread |
| `libspeech-threads.js`, `libspeech-threads.wasm` | the same library with threads, for a page that is cross-origin isolated |
| `speech.mjs` | the module a Worker imports to load a model and run requests |
| `wav.mjs` | reads a WAVE file into samples as `speech` reads it |
| `index.html`, `page.mjs`, `worker.mjs` | a page that finds the regions of speech in a WAVE file you choose |
| `serve.py` | serves the folder on 127.0.0.1 with the headers the build with threads needs |
| `speech.h` | the C API, which the library exports function for function |

It runs on the CPU in Chrome and Edge 95, Firefox 114 and Safari 16.4 and later, on phones too.

## Try the page

Serve the folder over HTTP, since a page opened from a file cannot load modules, and open it:

```sh
unzip speech-*-wasm.zip -d speech-wasm
cd speech-wasm
python3 serve.py
```

Open <http://127.0.0.1:8000/>, choose the model, `silero-vad-309K-v6.2-F32.gguf` (`speech pull silero-vad` fetches it,
or take it from [Hugging Face](https://huggingface.co/sakasegawa/silero-vad-GGUF)), and a WAVE file, and press
**Detect speech**. The page shows each region, the load time, the RTFx and the module's memory. With more than one
thread it runs the build with threads. A link can name the files on the same server:
`index.html?model=silero-vad-309K-v6.2-F32.gguf&audio=meeting.wav&threads=4` runs as it opens.

## Use it in a page

The library runs in a Worker. A module Worker of your page:

```js
import { openSpeech } from './speech.mjs';
import { readWav } from './wav.mjs';

const speech = await openSpeech();
onmessage = async ({ data: { model, audio } }) => {          // a File or Blob of the model, and of a WAVE file
  const m = speech.load(model);
  const { sampleRate, samples } = readWav(await audio.arrayBuffer(), audio.name);
  postMessage(m.detect(samples, sampleRate, { min_silence_duration_ms: 500 }));   // [{ start, end }] in seconds
  m.free();
};
```

and the page:

```js
const worker = new Worker(new URL('./my-worker.mjs', import.meta.url), { type: 'module' });
worker.onmessage = ({ data }) => console.log(data);
worker.postMessage({ model: modelInput.files[0], audio: audioInput.files[0] });
```

| In `speech.mjs` | What it does |
|---|---|
| `await openSpeech()` | loads `libspeech.js` and `libspeech.wasm` from beside `speech.mjs`; `openSpeech({ threads: true })` loads the build with threads |
| `speech.version` | the release, as `speech_version()` gives it |
| `speech.memory` | the bytes of the module's memory, which grows as needed and never shrinks |
| `speech.load(blob, { threads })` | loads a model from a Blob or File, as `speech_model_load()` does |
| `model.info` | the model's information, the JSON `speech info --json` prints |
| `model.detect(samples, sampleRate, options)` | the regions of a `Float32Array` of mono samples at any rate, with the request options of [Silero VAD](models/silero-vad.md#options) by name |
| `model.free()` | frees the model |

Every call blocks the Worker until the library returns. A failure throws a `SpeechError` with the C API's `status`
(`invalid_argument`, `model_file`, `out_of_memory`, …), its `message`, and the `option` at fault, or null.

## One thread or several

| | `libspeech` | `libspeech-threads` |
|---|---|---|
| threads | 1; more is `invalid_argument` | 4 by default, at most `navigator.hardwareConcurrency` |
| the page | any | cross-origin isolated: served with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` |
| the module's memory | up to 4 GB | up to 1 GB |
| at the start | the module | the module and a Worker for each logical processor |

The build with threads shares its memory between Workers, which a browser allows only in a page that is cross-origin
isolated; `openSpeech({ threads: true })` refuses to start in another. A cross-origin isolated page loads a resource of
another origin only when that origin allows it (`Cross-Origin-Resource-Policy`, or CORS), which `serve.py` does not
need for files of its own folder. Its memory is limited to 1 GB because Safari on iOS counts the most a shared memory
may grow to as the memory is made, and refuses it when that is too large.

## Limits

- **A Worker.** The library runs in a Worker, not on the page's own thread: its calls block, and it reads a model from
  its Blob with `FileReaderSync`, which a Worker alone has.
- **The model is read from its Blob.** The library reads the file a piece at a time as it loads, and holds the weights
  once, in the module's memory. The Blob must not change while the model is loaded. A model the page fetches is a Blob
  the browser keeps as well.
- **4 GB, or 1 GB with threads.** The module's memory grows as a model and its requests need, up to 4 GB, the most a
  32-bit WebAssembly address reaches, or 1 GB in the build with threads; a phone's browser may give a page less. A
  request copies its audio into the module's memory, beside the Worker's `Float32Array`, and audio past what the memory
  can hold is refused as `out_of_memory`.
- **Silero VAD alone.** A model of another family is refused as `model_file`, a model of an architecture this build
  does not run.

## Size and speed

Silero VAD F32 on an Apple M5 (macOS 26.2, 10 logical processors) in headless Chrome 158, built with Emscripten
7.0.0, a minute of audio, the medians of three runs, 2026-11-21:

| | `libspeech` | `libspeech-threads`, 2 threads | `libspeech-threads`, 4 threads |
|---|---|---|---|
| `.wasm` and `.js`, gzipped | 360 KB and 18 KB | 340 KB and 22 KB | 340 KB and 22 KB |
| the module started, from the same machine | 6 ms | 17 ms | 19 ms |
| load time | 9 ms | 8 ms | 9 ms |
| the first request after loading | 0.13 s, RTFx 480 | 0.09 s, RTFx 640 | 0.11 s, RTFx 560 |
| the fifth request | 0.09 s, RTFx 670 | 0.06 s, RTFx 1080 | 0.04 s, RTFx 1350 |
| the module's memory, at most | 29.9 MiB | 29.9 MiB | 35.9 MiB |

The first requests are slower because the browser compiles the module's code again, faster, once it runs often. With
all 10 logical processors the first request took 0.6 s, since the library starts its threads anew for each part of
the work. `speech vad` on the same CPU takes 0.091 s with one thread (RTFx 660) and 0.049 s with four (RTFx 1220). The page
gives the regions `speech vad` gives, number for number, in Chrome and in Safari on the iOS 26.5 simulator, with and
without threads.

## Build it

The build needs [Emscripten](https://emscripten.org/) 7.0.0, through emsdk:

```sh
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
git checkout 7.0.0
./emsdk install 7.0.0
./emsdk activate 7.0.0
source ./emsdk_env.sh
cd ../speech.cpp
emcmake cmake -B build-wasm
cmake --build build-wasm -j
emcmake cmake -B build-wasm-threads -DSPEECH_WASM_THREADS=ON
cmake --build build-wasm-threads -j
```

The first makes `build-wasm/libspeech.js`, `libspeech.wasm` and the files beside them in the table above, the second
`build-wasm-threads/libspeech-threads.js` and `libspeech-threads.wasm`; copy these two beside the others for the page to
find them. `-DSPEECH_WASM_FAMILIES="silero-vad;qwen3-asr"` builds other families in, by their folders in
`src/families/`, or `all`; each adds to the module, and only Silero VAD is checked in a browser.
