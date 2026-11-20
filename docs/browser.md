# Browser

This page describes speech.cpp's WebAssembly build, which runs the C API inside a web page, so that the audio never
leaves the browser. It holds Silero VAD, which finds where someone speaks in a recording.

## What it is

| File | What it is |
|---|---|
| `libspeech.js`, `libspeech.wasm` | the library: the C API compiled to WebAssembly, an ES module for a Worker |
| `speech.mjs` | the module a Worker imports to load a model and run requests |
| `wav.mjs` | reads a WAVE file into samples as `speech` reads it |
| `index.html`, `page.mjs`, `worker.mjs` | a page that finds the regions of speech in a WAVE file you choose |

It runs on the CPU, on one thread, in Chrome and Edge 95, Firefox 114 and Safari 16.4 and later, on phones too.

## Try the page

Serve the folder over HTTP, since a page opened from a file cannot load modules, and open it:

```sh
cd build-wasm
python3 -m http.server 8000
```

Open <http://localhost:8000/>, choose the model, `silero-vad-309K-v6.2-F32.gguf` (`speech pull silero-vad` fetches it,
or take it from [Hugging Face](https://huggingface.co/sakasegawa/silero-vad-GGUF)), and a WAVE file, and press
**Detect speech**. The page shows each region, the load time, the RTFx and the module's memory. A link can name both
files on the same server: `index.html?model=silero-vad-309K-v6.2-F32.gguf&audio=meeting.wav` runs as it opens.

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
| `await openSpeech()` | loads `libspeech.js` and `libspeech.wasm` from beside `speech.mjs` |
| `speech.version` | the release, as `speech_version()` gives it |
| `speech.memory` | the bytes of the module's memory, which grows as needed and never shrinks |
| `speech.load(blob, { threads })` | loads a model from a Blob or File, as `speech_model_load()` does |
| `model.info` | the model's information, the JSON `speech info --json` prints |
| `model.detect(samples, sampleRate, options)` | the regions of a `Float32Array` of mono samples at any rate, with the request options of [Silero VAD](models/silero-vad.md#options) by name |
| `model.free()` | frees the model |

Every call blocks the Worker until the library returns. A failure throws a `SpeechError` with the C API's `status`
(`invalid_argument`, `model_file`, …), its `message`, and the `option` at fault, or null.

## Limits

- **A Worker.** The library runs in a Worker, not on the page's own thread: its calls block, and it reads a model from
  its Blob with `FileReaderSync`, which a Worker alone has.
- **The model is read from its Blob.** The library reads the file a piece at a time as it loads, and holds the weights
  once, in the module's memory. The Blob must not change while the model is loaded. A model the page fetches is a Blob
  the browser keeps as well.
- **4 GB.** The module's memory grows as a model and its requests need, up to 4 GB, the most a 32-bit WebAssembly
  address reaches; a phone's browser gives a page less. A request copies its audio into the module's memory, beside
  the Worker's `Float32Array`.
- **One thread.** The library computes on one thread and refuses `threads` above 1 as `invalid_argument`.
- **Silero VAD alone.** A model of another family is refused as `model_file`, a model of an architecture this build
  does not run.

## Size and speed

Silero VAD F32 on an Apple M5 (macOS 26.2) in headless Chrome 158, built with Emscripten 7.0.0, 2026-11-21:

| | |
|---|---|
| `libspeech.wasm`, gzipped | 360 KB (1.1 MB as it is) |
| `libspeech.js`, gzipped | 18 KB |
| the module started, from the same machine | 6 ms |
| load time | 9 ms |
| a minute of audio, the first request after loading | 0.12 s, an RTFx of 490 |
| a minute of audio, the later requests | 0.09 s, an RTFx of 670 |
| the module's memory, at most | 28.6 MiB |

The first request is slower because the browser compiles the module's code again, faster, once it runs often. `speech
vad` on the same CPU with one thread takes 0.089 s, an RTFx of 675. The page gives the regions `speech vad` gives, number
for number, in Chrome and in Safari on iOS 26.5.

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
```

It makes `build-wasm/libspeech.js`, `libspeech.wasm` and the files beside them in the table above.
`-DSPEECH_WASM_FAMILIES="silero-vad;qwen3-asr"` builds other families in, by their folders in `src/families/`, or
`all`; each adds to the module, and only Silero VAD is checked in a browser.
