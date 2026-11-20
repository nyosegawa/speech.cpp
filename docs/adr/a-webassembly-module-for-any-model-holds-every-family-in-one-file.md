# A WebAssembly module for any model holds every family in one file

## Context

A WebAssembly module holds the families it was built with. A module that pages take without building it, as from a
package, cannot know the model a page will load. Built with Emscripten 7.0.0 (2026-11-21), a module of one family is
360 to 522 KB gzipped and one of every family 936 KB, while the files of the models measured below, but Silero VAD's
1.2 MB, are 40 MB to 2.2 GB. In Chrome 158 on an Apple M5 (2026-11-21), a request of one model each ran at: Silero
VAD RTFx 670 on one thread; Haas-ja-v1.0 Q8_0 (FastConformer) RTFx 3.5 on one thread and 8.9 on four; Qwen3-ASR 0.6B
and 1.7B Q8_0 RTFx 0.8 and 0.5; Irodori-TTS-MF Q4_K, Qwen3-TTS 0.6B Q8_0 and RE-USE F32 RTFx 0.16, 0.08 and 0.04.
Haru-4B, Drama 3 and Nemotron 3.5 Transcribe, whose files pass 4 GB, do not load; of the module with threads' 1 GB,
only Silero VAD, FastConformer and RE-USE fit, and the others are refused as out_of_memory.

## Decision

- **A module for any model holds every family**, one without threads and one with them, so that it refuses a model
  only where the library does: for its memory, and a page for its speed.
- **The release's archive holds the families checked in a browser**, Silero VAD; a family joins it once the browser's
  smoke checks it there.

The alternatives were turned down:

- A module per family, which a page chooses by the file's `general.architecture` before it loads the model. It saves
  0.4 to 0.6 MB of a download that a model's file outweighs seventy times or more, for twenty modules to build, check
  and publish, and a second reader of a GGUF file's header, in JavaScript, besides the library's.

## Consequences

A page that runs one model downloads the code of every family once, which the browser keeps for the next visit.
