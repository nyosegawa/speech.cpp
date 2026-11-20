# The WebAssembly build has a module with threads for isolated pages, of 1 GB, with a Worker per logical processor

## Context

Threads in WebAssembly are Workers that share the module's memory, a SharedArrayBuffer, which a browser gives only to a
page that is cross-origin isolated: served with `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`, and loading from other origins only what they allow. A page that embeds
content of other origins cannot always be isolated. A Worker starts only once the code that asked for it returns to
the browser, and ggml starts its threads for each graph and waits for them within the same call. A shared memory
declares the most it may grow to when it is made, and Safari on iOS counts that most against the process at once: it
refused a shared memory of 2 GB on iOS 16.2, as others reported, where 1 GB was made.

## Decision

- **Two modules**: `libspeech`, on one thread, for any page, and `libspeech-threads` (`SPEECH_WASM_THREADS`), for a page
  that is cross-origin isolated. `speech.mjs` loads either, and refuses the one with threads in a page that is not
  isolated.
- **The module with threads starts a Worker for each of the browser's logical processors before it runs**, and the
  library refuses more threads than that at `speech_load_params_set_threads()`. A thread past them fails to start
  rather than waits for a Worker.
- **It computes on 4 threads by default**, or on the logical processors where they are fewer. Silero VAD F32 on a minute
  of audio, on an Apple M5 of 10 logical processors in Chrome 158 (2026-11-21): RTFx 560 on the first request and 1350
  on the fifth with 4 threads, and 100 and 700 to 1000 with 10.
- **Its memory grows to 1 GB at most**, and the module without threads to 4 GB.

The alternatives were turned down:

- One module, with threads. A page that cannot be cross-origin isolated could not run speech.cpp at all.
- Starting a Worker when a thread is asked for. ggml waits for its threads inside the call that started them, and the
  Worker would start only after that call returned: the call would never end.
- The machine's logical processors by default, as `navigator.hardwareConcurrency` gives them. A browser does not tell
  performance cores from the others, and with all of them the first request was six times slower than with 4.
- A shared memory of 2 GB or 4 GB, which Safari on iOS refuses as the module starts.

## Consequences

A page that runs the module with threads serves its own files with both headers, and fetches a model from another
origin only where that origin sends CORS or `Cross-Origin-Resource-Policy` headers. A model larger than about 1 GB, with
its requests, runs only in the module without threads.
