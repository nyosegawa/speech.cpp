# The WebAssembly build runs the C API in a Worker and reads a model from its Blob through WORKERFS

## Context

A page that runs speech.cpp keeps the audio in the browser. The C API blocks until a request ends and loads a model
from a path, while a browser gives a page no file system of its own and a model only as a Blob: a File a person
chooses, or the body of a fetch. A page's memory is one WebAssembly memory of at most 4 GB, which a phone's browser
gives less of, and a page downloads every byte of the module before it runs. The library reads a model file with
`fread()` a piece at a time and copies each tensor into its buffer.

## Decision

- **The library is the C API compiled whole** with Emscripten, every function of `speech.h` exported, as an ES module
  that runs in a Worker. `wasm/speech.mjs` is a thin module over it in the Worker: it loads a model, runs a request
  and turns a status into an error with the library's message and the input at fault. It adds no behavior of its own,
  so what the C API checks and refuses, a page meets the same way.
- **A model is read from its Blob through Emscripten's WORKERFS**, which gives the library a file whose reads come from
  the Blob as the library asks. The library reads it as it reads a file on a disk, and the weights are held once, in the
  module's memory.
- **The families a module holds are chosen when it is built** (`SPEECH_WASM_FAMILIES`), each a line of the table of
  families that the build keeps or leaves out, so that a page of known models can download their code alone; a
  module for any model holds them all
  ([its record](a-webassembly-module-for-any-model-holds-every-family-in-one-file.md)).
- **A module refuses more threads than it can start** at `speech_load_params_set_threads()`, one without threads more
  than one, where ggml would abort the module when the thread it starts does not.
- **A WAVE file is read in JavaScript as the library reads one** (`wasm/wav.mjs`), since the C API takes samples, so
  that a page and `speech` give a model the same samples of the same file.

The alternatives were turned down:

- Emscripten's MEMFS, a file system in the module's memory. The file is copied into it whole before the library
  reads it, and the module holds the model twice while it loads, the file and the weights made from it.
- A function of the C API that loads a model from bytes in memory. The bytes are in the module's memory while the
  weights are made from them, twice as with MEMFS, and the C API would have a second way to load a model for one
  platform.
- The module on the page's own thread. Each call would stop the page until it returned, and WORKERFS reads a Blob with
  `FileReaderSync`, which only a Worker has.
- A JavaScript API of its own, with promises and its own names for options and errors. It would be a second API to
  keep in step with `speech.h` and to document, and a binding of the C API in any other language would differ from it.

## Consequences

A page starts a Worker for the module and talks to it with messages of its own. The Blob of a loaded model must not
change while the model is loaded. A family is in a module only when it is built in, and a model of another family is
refused as a model file of an architecture the build does not run.
