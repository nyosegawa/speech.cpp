# Files, catalog and measurement

## Hugging Face

- One repository per upstream model, `sakasegawa/<Upstream name>-GGUF`, holding the recommended file and the other
  types, each with `<file>.json`, the output of `speech info --json <file>`.
- The card (README.md) says: that the file runs only in speech.cpp and why other GGUF readers cannot run it; what the
  model is and its languages; how it was checked; the license of the upstream weights, with `base_model` and
  `base_model_relation` in the front matter; usage with `speech`; each file's SHA-256 and size.
- A file of a new layout replaces the old one in the same repository; the old one stays in the history, and programs
  that pin the old revision keep working.
- **Upload only when the maintainer approves**, with `hf upload <repo> <file>` after checking the file's SHA-256
  against the one you checked with.
- Lower-bit files (Q6_K, Q5_K, Q4_K) are published only with their measured accuracy, and never as the recommended
  file.

## The catalog

`tools/models/catalog.json` lists, for each model, its name, repository, recommended type and the languages it is the
one to start with; `python3 tools/models/update_catalog.py` fills in the revision, files, sizes, SHA-256 values, task
and languages from Hugging Face and the `.json` beside each file. Run it after every upload, and review its diff like
code: a release ships the files it was checked with.

## speech-bench

speech-bench (github.com/nyosegawa/speech-bench) measures accuracy and speed across runtimes.

- Add the model to its catalog, pinned by revision and SHA-256, with the other runtimes that run it.
- Recognition: Common Voice 8.0 Japanese (CER and CER with accepted spellings) and FLEURS. Synthesis: 20 sentences with a
  fixed voice and seed; time to first audio (median and p90), RTF, CER of a transcription, voice similarity, gaps and
  start-up time.
- Compare with the official implementation and the other runtimes on the same day, under the same conditions, with
  nothing else running. A release candidate is measured from a local build (`SPEECH_BENCH_SPEECH_CPP`).
- Never publish measurements of private recordings.
