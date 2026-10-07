# Fetched models live in the system's cache folder and are removed only on request

Decided 2026-10-08.

## Context

Models fetched by name (docs/adr/0033) need a folder. A model file is 0.8 to 8 GB, several programs may fetch the same
one at once (a script running `speech asr` while ASIST or `speech serve` starts), and a fetch over a slow line can stop
halfway. Each release's catalog pins its own revisions, so after an upgrade the folder can hold files the new release
names no more, and an older release still installed may use them.

## Decision

- **The folder is the system's cache folder**: `~/Library/Caches/speech.cpp/models` on macOS,
  `$XDG_CACHE_HOME/speech.cpp/models` or `~/.cache/speech.cpp/models` on Linux (the XDG Base Directory specification,
  which ignores a value that is not absolute), and `%LOCALAPPDATA%\speech.cpp\models` on Windows. `SPEECH_MODEL_DIR`
  replaces it and must be an absolute path, so that its meaning does not depend on the current folder.
- **Inside it, a file goes to `<owner>--<name>/<revision>/<file>`**, the repository and the commit it was pinned at,
  so that two releases that pin different revisions never share a path, and a path, once filled, always holds the same
  bytes. A file is put there only by a rename once its size and SHA-256 are the catalog's, so a file there is whole,
  and a name whose file is there loads without the network.
- **Nothing is removed automatically.** `speech models` shows every file in those folders that no entry of this
  release's catalog names as old, with its size, and `speech rm --old` removes them; `speech rm NAME[:TYPE]` removes a
  model's file and what was fetched of it. Only the folders a fetch makes are looked into, so a `SPEECH_MODEL_DIR` that
  also holds other files never offers them for removal. Removing takes the file's lock, so it waits for a fetch of the
  same file.
- **A fetch writes `<file>.part`** and resumes from its end the next time; `<file>.lock` is held while a process
  fetches or removes the file and is removed before it is released.

The alternatives were turned down:

- Removing a release's old files when a newer release first runs. Another release still installed, or a program that
  pinned that release, may be using them, and a download of gigabytes is not to be thrown away without being asked.
- A folder of its own beside the executable or under `~/.speech.cpp`. A cache folder is where each system keeps files
  that can be fetched again, and where its users and their cleaning tools look for them; beside the executable, an
  update of speech.cpp would have to carry the models along.
- Hugging Face's own cache (`~/.cache/huggingface/hub`). Its layout is the hub library's to change, and its blobs and
  symbolic links work differently on Windows without Developer Mode.
- A flat folder of file names. Two revisions of a file of the same name, as Irodori-TTS's would be if its name did not
  change with its size, would overwrite each other.

## Consequences

An upgrade leaves the previous release's files until the user removes them, which `speech models` points out. A file
fetched once serves every program that names it at the same revision. Uninstalling speech.cpp leaves the folder,
which README says how to remove.
