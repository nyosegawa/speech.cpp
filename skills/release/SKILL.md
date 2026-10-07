---
name: release
description: Release a new version of speech.cpp. Covers the model files on Hugging Face and their cards, the catalog compiled into speech, the pull request that raises VERSION, the tag that builds and publishes the archives, and checking what was published. Use when asked to release, publish, cut or tag a version, bump VERSION, or upload converted model files for a release (リリース、バージョンを上げる、タグ、公開、HF に上げる). Not for an ordinary pull request (pull-request) or for porting a model (add-model).
---

# Releasing speech.cpp

A release is the archives CI builds from a tag, and the model files the catalog in those archives names. The catalog
is compiled into `speech`, so the files on Hugging Face are settled before the version is raised.

## 1. Before the release

- Everything the release carries is on main, its CI green, and checked on Vulkan as well (windows-check).
- The C API's minor version was raised once for this release by the first change that added to it; nothing raises it
  again. A change an existing caller notices raises the major version, which is the shared library's SOVERSION.

## 2. Model files (only when a file changes)

A file changes when its family's layout changes, a type is added, or a checkpoint is converted again.

1. Make the files: `convert.py` writes F32 and `speech quantize F32.gguf OUT --type <type>` the other types
   (docs/gguf.md, Convert a model); `tools/quantize_compare.py` shows that an unchanged file comes out byte for byte.
   Check each with the family's stage checks and `speech-api-check`. Before a type is published for the first time,
   `tools/quantize_releases.py` with the previous release's `speech` shows that each file's `speech.requires` names a
   release that reads it.
2. Write each file's information beside it: `speech info --json <file> > <file>.json`.
3. **With the maintainer's approval**, upload to the model's repository, replacing the file of the same name or
   adding the new one: `hf upload sakasegawa/<Upstream>-GGUF <file>` and `<file>.json`. Old files stay in the history.
4. Update the card: the files, their SHA-256 and size, what reads them (`speech.requires`), and anything the change
   alters in usage. Lower-bit files go up only with their measured accuracy in the card.

## 3. The catalog

`python3 tools/models/update_catalog.py` pins every model at its repository's current revision. Review the diff: each
changed revision must be one you uploaded on purpose. Build, and check that `speech pull` fetches each changed file and
the model loads.

## 4. The version

A pull request that raises `VERSION` to the new number and carries the catalog's update, with every doc that names the
version. Merge it when CI passes (pull-request skill).

## 5. Tag and publish

```sh
git fetch origin && git tag v<version> origin/main && git push origin v<version>
```

The tag runs CI's build on every platform and the release job, which checks the tag against `VERSION`, sums the
archives into `SHA256SUMS` and publishes the GitHub release with them. Watch it with `gh run watch`.

## 6. Check what was published

- `gh release view v<version>` lists the four archives and `SHA256SUMS`.
- The installer installs it: `curl -fsSL …/install.sh | sh` in a temporary `HOME` on macOS, and `install.ps1` on
  Windows with `LOCALAPPDATA` pointed at a temporary folder and `-NoModifyPath`; `speech --version` names the release.
- `speech pull <name>` and one synthesis and one recognition by name work from the installed `speech`.

## 7. After it

Programs that pin speech.cpp take the release in their own repositories: speech-bench's runtime pin, and ASIST's
accepted series and the model pins it reads from `speech models --json`. Note them for their maintainers; do not change
other repositories from this one.
