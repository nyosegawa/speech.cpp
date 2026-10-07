# Releasing

This page says how a release is numbered and what the release workflow, `.github/workflows/build.yml`, does. The
steps of a release, from the model files to checking what was published, are the `release` skill's
([skills/release](../../skills/release/SKILL.md)).

## The number

- The release's number is written in `VERSION` and nowhere else. CMake reads it, `speech_version()` and
  `speech --version` report it, and the worker's `ready` carries it.
- Versions follow [Semantic Versioning](https://semver.org). While they are 0.x, a change a caller must adapt to (the
  worker protocol, the C API, the GGUF layout, a voice file's form, the command line's arguments) raises the minor
  version, and any other release raises the patch.
- `VERSION` is raised by a pull request of its own just before a release.
- Releases and tags are never deleted or moved: callers such as ASIST pin the archives by SHA-256.

## Before the tag

- Converted files go to Hugging Face only with the maintainer's approval, each with its `speech info --json` beside it
  ([gguf.md](../gguf.md#file-names)).
- `python3 tools/models/update_catalog.py` pins the catalog to the repositories' current files, and its diff is
  reviewed like code.

## What the workflow does

On every pull request and every push to main:

1. It builds `speech` and `libspeech` on four platforms (macOS arm64 with Metal, Windows x64 with Vulkan, Linux x64 with
   Vulkan and with the CPU alone) and packs each with `speech.h` into `speech-<VERSION>-<platform>.zip`.
2. It checks each archive as a user unpacks it: `speech` starts, it has the library linked in, the link names the
   versioned library, and `libspeech` exports the C API alone. On Linux it also checks that both files need no glibc
   newer than 2.34 and no shared library but glibc's and, in the Vulkan build, `libvulkan.so.1`
   ([ADR 0010](../adr/0010-linux-releases-run-on-glibc-2.34-in-a-vulkan-and-a-cpu-build.md)).
3. It keeps each archive as an artifact of the run ([build.md](../build.md#ci-artifacts)).
4. It runs `install.sh` on Ubuntu and macOS against the published releases: a first install, an update, a run that finds
   it installed, a build of another archive of the same version replaced, the line added to `PATH` once, and an archive
   whose SHA-256 differs from `SHA256SUMS` refused with nothing installed.

On a tag `v*`, once the builds pass:

5. It refuses a tag that is not `v` followed by `VERSION`.
6. It sums the archives into `SHA256SUMS`.
7. It publishes the GitHub release with the archives and `SHA256SUMS`.

The tag's archives are the ones CI builds and checks on every run, so a change to what they hold changes the packaging
steps of the workflow and [install.md](../install.md#release-archives) together.
