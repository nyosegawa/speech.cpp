# Build from source

This page builds speech.cpp from source, for a machine without a release or to change the code.

## Build

```sh
git clone --recurse-submodules https://github.com/nyosegawa/speech.cpp.git
cd speech.cpp
cmake -B build                  # Metal on macOS, the CPU elsewhere
cmake --build build --config Release -j
```

The build makes the library twice from the same sources:

| Output | What it is |
|---|---|
| `build/speech` | the command, one file with the library and ggml linked in statically |
| `libspeech.3.dylib`, `libspeech.so.3` or `speech.dll` | the shared library, which exports the functions of `speech.h` and nothing else |
| `build/*-check` | the checks, which need weights and reference dumps to run ([development/checks.md](development/checks.md)) |

Inside this CMake project, a program links the target `speech` (the shared library) or `speech-static`.

## Backends

| Backend | Configure with | Notes |
|---|---|---|
| Metal | `cmake -B build` on macOS | checked and released |
| CPU | `cmake -B build` elsewhere | checked and released for Linux; runs `speech` without `--device` on a machine without a GPU |
| Vulkan | `cmake -B build -DGGML_VULKAN=ON` | needs the Vulkan SDK to build; checked on NVIDIA and released for Windows and Linux |
| CUDA | `cmake -B build -DGGML_CUDA=ON` | builds, but is not checked or released |

## Linux

The build needs CMake 3.20 or later and GCC or Clang with C++17. It is checked with Ubuntu 22.04's CMake 3.22 and GCC
11.4. For Vulkan, the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#linux) brings the loader, the headers,
SPIRV-Headers and `glslc` in one archive that needs no installation. CI uses 1.4.357.0:

```sh
curl -LO https://sdk.lunarg.com/sdk/download/1.4.357.0/linux/vulkansdk-linux-x86_64-1.4.357.0.tar.xz
tar -xJf vulkansdk-linux-x86_64-1.4.357.0.tar.xz
source 1.4.357.0/setup-env.sh
cmake -B build -DGGML_VULKAN=ON
cmake --build build -j
```

## A build from source and a release

A build from source is tuned to the CPU it is built on and uses OpenMP, so it needs `libgomp.so.1`. The releases are
built for any CPU with AVX2 (`-DGGML_NATIVE=OFF`) and with ggml's own threads (`-DGGML_OPENMP=OFF`), as
`.github/workflows/build.yml` shows.

## CI artifacts

CI builds and checks the release archives on every pull request and every push to main, and keeps each archive as an
artifact of the run, named after its platform (`speech-macos-arm64-metal`, `speech-windows-x64-vulkan`,
`speech-linux-x64-vulkan`, `speech-linux-x64-cpu`). To try the build of a pull request:

```sh
gh run download <run id> --repo nyosegawa/speech.cpp --name speech-macos-arm64-metal
```

CI builds the checks so that they keep compiling, but does not run them, since they need weights and dumps that are not in
the repository. [development/releasing.md](development/releasing.md) says what else the workflow checks.
