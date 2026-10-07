# Linux releases run on glibc 2.34, in a Vulkan and a CPU build

Superseded in part by docs/adr/0017: a release has one archive per platform, `speech-<version>-<platform>.zip`.

Decided 2026-10-05.

## Context

Releases carried macOS arm64 (Metal) and Windows x64 (Vulkan). Linux x64 joins them. Unlike the other two
systems, a Linux binary runs only where the C library is at least as new as the one it was linked against,
and the GPU stack differs from machine to machine: some have a Vulkan driver, some have NVIDIA's CUDA, and
servers and containers often have no GPU and no Vulkan loader at all.

Measured on Ubuntu 22.04 (glibc 2.35, GCC 11.4) on 2026-10-05: the worker, the tools and `libspeech.so` need
glibc symbols up to `GLIBC_2.34` (`__libc_start_main`, `pthread_create`, `dlopen` and the other functions
glibc 2.34 moved into libc), and nothing from 2.35. The Linux Vulkan SDK 1.4.357.0's `glslc` and loader need
glibc 2.34 and GCC 11's libstdc++ (`GLIBCXX_3.4.29`). GitHub retired its Ubuntu 20.04 runners in April 2025;
`ubuntu-22.04` is the oldest it hosts.

## Decision

- The Linux jobs run on `ubuntu-22.04`, and the binaries run on glibc 2.34 or later: Ubuntu 22.04, Debian 12,
  Fedora 35, RHEL 9 and the releases after them. CI reads the newest glibc symbol each file needs and fails
  when it is newer than 2.34.
- The C++ runtime is linked into each file (`-static-libstdc++ -static-libgcc`) and ggml's OpenMP is off, so
  that a file needs no shared library but glibc's and, in the Vulkan build, `libvulkan.so.1`. CI fails a
  file that needs another one. `libspeech.so` carries its own libstdc++ hidden behind `src/speech.map`, which
  is safe because no C++ type or exception crosses the C API.
- There are two builds, `linux-x64-vulkan` and `linux-x64-cpu`. The Vulkan build runs on a GPU of any vendor
  whose driver speaks Vulkan; the CPU build needs no Vulkan loader, for machines without a GPU or its driver.
  Both are built for x86-64 with AVX2, FMA and F16C (`-DGGML_NATIVE=OFF`), as the Windows build is.
- There is no CUDA build.

The alternatives were turned down:

- Building in a container with an older glibc, such as manylinux_2_28 (glibc 2.28, which would add RHEL 8,
  Debian 11 and Ubuntu 20.04). The Vulkan SDK's `glslc` does not run there, so ggml's shaders would need a
  shader compiler built from source in a second stage, and a pinned toolchain of the container's own. Ubuntu
  20.04 and Debian 11 are past their regular support; RHEL 8 is the one supported system left out, and it
  can build from source.
- Building on `ubuntu-latest` (24.04, glibc 2.39). The floor would follow whatever 2.39's headers pick and
  leave out Ubuntu 22.04 and Debian 12, the most common systems with 2.34 to 2.38.
- Linking glibc statically. A static executable cannot load the Vulkan loader and the drivers it loads, and
  `libspeech.so` cannot be static.
- One Vulkan build alone. It does not start where `libvulkan.so.1` is missing, `--device cpu` included,
  since the dynamic linker refuses it before `main`.
- Loading ggml's backends at run time (`GGML_BACKEND_DL`), so that one build finds Vulkan when it is there. A
  release would become several files per tool, against the rule that each tool is one file, and every
  platform would link differently.
- A CUDA build. Vulkan already runs NVIDIA GPUs at the speeds docs/models/ lists for an RTX 2080. A CUDA build
  would carry cuBLAS, several hundred megabytes, or require the user's CUDA toolkit to match, would tie the
  binary to a range of NVIDIA drivers, and would put the CUDA toolkit and its kernel compilation into CI. It
  stays a build from source (`-DGGML_CUDA=ON`).

## Consequences

A release has four more archives, `speech-worker-<version>-linux-x64-{vulkan,cpu}.zip` and
`speech-cpp-tools-<version>-linux-x64-{vulkan,cpu}.zip`. The Vulkan build's files are about 48 MB each with
their shaders, the CPU build's about 4 MB. A CPU without AVX2 is not supported by any release. When GitHub
retires `ubuntu-22.04`, the job moves to the next image, and the glibc check shows at once whether the floor
moves; the floor in the workflow, the README, docs/install.md, AGENTS.md and this record then change together.
