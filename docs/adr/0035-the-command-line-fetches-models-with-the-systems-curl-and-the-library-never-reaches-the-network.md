# The command line fetches models with the system's curl, and the library never reaches the network

Decided 2026-10-08.

## Context

Fetching a model by name (docs/adr/0033) needs HTTPS to Hugging Face and its content servers, through whatever proxy
and certificates the user's system has: corporate proxies, a certificate a company installs, or `https_proxy` set for
the shell. speech.cpp has no dependency but ggml and cpp-httplib's header, which speaks HTTPS only when linked with
OpenSSL. macOS, Windows 10 1803 and later, and nearly every Linux distribution ship curl.

## Decision

- **`speech` runs the system's curl**: `curl` on the PATH on macOS and Linux, and `System32\curl.exe` on Windows, which
  a bare `curl.exe` would look for in the current folder first. curl follows the redirect to Hugging Face's content
  servers, honors the proxy variables and its own configuration, and checks certificates against the system's store
  (Schannel on Windows, the keychain for macOS's curl). Without curl, a fetch fails saying to install it or to give a
  model file's path.
- **curl writes the file's bytes to a pipe**, and `speech` appends them to `<file>.part`, hashing them as they come,
  so that no path, which may hold characters outside ASCII, crosses to a program that may read it in another encoding.
  curl gets no handle or file descriptor of `speech` but its stdin, stdout and stderr, so it holds no model file or
  socket of the server open. A fetch that stops keeps the part; the next one asks for the rest with `--continue-at`.
  The file is renamed into place once its size and SHA-256 are the catalog's; a part whose size or SHA-256 cannot be
  the file's is removed, and the failure says that the same command fetches it again. A transfer slower than 1 kB/s
  for a minute fails rather than hangs.
- **Two processes fetching the same file take turns on `<file>.lock`**, `flock()` on macOS and Linux and
  `LockFileEx()` on Windows, which the system releases when a process ends however it ends. The second finds the file
  in place and loads it.
- **Progress goes to stderr**: one line rewritten on a terminal, a line for each tenth elsewhere, so that a worker's
  stdout carries the protocol alone.
- **Only the command line and the server fetch.** The library and the C API load files and never reach the network;
  a binding or a program on `libspeech` fetches as it chooses, and can read the pins from `speech models --json`.

The alternatives were turned down:

- cpp-httplib with OpenSSL. OpenSSL would be a second dependency in every release, with a certificate store of its own
  that a company's certificate and the system's settings do not reach.
- Each system's own library (NSURLSession, WinHTTP, libcurl on Linux). Three implementations of the same thing, and
  linking libcurl would add a shared library the Linux releases do not carry (docs/adr/0010).
- curl writing the part itself with `--output`. A path outside ASCII on Windows depends on how the installed curl
  reads its command line, and `speech` could not hash the bytes as they arrive.
- Fetching inside the library. Programs that pin and fetch their files themselves, ASIST among them, would carry
  network code they never call, and a library that may start downloads is harder to trust.

## Consequences

A fetch behaves as the user's curl does, proxies and certificates included. A machine without curl, such as a minimal
container, fetches nothing and gives paths. Resuming depends on the server answering ranges, which Hugging Face's
content servers do.
