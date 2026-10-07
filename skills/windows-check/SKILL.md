---
name: windows-check
description: Check a speech.cpp change on a Windows x64 machine with a Vulkan GPU. Covers taking the build and the check executables from CI, getting models and small dumps there, running stage checks, bit-for-bit comparisons with the previous release, the smoke scripts and the installer, and leaving the machine as it was. Use when a change must be verified on Windows or Vulkan, before a release, or when asked to test on Windows (Windows で確かめる、Vulkan で確認、RTX で計測、install.ps1 を試す). Not for building on macOS or Linux, or for measuring speed across runtimes (speech-bench).
---

# Checking on Windows with Vulkan

CI builds and links on Windows but has no GPU, so a change that touches computation is checked on a Windows machine
with a Vulkan GPU before it is released. The commands below run in PowerShell or cmd on that machine, which a session
reaches through whatever remote shell it has.

## Ground rules

- Work in one temporary folder (`%TEMP%\speech-<task>`), and remove it when done. Change nothing else on the machine:
  no installs, no PATH, no settings, no other program's data.
- Large files are fetched by the Windows machine itself (GitHub, Hugging Face), not copied over a remote shell, which
  may be slow. Send only small files: scripts and short dumps.

## 1. The build

- The pull request's archive: `gh run download <run> -R nyosegawa/speech.cpp -n speech-windows-x64-vulkan -D build`,
  then unpack it. `<run>` is the pull request's latest CI run (`gh pr checks <n>` links it).
- The check executables are not in that archive. Run the workflow on the branch with them kept:
  `gh workflow run build.yml --ref <branch> -f checks=true`, then download the artifact
  `speech-checks-windows-x64-vulkan` from that run.
- The previous release, for comparisons: its `speech-<version>-windows-x64-vulkan.zip` from GitHub releases.

## 2. Models and dumps

- Models: `speech pull <name>` with `SPEECH_MODEL_DIR` set to a folder inside the temporary one, or a copy already on
  the machine whose SHA-256 you checked.
- Dumps: make a short case on the Mac (`reference/<family>/dump.py` with a sentence of a second or two), check it there
  with the same check, pack it and send it; a full dump is hundreds of megabytes.

## 3. What to run

| What | How | Passes when |
|---|---|---|
| Stage checks | `<family>-<stage>-check.exe <model> <dump> … gpu` | within the bounds the check prints |
| Window or chunk independence | the family's codec check with its forced patterns | every pattern gives the whole decode's samples |
| Unchanged behaviour | `python same_audio.py <old speech.dll> <model> <new speech.dll> <model> <prompts.json> --voice … --seeds 1,2,3` | the same audio, bit for bit |
| Entry points | `worker_smoke.py`, `server_smoke.py`, `speech_cli_smoke.py` with a model of each family | ok |
| Installer | `$env:LOCALAPPDATA` set to a folder in the temporary one, then `install.ps1 -NoModifyPath`: a first install, a second run, `-Version` of the previous release and back | each step says what it did; the user PATH is unchanged |

A fast GPU may never take the path a slow machine takes (smaller windows, for example); force it with the check's
patterns rather than concluding from the fast path alone.

## 4. Windows specifics

- Paths and text are UTF-8 in speech.cpp; PowerShell 5.1 pipes may garble Japanese. Write results to files from the
  program itself, or pass a script with `powershell -EncodedCommand` (UTF-16LE, base64).
- `--host 0.0.0.0` may raise the firewall's prompt; prefer loopback.
- Processes started from a remote shell may keep it waiting; start long ones so that they return, and stop what you
  started.

## 5. Record it

Write what ran, on what GPU and driver, and what it gave, as a comment on the pull request. Then remove the temporary
folder.
