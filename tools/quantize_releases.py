"""Checks that every file `speech quantize` writes names in speech.requires the first release that reads it: for each F32
model file and each type speech quantize writes, it writes the file and runs `speech info` of each earlier release given
on it, and fails where a release at or after the file's speech.requires refuses it, or one before reads it. The earlier
releases are `speech` built from their tags (git archive v0.7.1, with the submodule's ggml); a release reads a file when
its `speech info` takes it.

usage: python3 tools/quantize_releases.py <speech> <work dir> <F32.gguf>... --release <speech of an earlier release>...
"""

import json
import os
import subprocess
import sys
import tempfile

TYPES = ["f16", "q8_0", "q6_k", "q5_k", "q4_k"]


def numbers(release):
    return tuple(int(n) for n in release.split("."))


def version(speech):
    """The release of a `speech`, from its --version: "speech.cpp 0.7.1, C API 3.0"."""
    out = subprocess.run([speech, "--version"], capture_output=True, text=True, check=True).stdout
    return out.split()[1].rstrip(",")


def main():
    args = sys.argv[1:]
    if "--release" not in args or len(args) < 4:
        raise SystemExit(__doc__)
    at = args.index("--release")
    speech, work, models, earlier = args[0], args[1], args[2:at], args[at + 1:]
    releases = sorted(((version(s), s) for s in earlier), key=lambda r: numbers(r[0]))
    os.makedirs(work, exist_ok=True)
    failures = 0
    for model in models:
        for kind in TYPES:
            fd, out = tempfile.mkstemp(prefix="quantize-releases-", suffix=".gguf", dir=work)
            os.close(fd)
            try:
                subprocess.run([speech, "quantize", model, out, "--type", kind], check=True, capture_output=True)
                meta = json.loads(subprocess.run([speech, "info", out, "--json", "--meta"], capture_output=True, text=True, check=True).stdout)["meta"]
                requires = meta["speech.requires"]
                read = {}
                for release, binary in releases:
                    r = subprocess.run([binary, "info", out], capture_output=True, text=True)
                    read[release] = r.returncode == 0
                    if read[release] != (numbers(release) >= numbers(requires)):
                        failures += 1
                        verdict = r.stderr.strip().splitlines()[-1] if r.returncode else "reads it"
                        print(f"FAIL {os.path.basename(model)} in {kind.upper()} names {requires}, and {release}: {verdict}")
                seen = ", ".join(f"{release} {'reads it' if ok else 'refuses it'}" for release, ok in read.items())
                print(f"{os.path.basename(model)} in {kind.upper()}: speech.requires {requires}; {seen}")
            finally:
                os.remove(out)
    print("ok" if not failures else f"FAIL: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
