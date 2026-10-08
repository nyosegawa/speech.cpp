"""Checks models by name: the catalog built into `speech` against tools/catalog/catalog.json, `speech models` in text
and JSON, `speech pull`, `speech rm` and a subcommand given a name, in a model folder of its own (SPEECH_MODEL_DIR).

It refuses what it should: a subcommand given no model (exit 2, naming the models to start with of its task, and for
`speech vad` every detection model), a name and a type the catalog does not hold (exit 2, listing what it holds), a
model file as the argument of pull, and rm of a model that is not fetched (exit 1). A path to a .gguf file stays a path, and one that is missing is the library's io
failure, with nothing fetched. It fetches through curl from Hugging Face no more than a mebibyte, from a local copy of
one of the catalog's files, given as LOCAL.gguf and found by its size and SHA-256:

- a part of the file's whole size whose bytes are not the file's is refused by its SHA-256 and removed;
- a part with all but the last mebibyte of the file is resumed: the fetch appends that mebibyte, renames the file into
  place, and the file is the local copy byte for byte; the same part with its first byte changed is refused, so the
  fetch kept the part rather than fetching the file again;
- while another process holds the file's lock, pull waits and writes no part, and once the file is in place and the
  lock released, it ends without fetching;
- a subcommand given the name loads the fetched file without fetching (speech info --json equals that of LOCAL.gguf);
- a file no entry names, an earlier revision's or another type's, is old in `speech models`, and `speech rm --old`
  removes it and leaves the catalog's file, and files outside the folders a fetch makes, which are not old;
  `speech rm NAME` removes the file and the folders it leaves empty.

usage: python3 checks/smoke/models_smoke.py <speech> <work dir> <LOCAL.gguf>
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

if len(sys.argv) != 4:
    raise SystemExit(__doc__.strip().splitlines()[-1])
speech, work, local = sys.argv[1:]
folder = os.path.join(os.path.abspath(work), "models-smoke")
shutil.rmtree(folder, ignore_errors=True)
os.makedirs(folder)
env = {**os.environ, "SPEECH_MODEL_DIR": folder}
MIB = 1 << 20


def run(*command, code=0):
    r = subprocess.run([speech, *command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if r.returncode != code:
        raise SystemExit(f"speech {' '.join(command)} exited with {r.returncode}, not {code}: {r.stderr.decode()[-800:]}")
    return r


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 * MIB):
            h.update(chunk)
    return h.hexdigest()


def failure(r, code):
    """The message of the failure a run ended with, checked to be "speech: <code>: <message>"."""
    line = r.stderr.decode().strip().splitlines()[-1]
    m = re.fullmatch(r"speech: ([a-z_]+): (.+)", line)
    if not m or m.group(1) != code:
        raise SystemExit(f"expected speech: {code}: ..., got {line!r}")
    return m.group(2)


def write_part(path, size, first=None):
    """The first `size` bytes of the local copy at `path`, with the first byte replaced by `first` when given."""
    with open(local, "rb") as src, open(path, "wb") as dst:
        left = size
        while left:
            chunk = src.read(min(left, 16 * MIB))
            dst.write(chunk)
            left -= len(chunk)
        if first is not None:
            dst.seek(0)
            dst.write(bytes([first]))


listing = json.loads(run("models", "--json").stdout)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "catalog", "catalog.json"), encoding="utf-8") as f:
    committed = json.load(f)["models"]
assert listing["directory"] == folder, listing["directory"]
assert listing["old"] == [], listing["old"]
pinned = ["name", "repository", "revision", "task", "languages", "voice_files", "start", "type"]
for m, c in zip(listing["models"], committed):
    assert {k: m[k] for k in pinned} == {k: c[k] for k in pinned}, (m["name"], c["name"])
    assert [{k: f[k] for k in ("type", "file", "size", "sha256")} for f in m["files"]] == c["files"], m["name"]
    assert re.fullmatch(r"[0-9a-f]{40}", m["revision"]) and m["type"] in [f["type"] for f in m["files"]], m["name"]
    for f in m["files"]:
        assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]), f
        assert f["url"] == f"https://huggingface.co/{m['repository']}/resolve/{m['revision']}/{f['file']}", f["url"]
        assert f["path"] == os.path.join(folder, m["repository"].replace("/", "--"), m["revision"], f["file"]), f["path"]
        assert not f["fetched"] and f["partial"] == 0, f
assert len(listing["models"]) == len(committed), "speech's catalog has other models than catalog.json"
starts = {}
for m in listing["models"]:
    assert set(m["start"]) <= set(m["languages"]), m["name"]
    for language in m["start"]:
        assert (m["task"], language) not in starts, f"{language} has two {m['task']} models to start with"
        starts[(m["task"], language)] = m["name"]
text = run("models").stdout.decode()
assert all(m["name"] in text for m in listing["models"]) and folder in text, text
print(f"speech models: the {len(committed)} models of catalog.json with every pin, none fetched, in {folder}")


def starting(task):
    return {m["name"] for m in listing["models"] if m["task"] == task and m["start"]}


for command, task in [(["asr"], "recognition"), (["asr", "a.wav"], "recognition"), (["tts", "-o", "x.wav"], "synthesis")]:
    message = run(*command, code=2).stderr.decode()
    other = "synthesis" if task == "recognition" else "recognition"
    assert all(n in message for n in starting(task)) and not any(n in message for n in starting(other)), message
message = run("worker", code=2).stderr.decode()
assert all(n in message for n in starting("synthesis") | starting("recognition")), message
# A detection model takes no language, and speech vad given no model names every one of the catalog.
detecting = {m["name"] for m in listing["models"] if m["task"] == "detection"}
for command in (["vad"], ["vad", "a.wav"]):
    message = run(*command, code=2).stderr.decode()
    assert all(n in message for n in detecting) and not any(n in message for n in starting("synthesis") | starting("recognition")), message
message = run("asr", "no-such-model", "a.wav", code=2).stderr.decode()
assert all(m["name"] in message for m in listing["models"]), message
some = listing["models"][0]
message = run("info", some["name"] + ":q9_9", code=2).stderr.decode()
assert all(f["type"] in message for f in some["files"]), message
run("pull", local, code=2)
run("pull", "no-such-model", code=2)
missing = os.path.join(work, "no-such-file.gguf")
failure(run("info", missing, code=1), "io")
assert os.listdir(folder) == [], os.listdir(folder)
print("no model, an unknown name and an unknown type: exit 2 naming the catalog's models; a missing .gguf file: io, nothing fetched")

size, digest = os.path.getsize(local), sha256(local)
found = [(m, f) for m in listing["models"] for f in m["files"] if f["size"] == size and f["sha256"] == digest]
if not found:
    raise SystemExit(f"{local} is no file of the catalog")
model, file = found[0]
name = model["name"] if file["type"] == model["type"] else f"{model['name']}:{file['type']}"
path, part, lock = file["path"], file["path"] + ".part", file["path"] + ".lock"
os.makedirs(os.path.dirname(path))

with open(part, "wb") as f:
    f.truncate(size)
zeros = hashlib.sha256()
for left in range(size, 0, -16 * MIB):
    zeros.update(bytes(min(left, 16 * MIB)))
message = failure(run("pull", name, code=1), "io")
assert file["sha256"] in message and zeros.hexdigest() in message, message
assert not os.path.exists(part) and not os.path.exists(path), os.listdir(os.path.dirname(path))
print(f"pull {name} over a part of {size} zero bytes: exit 1 with the SHA-256s, and the part removed")

write_part(part, size - MIB, first=0 if open(local, "rb").read(1) != b"\0" else 1)
message = failure(run("pull", name, code=1), "io")
assert file["sha256"] in message and not os.path.exists(part) and not os.path.exists(path), message
write_part(part, size - MIB)
r = run("pull", name)
assert r.stdout.decode() == path + "\n", r.stdout
assert sha256(path) == digest, "the resumed file is not the catalog's"
assert sorted(os.listdir(os.path.dirname(path))) == [os.path.basename(path)], os.listdir(os.path.dirname(path))
print(f"pull {name} over all but its last MiB: the file in place, the local copy byte for byte; a part whose first byte differs is "
      "refused, so the part was kept")

os.rename(path, os.path.join(work, "models-smoke-held.gguf"))
holder = open(lock, "a+b")
if os.name == "nt":
    import msvcrt
    msvcrt.locking(holder.fileno(), msvcrt.LK_NBLCK, 1)
else:
    import fcntl
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
waiting = subprocess.Popen([speech, "pull", name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
time.sleep(3)
assert waiting.poll() is None, f"pull did not wait for the lock: {waiting.stderr.read().decode()}"
assert not os.path.exists(part), "pull wrote a part while another process held the lock"
os.rename(os.path.join(work, "models-smoke-held.gguf"), path)
if os.name == "nt":
    holder.seek(0)
    msvcrt.locking(holder.fileno(), msvcrt.LK_UNLCK, 1)
else:
    fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
holder.close()
out, err = waiting.communicate(timeout=60)
assert waiting.returncode == 0 and out.decode() == path + "\n", err
assert sha256(path) == digest and not os.path.exists(part), "the waiting pull changed the file"
print("pull while another process holds the lock: it waits, writes nothing, and ends once the file is in place")

ours = json.loads(run("info", local, "--json").stdout)
r = run("info", name, "--json")
assert json.loads(r.stdout) == ours and not os.path.exists(part), "speech info NAME differs from speech info on the local copy"
entry = next(m for m in json.loads(run("models", "--json").stdout)["models"] if m["name"] == model["name"])
assert next(f for f in entry["files"] if f["type"] == file["type"])["fetched"], entry
print(f"speech info {name}: the local copy's information, and speech models says it is fetched")

repository_dir = os.path.dirname(os.path.dirname(path))
earlier = os.path.join(repository_dir, "0" * 40, file["file"])
other = os.path.join(os.path.dirname(path), "Other-F32.gguf")
theirs = [os.path.join(folder, "notes.txt"), os.path.join(folder, "mine", "x.gguf"), os.path.join(repository_dir, "y.gguf")]
os.makedirs(os.path.dirname(earlier))
os.makedirs(os.path.join(folder, "mine"))
for p, n in ((earlier, 3), (other, 5), *((t, 1) for t in theirs)):
    with open(p, "wb") as f:
        f.write(b"x" * n)
old = json.loads(run("models", "--json").stdout)["old"]
assert sorted((o["path"], o["size"]) for o in old) == sorted([(earlier, 3), (other, 5)]), old
text = run("models").stdout.decode()
assert all(os.path.relpath(p, folder) in text for p in (earlier, other)), text
run("rm", "--old")
assert not os.path.exists(earlier) and not os.path.exists(os.path.dirname(earlier)) and not os.path.exists(other) and os.path.exists(path)
assert all(os.path.exists(t) for t in theirs), "rm --old removed a file outside the folders a fetch makes"
for t in theirs:
    os.remove(t)
print("an earlier revision's file and another type's: old in speech models; rm --old removes them and their emptied folder, "
      "and no file outside the fetch's folders")

run("rm", name)
assert not os.path.exists(repository_dir), os.listdir(folder)
failure(run("rm", name, code=1), "invalid_argument")
print(f"rm {name}: the file and its emptied folders removed; again: exit 1")
shutil.rmtree(folder)
