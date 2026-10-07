"""A client of the worker protocol 2 for the smoke scripts. It starts `speech worker`, and on every line it reads it
checks that the line is one JSON object with a string "type", that a chunk, progress, partial text or terminal message
belongs to a request in flight, and that a request gets exactly one terminal message (end, error or cancelled) and
nothing for its id after it. It also compares the model information of ready with `speech info --json`, and reads the
requests a recognition model's reference dumped.
"""

import json
import os
import subprocess
import time

TERMINAL = {"end", "error", "cancelled"}


class Worker:
    def __init__(self, speech, model, options, stderr=subprocess.DEVNULL):
        self.speech, self.model, self.options = speech, model, options
        self.proc = subprocess.Popen([speech, "worker", model, *options], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=stderr)
        self.in_flight = set()
        self.backlog = []
        self.started = time.perf_counter()
        self.ready = self._read_line()
        if self.ready["type"] != "ready":
            raise SystemExit(f"the worker did not start: {self.ready}")

    def _read_line(self):
        line = self.proc.stdout.readline()
        if not line:
            raise SystemExit("the worker exited")
        text = line.decode("utf-8").rstrip("\n")
        if text.endswith("\r"):
            raise SystemExit("a line ends with \\r")
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            message = None
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            raise SystemExit(f"a line on stdout is not a JSON object with a type: {text[:200]!r}")
        message["_at"] = time.perf_counter()
        return message

    def read(self):
        """The next message, checked against the requests in flight."""
        m = self._read_line()
        kind, id = m["type"], m.get("id")
        if kind in TERMINAL and id is None:
            if kind != "error":
                raise SystemExit(f"a terminal message without an id: {m}")
        elif kind in TERMINAL or kind in ("chunk", "progress") or (kind == "partial" and "error" not in m):
            if id not in self.in_flight:
                raise SystemExit(f"{kind} for {id!r}, which has no request in flight: {short(m)}")
            if kind in TERMINAL:
                self.in_flight.remove(id)
        return m

    def send_line(self, line):
        self.proc.stdin.write((line + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def send(self, message):
        """Sends a line that starts no request: a chunk after the first, a peek or a cancel."""
        self.send_line(json.dumps(message, ensure_ascii=False))

    def expect(self, id):
        """Notes that a line sent under `id` starts a request, which must get one terminal message."""
        if id in self.in_flight:
            raise SystemExit(f"the smoke started a second request under {id!r}")
        self.in_flight.add(id)

    def request(self, message):
        self.expect(message["id"])
        self.send(message)

    def next_for(self, id):
        """The next message for `id`, keeping the others for later calls."""
        for i, m in enumerate(self.backlog):
            if m.get("id") == id:
                return self.backlog.pop(i)
        while True:
            m = self.read()
            if m.get("id") == id:
                return m
            self.backlog.append(m)

    def until(self, id):
        """The messages for `id` up to and including its terminal message."""
        out = []
        while True:
            m = self.next_for(id)
            out.append(m)
            if m["type"] in TERMINAL:
                return out

    def terminal(self, id, kind=None, code=None, option=None):
        """The terminal message of `id`, checked to be of `kind`, and for an error of `code` and `option`."""
        m = self.until(id)[-1]
        check_message(m, kind, code, option)
        return m

    def error_without_id(self, code="invalid_argument", option=None):
        for i, m in enumerate(self.backlog):
            if m["type"] == "error" and "id" not in m:
                self.backlog.pop(i)
                break
        else:
            while True:
                m = self.read()
                if m["type"] == "error" and "id" not in m:
                    break
                self.backlog.append(m)
        check_message(m, "error", code, option)
        return m

    def close(self):
        """Closes stdin, reads to the end, and checks that every request was answered and the worker exited with 0."""
        self.proc.stdin.close()
        rest = []
        while self.in_flight or self.backlog:
            if self.backlog:
                rest.append(self.backlog.pop(0))
            else:
                rest.append(self.read())
        tail = self.proc.stdout.read()
        if tail:
            raise SystemExit(f"the worker wrote after its last answer: {tail[:200]!r}")
        code = self.proc.wait(timeout=60)
        if code != 0:
            raise SystemExit(f"the worker exited with {code}")
        return rest

    def check_model_information(self, added_voices=()):
        """Checks that ready's model is `speech info --json` of the model with the device, the threads and the voices added."""
        check_model_information(self.speech, self.model, self.ready["model"], added_voices)


def check_message(m, kind=None, code=None, option=None):
    if kind and m["type"] != kind:
        raise SystemExit(f"expected {kind}, got {short(m)}")
    if code or option:
        error = m.get("error") or {}
        if code and error.get("code") != code:
            raise SystemExit(f"expected the code {code}, got {short(m)}")
        if option is not None and error.get("option") != option:
            raise SystemExit(f"expected the option {option}, got {short(m)}")
        if not isinstance(error.get("message"), str) or not error["message"]:
            raise SystemExit(f"an error without a message: {m}")


def check_model_information(speech, model, loaded, added_voices=()):
    """Checks the information of a loaded model against `speech info --json` of its file and the voices added to it."""
    r = subprocess.run([speech, "info", model, "--json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    want = json.loads(r.stdout)
    for name in added_voices:
        want["voices"].append({"name": name, "language": "", "gender": "", "description": ""})
        for option in want["options"]:
            if option["name"] == "voice":
                option["choices"].append(name)
    got = dict(loaded)
    if not isinstance(got.pop("device", None), str) or not isinstance(got.pop("threads", None), int):
        raise SystemExit(f"the loaded model's information lacks the device or the threads: {loaded}")
    if json.dumps(got) != json.dumps(want):
        raise SystemExit(f"the loaded model's information differs from speech info --json:\n  {json.dumps(got)[:600]}\n  {json.dumps(want)[:600]}")


def dump_requests(speech, model, dump):
    """The requests of a dump of a recognition model's reference, each as (name, members, text, languages): for one of
    reference/fastconformer/dump.py, the request without options, the dump's text and no language, since FastConformer
    writes none; for one of reference/qwen3-asr/dump.py, which keeps each request in a folder of its own, auto, forced,
    auto-prompt and forced-prompt, the forced language given by the tag of general.languages whose name
    qwen3-asr.language_names gives (read with speech info --json --meta), the prompt by its text, and the language
    qwen-asr parsed by its tag, or none."""
    def text(folder):
        with open(os.path.join(folder, "text.txt"), encoding="utf-8") as f:
            return f.read()

    if not os.path.isfile(os.path.join(dump, "auto", "meta.json")):
        return [("auto", {}, text(dump), [])]
    r = subprocess.run([speech, "info", model, "--json", "--meta"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    meta = json.loads(r.stdout)["meta"]
    tags = dict(zip(meta["qwen3-asr.language_names"], meta["general.languages"]))
    requests = []
    for name in ("auto", "forced", "auto-prompt", "forced-prompt"):
        folder = os.path.join(dump, name)
        with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
            asked = json.load(f)
        members = {}
        if asked["language"]:
            members["language"] = tags[asked["language"]]
        if asked["prompt"]:
            members["prompt"] = asked["prompt"]
        requests.append((name, members, text(folder), [tags[asked["parsed_language"]]] if asked["parsed_language"] else []))
    return requests


def short(m):
    m = {k: v for k, v in m.items() if k != "_at"}
    if "pcm" in m:
        m["pcm"] = f"<{len(m['pcm'])} base64 bytes>"
    text = json.dumps(m, ensure_ascii=False)
    return text if len(text) < 400 else text[:400] + "..."
