"""A client of `speech serve` for the smoke scripts: starts it on a free port, waits until it answers, and stops it when
the script ends, however it ends, so that a failed check leaves no server behind; and sends requests with the standard
library alone.
"""

import atexit
import http.client
import json
import re
import socket
import subprocess
import threading
import time


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, speech, args, env=None, host="127.0.0.1"):
        self.port = free_port()
        self.process = subprocess.Popen([speech, "serve", *args, "--port", str(self.port), "--host", host], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, env=env)
        atexit.register(self.stop)
        # stderr is read as the server writes it, since a log that fills the pipe would stop the server. The reader is a
        # daemon: Python joins the other threads before it runs atexit, and this one ends only once the server has.
        self.lines = []
        self.reader = threading.Thread(target=lambda: self.lines.extend(line.decode(errors="replace") for line in self.process.stderr),
                                       daemon=True)
        self.reader.start()
        t0 = time.perf_counter()
        while True:
            if self.process.poll() is not None:
                self.reader.join()
                raise SystemExit(f"the server exited with {self.process.returncode}: {''.join(self.lines)[-600:]}")
            try:
                if self.call("GET", "/health")[0] == 200:
                    break
            except OSError:
                time.sleep(0.2)
        self.started = time.perf_counter() - t0

    @property
    def page(self):
        """The page's address with its token as the server printed it before it listened, or None where it has no page."""
        for _ in range(50):
            for line in self.lines:
                found = re.search(r"(http://\S+/#token=([0-9a-f]+))", line)
                if found:
                    return found.group(1)
            if any("listening on" in line for line in self.lines):
                time.sleep(0.1)
                if not any("#token=" in line for line in self.lines):
                    return None
            time.sleep(0.1)
        return None

    @property
    def token(self):
        return self.page.rsplit("=", 1)[1]

    def call(self, method, path, body=None, headers=None, stream=False):
        """The status, the headers by lower-case name and the body, or the open response of a stream."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        if stream:
            return r
        data = r.read()
        c.close()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, data

    def post_json(self, path, member, headers=None):
        return self.call("POST", path, json.dumps(member, ensure_ascii=False).encode(), {"Content-Type": "application/json", **(headers or {})})

    def stop(self):
        """Stops the server and returns what it wrote on stdout."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.join()
        return self.process.stdout.read()


def expect_error(got, status, code, param, what):
    """Checks an error in OpenAI's shape, with its status, type, code and param, and prints it."""
    s, _, body = got
    error = json.loads(body).get("error", {}) if body else {}
    kind = "server_error" if status >= 500 else "invalid_request_error"
    if s != status or error.get("code") != code or error.get("param") != param or error.get("type") != kind or not error.get("message"):
        raise SystemExit(f"{what}: expected {status} {code} ({param}), got {s} {body[:300]!r}")
    print(f"{what}: {status} {kind} {code} ({param}) as expected: {error['message'][:90]}")
