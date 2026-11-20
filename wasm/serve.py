"""Serves the folder this file is in over HTTP on 127.0.0.1, with the headers that make a page cross-origin isolated
(Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy), which the build with threads needs to share its memory.

usage: python3 serve.py [port]
"""

import functools
import http.server
import os
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
folder = os.path.dirname(os.path.abspath(__file__))
with http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=folder)) as server:
    print(f"serving {folder} at http://127.0.0.1:{port}/")
    server.serve_forever()
