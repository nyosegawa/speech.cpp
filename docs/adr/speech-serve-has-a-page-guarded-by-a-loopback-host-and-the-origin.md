# speech serve has a page, guarded by a loopback Host and the Origin

## Context

Without a page, trying a model takes a command line per request; audio.cpp's web UI lets a user pick models, fetch them
and try them in a browser. `speech serve` has no authentication, and any web page the user has open can send requests
to 127.0.0.1: a simple request needs no CORS to arrive, and a page that rebinds its own host name to 127.0.0.1 can read
the answers. Endpoints that fetch gigabytes into the user's cache folder and replace the server's models must not be
reachable that way.

## Decision

- **The page is plain HTML, CSS and JavaScript in `tools/server/page/`, built into `speech` as the catalog is**, with no
  framework, no build step and nothing from the network. Its scripts are ES modules, one per responsibility: the
  server's endpoints, the tabs, the model picker of each task, the option form built from the model information, the
  names of languages, playback and the waveform, recording and WAVE files, the text transcribed, each panel. Colours and radii are CSS variables in one
  place, light and dark as the system sets it. It is served with a Content-Security-Policy that allows the server's own
  scripts, styles and connections alone, and that no other page frames it.
- **The page speaks and transcribes through OpenAI's endpoints**, as any client does: speech as SSE, played through Web
  Audio as it arrives, and audio of any format the browser decodes, or a recording, sent as a 16-bit WAVE file at the
  model's rate. Its own endpoints, under `/speech/`, list the catalog with the models held, load a model of the catalog
  in place of its task's
  ([the record of a model of each task](speech-serve-holds-a-model-of-each-task-and-replaces-one-only-after-its-requests-end.md))
  with the fetch's progress as server-sent events, and add a voice made from a recording through `speech_voice_add()`.
- **The page and its endpoints answer only a Host of 127.0.0.1, localhost or [::1] with the server's port**, against DNS
  rebinding, **and exist only while the server listens on 127.0.0.1, ::1 or localhost.** On another address they answer
  how to reach the page through an SSH tunnel, and `--open` there is a usage error.
- **Every endpoint refuses a request whose Origin is neither the server's own nor one of `--cors-origin`.** A browser
  sends Origin on every cross-origin request, so no other web page reaches the server, while curl and scripts, which send
  none, keep working.
- **The page fetches models of the catalog alone**, by their names, and nothing on it removes a file.

The alternatives were turned down:

- A framework (Svelte with Tailwind), which needs Node and a build step for every change, and whose output is harder to
  read in the repository than the files it serves.
- A token in the page's address, as Jupyter has, which the page had until 2026-10-08: 128 random bits made at each start,
  printed after `#` and sent as `Authorization: Bearer`. With the Origin and the Host rules, no web page reaches the
  page's endpoints, and what the token added was a guard against other users and programs of the same machine, which
  reach 127.0.0.1 as well. They could fetch catalog models into the cache and replace the models held, but not read a
  file or run code. Ollama guards its API with the Origin and the Host alone, and llama.cpp's server, its page included,
  with nothing unless given `--api-key`. The token made every opening of the page go through the address the server
  printed, and the user decided that it is not worth that.
- Answering foreign origins without CORS headers. A simple request still arrives and runs: a page could make the server
  speak or load a model without reading the answer.
- The page on any address. Its requests would travel in clear over HTTP from another machine; an SSH tunnel gives the
  same page with the server on 127.0.0.1.

## Consequences

A web app on another origin calls the server only when the server names its origin with `--cors-origin`. The page works
offline once its models are fetched. Every change to the page is a change to `speech`, released with it.
