# Server and page

This page describes `speech serve`, which serves a synthesis model and a recognition model over HTTP with a subset of
OpenAI's audio API, and a page on which to try models.

```
speech serve [MODEL [MODEL]] [--open] [--host 127.0.0.1] [--port 8080] [--cors-origin ORIGIN|*]...
             [--add-voice NAME=FILE]... [--device NAME] [--threads N] [--no-warmup]
```

```sh
speech serve --open                                   # the page, without a model until one is picked there
speech serve qwen3-tts-0.6b qwen3-asr-0.6b            # speech and transcriptions
speech serve irodori-tts-mf --add-voice me=me.voice.gguf --cors-origin http://localhost:5173
```

The server holds at most one model of each task
([ADR 0038](adr/0038-speech-serve-holds-a-model-of-each-task-and-replaces-one-only-after-its-requests-end.md)). It loads
the models given, warmed up as the worker warms them, listens once they are ready, and logs to stderr.

| Option | Meaning |
|---|---|
| `--host ADDRESS` | the address to listen on, 127.0.0.1 unless given. The server has no authentication and no TLS, so a server reachable from other machines belongs behind a proxy that adds them. On an address other than 127.0.0.1, `::1` or `localhost`, the page is off |
| `--port N` | the port, 8080 unless given, or 0 for any free one |
| `--open` | open the page in the browser once the server listens; without a MODEL the server starts with none |
| `--cors-origin ORIGIN` | an origin a web page may call the server from, such as `http://localhost:5173`, or `*` for any; repeatable. Preflight requests are answered, and `X-Sample-Rate`, `X-Speech-Seed` and `X-Speech-Stop` are exposed |
| `--add-voice NAME=FILE` | add a voice to the synthesis model |
| `--device`, `--threads`, `--no-warmup` | as for the worker, for every model the server loads, the page's included |

## Endpoints

| Endpoint | What it does |
|---|---|
| `GET /health` | answers `{"status":"ok"}` |
| `GET /v1/models` | lists the models held, the synthesis model first, as OpenAI's model objects (`id` is the model's name, `owned_by` is `"speech.cpp"`) with `version`, the release, and `speech`, the model information ([c-api.md](c-api.md#model-information-as-json)) |
| `GET /v1/models/{id}` | one model; another id is a 404 (`model_not_found`) |
| `POST /v1/audio/speech` | speaks a text, as [OpenAI's create speech](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create) does |
| `POST /v1/audio/transcriptions` | recognizes the speech in a WAV file, as [OpenAI's create transcription](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create) does |

An endpoint of a task the server holds no model of answers a 404 whose message says so. One whose model the page is
replacing answers a 503 (`model_loading`) with `Retry-After`.

Each model serves one request at a time, in the order they arrive. A client that disconnects while it waits is dropped,
and one that disconnects while its request runs stops it at the next chunk, sampler step, codec window or stage, as a
worker's `cancel` does.

## Speech

A speech request is a JSON object:

| Member | Meaning |
|---|---|
| `input` | the text, required |
| `model` | the synthesis model's `id`, or left out. Any other model is a 404 (`model_not_found`) |
| `voice` | one of the model's voices, required: a Qwen3-TTS speaker or a voice of `--add-voice` |
| `response_format` | `wav` (the default) or `pcm`. OpenAI's default is `mp3`, which speech.cpp does not encode; `mp3`, `opus`, `aac` and `flac` are refused |
| `stream_format` | `audio` (the default) or `sse`, which needs `pcm` |
| `speed`, `instructions` | OpenAI's members, as the options of the same names |
| any other option | every option of the vocabulary by its name ([c-api.md](c-api.md#options)), such as `language`, `seed`, `steps` or `temperature`, which the model checks |

A member speech.cpp does not take is refused rather than ignored, and so is `instructions` other than `""` for a model
that takes none, such as Qwen3-TTS 0.6B. A member set to `null` counts as left out. A request without `seed` gets one
drawn from 0 to 2^53 - 1, and the same request with that seed gives the same audio on the same device.

| Response | What it sends |
|---|---|
| `wav` | the whole file, 16-bit mono at the model's rate, once the speech is made, with the headers `X-Sample-Rate`, `X-Speech-Seed` and `X-Speech-Stop` (`complete`, `max_seconds` or `model_limit`) |
| `pcm` | raw 16-bit little-endian mono at the model's rate, `Content-Type: audio/pcm`, streamed as the model makes it, with `X-Sample-Rate` and `X-Speech-Seed`. The headers leave before the speech ends, so they cannot carry the stop reason |
| `pcm` with `stream_format: "sse"` | the same PCM as server-sent events: `{"type":"speech.audio.delta","audio":"<base64 PCM>"}` for each chunk, then `{"type":"speech.audio.done","seed":42,"samples":134400,"stop":"complete"}`. OpenAI's done event carries the usage in tokens, which speech.cpp does not count |

```sh
# A WAV file
curl http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"input": "明日の東京は晴れです。", "voice": "ono_anna"}' -o out.wav

# PCM played as it arrives; the rate is the X-Sample-Rate header (24000 for Qwen3-TTS, 48000 for Irodori-TTS)
curl -sN http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"input": "明日の東京は晴れです。", "voice": "ono_anna", "response_format": "pcm"}' |
    ffplay -nodisp -autoexit -f s16le -ar 24000 -ch_layout mono -i -
# On Linux with ALSA: ... | aplay -f S16_LE -r 24000 -c 1

# Server-sent events
curl -N http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"input": "明日の東京は晴れです。", "voice": "ono_anna", "response_format": "pcm", "stream_format": "sse"}'
```

From Python with [requests](https://requests.readthedocs.io), streaming PCM into a WAV file as it arrives:

```python
import requests
import wave

with requests.post("http://127.0.0.1:8080/v1/audio/speech",
                   json={"input": "明日の東京は晴れです。", "voice": "ono_anna", "response_format": "pcm"},
                   stream=True) as r:
    r.raise_for_status()
    with wave.open("out.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(r.headers["X-Sample-Rate"]))
        for chunk in r.iter_content(chunk_size=None):
            w.writeframes(chunk)
```

## Transcriptions

A transcription request is a `multipart/form-data` form, as OpenAI's API reference defines `CreateTranscriptionRequest`
(github.com/openai/openai-openapi at commit 31af4fc, 2026-10-05). The upload may be up to 25 MB, OpenAI's limit.

| Member | Meaning |
|---|---|
| `file` | the audio, required: a WAV file, 16-, 24- or 32-bit PCM or 32-bit float at any rate, its channels averaged and resampled to the model's rate. Any other file is refused with a 400 (`param` `file`); convert it first (`ffmpeg -i in.mp3 out.wav`) |
| `model` | the recognition model's `id`, or left out; any other model is a 404 (`model_not_found`) |
| `language` | a tag of one of the model's languages, or `auto` (the default) |
| `prompt` | what the model is told of the audio before it hears it, for a model that takes it (Qwen3-ASR) |
| `response_format` | `json` (the default), which answers `{"text":"..."}`; `text`, the text alone as `text/plain`; or `verbose_json` (below). `srt`, `vtt` and `diarized_json` are refused: speech.cpp gives neither subtitles nor speakers |
| `timestamp_granularities[]` | `segment`, with `verbose_json`; a model that gives no times refuses it. `word` is refused |
| `decoding` | speech.cpp's own: `beam` (the default) or `greedy`, for a model that takes it (reazonspeech-nemo-v2) |

`verbose_json` answers `{"task":"transcribe","language":…,"duration":…,"text":…,"segments":[{"id":0,"start":…,"end":…,"text":…}]}`.
`duration` is the file's length in seconds. `language` is the BCP 47 tag of the language the model heard or was forced
to (`ja`), and for audio over 1200 s that Qwen3-ASR heard in several languages, their tags joined with commas in the
order of the audio (`ja,en`). It is left out where the model names no language, as FastConformer does, and for audio
without speech. `segments` come from a model that gives times (FastConformer); Qwen3-ASR gives none, which OpenAI's schema
allows. The members of OpenAI's segment that the recognizers have no value for (`seek`, `tokens`, `temperature`,
`avg_logprob`, `compression_ratio`, `no_speech_prob`) and the usage are left out.

Every answer carries `X-Speech-Stop`: `complete`, or `model_limit` when the model reached the most tokens it writes, with
the text written up to there. OpenAI's other members (`temperature`, `stream`, `include[]` and the rest) are refused with
a 400 rather than ignored, and so is a member given twice.

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@utterance.wav -F response_format=text
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@meeting.wav -F response_format=verbose_json
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@meeting.wav -F language=ja -F prompt="Claude Code、渋谷"
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@meeting.wav -F decoding=greedy   # reazonspeech-nemo-v2
```

## Errors

Errors have OpenAI's shape, `{"error":{"message":...,"type":...,"param":...,"code":...}}`. A failure of the library
becomes an error by its category alone ([c-api.md](c-api.md#errors)), and `param` is the input at fault, with `text`
written as `input` and `audio` as `file`:

| Category | HTTP | `type` | `code` |
|---|---|---|---|
| `invalid_argument` | 400 | `invalid_request_error` | `invalid_value` |
| `unsupported` | 400 | `invalid_request_error` | `unsupported_parameter` |
| `out_of_range` | 400 | `invalid_request_error` | `unsupported_value` |
| `model_file`, `device`, `out_of_memory`, `io`, `internal` | 500 | `server_error` | the category's name |

So an unknown voice is a 400 `unsupported_value` with `param` `voice`, a `speed` on Qwen3-TTS a 400
`unsupported_parameter`, and a text longer than the model takes a 400 `unsupported_value` with `param` `input`. The
server's own refusals have their own codes: `unknown_parameter` for a member it does not have, `invalid_type` for a value
of another type, `missing_required_parameter`, `unsupported_value` for a format it does not give, and 404
`model_not_found`. A body that is not JSON is a 400 without a code.

A `pcm` or SSE stream begins once the library has begun the request's work, so a request it refuses gets its error
status. An error after that ends a `pcm` stream without its last chunk, which the client reads as a broken transfer, and
an SSE stream with `{"type":"error","error":{...}}`.

## The page

On 127.0.0.1, `::1` or `localhost`, the server also serves a page on which to try models without writing a request.
`speech serve --open` opens it.

- A panel to speak and a panel to transcribe. At the top of each, a picker lists the catalog's models of its task with
  their sizes, whether they are fetched and their languages. Picking one fetches it, with progress that can be
  cancelled and is resumed the next time, and loads it in place of the model of its task.
- Speak takes a text and the model's options, plays the speech as the server streams it, and can play it again, save it
  as a WAVE file or hand it to Transcribe. For Irodori-TTS it makes a voice from a recording dropped on it or recorded in
  the browser.
- Transcribe takes an audio file in any format the browser decodes, or a recording, and shows the text, the language
  heard, the time it took and the segments.
- The page is plain HTML, CSS and JavaScript built into `speech`, and needs nothing from the network.

The server has no authentication, and any web page open in a browser can send requests to 127.0.0.1, so the page is
guarded ([ADR 0039](adr/0039-speech-serve-has-a-page-guarded-by-a-token-a-loopback-host-and-the-origin.md)):

- The endpoints that only the page calls take a token that the server makes each time it starts. It prints the page's
  address with the token after `#`, `http://127.0.0.1:8080/#token=…`, which `--open` opens. The fragment reaches no
  server log and no referrer, and the page sends the token as `Authorization: Bearer`. Without it these endpoints answer
  401 (`missing_token`), and with another token 403 (`invalid_token`).
- The page and its endpoints answer only a `Host` of 127.0.0.1, localhost or [::1] with the server's port, against DNS
  rebinding (403 `host_not_allowed`), and only while the server listens on such an address. Elsewhere `GET /` says how to
  reach the page through an SSH tunnel with the same port at both ends, and the endpoints answer 404 (`page_off`).
- Every endpoint, OpenAI's included, refuses a request from a web page whose `Origin` is neither the server's own nor one
  of `--cors-origin` (403 `origin_not_allowed`). A request without an `Origin`, from curl or a script, is not refused.
- The page loads models of the catalog alone, by their names, and nothing on it removes a file; `speech rm` does.

The page's endpoints, for the page and anything that has the token:

| Endpoint | What it does |
|---|---|
| `GET /speech/models` | answers `{"catalog": …, "synthesis": {"held": …, "replacing": …}, "recognition": {…}}`: the catalog as `speech models --json` gives it, and for each task the model held, `{"name", "path", "model"}` (`name` is `null` for a file given on the command line), and the name of the model being loaded in its place |
| `POST /speech/load` | with `{"model": "NAME[:TYPE]"}`, fetches the model and loads it in place of its task's, answering with server-sent events: `{"type":"fetch","done":…,"total":…}`, `{"type":"load"}`, then `{"type":"loaded","task":…,"held":{…}}` or `{"type":"error","error":{…}}`. A name outside the catalog is a 404 (`model_not_found`), and a second load of a task while one runs a 409 (`model_loading`) |
| `POST /speech/voices` | a form with `name` and a WAVE file `file`: adds a voice made from the recording to the synthesis model, which keeps it while it stays loaded, and answers `{"held": {…}}` |

A client that goes away stops a fetch, whose part the next load resumes. The model held serves until the fetch is done,
and the load replaces it once its requests have ended.
