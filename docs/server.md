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

The server holds at most one model of each task. It loads the models given, listens once they are ready, and logs to
stderr.

| Option | Meaning |
|---|---|
| `--host ADDRESS` | the address to listen on, 127.0.0.1 unless given. The server has no authentication and no TLS, so a server reachable from other machines belongs behind a proxy that adds them. The page is served only on 127.0.0.1, `::1` or `localhost` |
| `--port N` | the port, 8080 unless given, or 0 for any free one |
| `--open` | open the page in the browser once the server listens; without a MODEL the server starts with none |
| `--cors-origin ORIGIN` | an origin a web page may call the server from, such as `http://localhost:5173`, or `*` for any; repeatable. A web page of any other origin is refused (403 `origin_not_allowed`); curl and scripts send no origin and are not |
| `--add-voice NAME=FILE` | add a voice to the synthesis model |
| `--device`, `--threads`, `--no-warmup` | as for the worker, for every model the server loads, the page's included |

## Endpoints

| Endpoint | What it does |
|---|---|
| `GET /health` | answers `{"status":"ok"}` |
| `GET /v1/models` | lists the models held, the synthesis model first, as OpenAI's model objects whose `id` is the model's name, with `speech`, the model information ([c-api.md](c-api.md#model-information-as-json)) |
| `GET /v1/models/{id}` | one model; another id is a 404 (`model_not_found`) |
| `POST /v1/audio/speech` | speaks a text, as [OpenAI's create speech](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create) does |
| `POST /v1/audio/transcriptions` | recognizes the speech in a WAV file, as [OpenAI's create transcription](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create) does |

An endpoint of a task the server holds no model of answers a 404 whose message says so. One whose model the page is
replacing answers a 503 (`model_loading`) with `Retry-After`.

Each model serves one request at a time, in the order they arrive. A client that disconnects stops its request.

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

A member speech.cpp does not take is refused rather than ignored, and so is `instructions` for a model that takes none,
such as Qwen3-TTS 0.6B. A request without `seed` gets one drawn, which `X-Speech-Seed` returns; the same request with that
seed gives the same audio on the same device.

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

A transcription request is a `multipart/form-data` form, as OpenAI's API takes it. The upload may be up to 25 MB,
OpenAI's limit.

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
`duration` is the file's length in seconds. `language` is the tag of the language Qwen3-ASR heard (`ja`, or `ja,en` for a
long recording in several), and is left out by FastConformer, which names none. `segments` come from a model that gives
times (FastConformer). Members the recognizers have no value for are left out.

Every answer carries `X-Speech-Stop`: `complete`, or `model_limit` when the model reached the most tokens it writes, with
the text written up to there. OpenAI's other members (`temperature`, `stream`, `include[]` and the rest) are refused with
a 400 rather than ignored.

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@utterance.wav -F response_format=text
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@meeting.wav -F response_format=verbose_json
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@meeting.wav -F language=ja -F prompt="Claude Code、渋谷"
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@meeting.wav -F decoding=greedy   # reazonspeech-nemo-v2
```

## Errors

Errors have OpenAI's shape, `{"error":{"message":...,"type":...,"param":...,"code":...}}`, and `param` is the input at
fault:

| Category | HTTP | `type` | `code` |
|---|---|---|---|
| `invalid_argument` | 400 | `invalid_request_error` | `invalid_value` |
| `unsupported` | 400 | `invalid_request_error` | `unsupported_parameter` |
| `out_of_range` | 400 | `invalid_request_error` | `unsupported_value` |
| `model_file`, `device`, `out_of_memory`, `io`, `internal` | 500 | `server_error` | the category's name |

So an unknown voice is a 400 `unsupported_value` with `param` `voice`, and a `speed` on Qwen3-TTS a 400
`unsupported_parameter`. A member the server does not have is a 400 `unknown_parameter`.

A refused request gets its error status before any audio. An error while a stream runs ends a `pcm` stream early, which
the client reads as a broken transfer, and an SSE stream with `{"type":"error","error":{...}}`.

## The page

`speech serve --open` opens a page on which to try models without writing a request.

- A panel to speak and a panel to transcribe. At the top of each, a picker lists the catalog's models of its task with
  their sizes, whether they are fetched and their languages. Picking one fetches it, with progress that can be
  cancelled and is resumed the next time, and loads it in place of the model of its task.
- Speak takes a text and the model's options, plays the speech as the server streams it, and can play it again, save it
  as a WAVE file or hand it to Transcribe. For Irodori-TTS it makes a voice from a recording dropped on it or recorded in
  the browser.
- Transcribe takes an audio file in any format the browser decodes, or a recording, and shows the text, the language
  heard, the time it took and the segments. Set to Live, it shows the text while the recording goes on, updated every
  second or as often as set.
- The page sends audio in pieces of at most 20 s, or as long as set, each cut at a pause, so that a recording or a file
  of an hour or more needs no more of the server's memory than a minute does. The texts of the pieces are joined.
- The page is built into `speech` and needs nothing from the network.

The server prints the page's address, `http://127.0.0.1:8080/#token=…`, with a token that changes each time it starts;
the page opens only through that address, so that other web pages cannot use it. It is served while the server listens
on 127.0.0.1, `::1` or `localhost`. To use it from another machine, forward the port with SSH, the same number at both
ends (`ssh -L 8080:127.0.0.1:8080 host`), and open the printed address in your browser. The page loads only models of the catalog,
and removes none; `speech rm` does.
