# The page transcribes by regions with a detection model, and whole only up to a minute without one

## Context

The page's transcribe panel cut what it sent into pieces of at most 20 s at the quietest point, a length the user could
set. A piece of 20 s still holds several sentences, of which a FastConformer model drops some, and a piece without speech
is still recognized. `speech serve` now holds a detection model and transcribes by regions with `chunking_strategy`
([the record of transcription by regions](transcription-by-regions-recognizes-each-region-where-someone-speaks-alone-in-the-tools-and-joins-the-texts.md)).

Two limits bound what can be sent whole. The server takes at most 25 MB in a request, OpenAI's limit for the same
endpoint: about 13 minutes of 16-bit audio at 16 kHz. parakeet's memory grows with the square of the audio's length:
4.14 GB for 311 s on the CPU, which reaches about 12 GB at ten minutes.

## Decision

- **The transcribe panel has a second picker, for the detection model**, which the page loads as it loads the others.
- **With a detection model, the audio goes whole with `chunking_strategy` "auto"**, and the server transcribes its
  regions. Audio larger than one request carries goes in pieces of about 12 minutes, cut at the quietest point.
- **Without one, audio up to a minute goes whole, as the API transcribes it, and longer audio is refused** with a pointer
  to a detection model such as silero-vad, before anything is sent. A minute holds a few sentences at most, which
  FastConformer keeps, and parakeet's memory for it is under 1.5 GB.
- The length of a piece is no longer a setting.

The alternatives were turned down:

- A detection model required for every transcription. A short recording, and Qwen3-ASR, which reads long audio well,
  would need a second model for nothing.
- Audio of any length whole without a detection model. parakeet given ten minutes runs out of memory on a 16 GB Mac,
  and FastConformer drops whole sentences of long audio (522 of 600 Common Voice sentences joined into minutes).
- Pieces of a fixed length, as before. They hold several sentences each, and cut by length rather than where someone
  stops speaking.

## Consequences

`tools/server_page_browser_smoke.mjs` drives the page in a headless Chrome: the pickers in the tabs, a short file whole
with the API's text, a long file refused, the detection model chosen in its picker, and the long file then transcribed
with the text of `chunking_strategy` "auto". The live panel keeps its pieces until it is rebuilt on the Realtime API.
