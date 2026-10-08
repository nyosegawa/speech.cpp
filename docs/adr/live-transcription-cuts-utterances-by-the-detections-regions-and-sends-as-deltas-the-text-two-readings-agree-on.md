# Live transcription cuts utterances by the detection's regions and sends as deltas the text two readings agree on

## Context

speech.cpp transcribes speech as it is said: `speech asr --vad` from 16-bit PCM on stdin or the microphone, and
`speech serve` with OpenAI's Realtime transcription. OpenAI's form for this is a run of events per utterance:
`input_audio_buffer.speech_started` and `.speech_stopped` where turn detection finds them, `.committed`, then
`conversation.item.input_audio_transcription.delta` events that only add to the end of the text and `.completed`, whose
transcript is the whole text. OpenAI's own live model, `gpt-live-transcribe`, sends deltas while someone speaks, and
its client appends them and replaces them with the completed transcript
(developers.openai.com/api/docs/guides/realtime-transcription).

The library recognizes a request's audio whole and gives its text once the recognition ends; no family decodes audio
as it arrives. The detection of audio given a piece at a time gives `speech_detect()`'s regions, each once it is
certain ([the record of that detection](a-detection-of-audio-given-a-piece-at-a-time-gives-speech-detects-regions-each-once-it-is-certain.md)),
and `speech_detection_speaking()` reports the region under way, and whether it is kept: until its speech has lasted
longer than `min_speech_duration_ms`, it is dropped if its speech ends that soon. Transcription by regions cuts a file with the options `region_options()` gives
([its record](transcription-by-regions-recognizes-each-region-where-someone-speaks-alone-in-the-tools-and-joins-the-texts.md)).

Silero VAD's graphs are small (Apple M5, 2026-10-08): a minute of 16 kHz audio pushed in pieces of 20 ms takes 0.12 s
on the CPU with one thread, 0.17 s with the performance cores and 0.52 s on Metal; a minute given whole 0.087 s,
0.042 s and 0.11 s.

## Decision

- **One assembly of utterances in `tools/common`**, for the command line and the server: it takes audio at the
  caller's rate, cuts it into utterances, recognizes each alone, one at a time on a thread of its own, and gives its
  events in order: speech started, speech stopped, committed, delta, completed or failed. With a detection, the
  utterances are the detection's regions, with the options of transcription by regions, and each region's samples are
  cut as transcription by regions cuts a file's, so a stream gets the regions and the texts that `speech asr --vad` gets
  for the same audio. Without one, the caller commits, and an utterance is the audio taken since the last commit or
  clear.
- **A delta carries the beginning that two readings in a row agree on**, beyond what the deltas sent: while an utterance
  goes on, its audio from its start is recognized again, and two readings in a row agree on the code points both begin
  with, back to the end of a word both have whole where the script puts spaces between words, and, where that is all of
  the later reading, back to the last mark or space before its end. The end of the audio closes whatever was said last,
  with a mark a recognizer writes there, and two readings that end alike have most often heard a pause or a noise: with
  the whole of both taken, two readings of a buffer's first moments of silence agreed on 「何」, and two of a region on
  「湖では必ずしも。」 where a later reading went on 「…必ずしもヨットは」, and the deltas stopped there. An agreed
  beginning that does not begin with what was sent adds nothing, since a delta cannot take back text. The completed
  event carries the recognition of the whole utterance, which may differ from the deltas.
- **Readings run as often as the machine allows, with no setting**: once the previous reading has ended and at least
  0.2 s of new audio has come, the next starts at once, so readings never pile up. A committed utterance is recognized
  before any reading, and a reading under way when its utterance ends is stopped with `speech_request_cancel()`. 0.2 s,
  because the audio arrives every 0.1 to 0.2 s and the detection works in chunks of 32 ms.
- **Each recognition takes its place among the host's requests as it is accepted**: a committed utterance's when it is
  committed, and a reading's when it starts, with the assembly's lock held, so that a commit accepted before the reading
  runs before it and one accepted after it runs after it. A reading takes no place while it waits for audio, since it
  starts only once the audio is there, and it holds its place no longer than its run.
- **The assembly holds the buffer and the utterances committed and not yet recognized**, which a host bounds; a reading
  hands its copy of the buffer to the library's request and holds none of its own. With a detection, the buffer keeps
  only what a region may still need: from the start of the region under way, or without one the last `speech_pad_ms`
  and 0.1 s, so that silence does not pile up.
- **Speech started is given once the detection says the region under way is kept**, certain to be given: with the
  defaults 0.26 s after the speech begins. The utterance's deltas wait for it, so every speech started is followed by its
  speech stopped and its commit, and the utterances are the regions.
- **A detection model runs on the CPU with one thread** wherever it is loaded beside another model: `speech asr --vad`
  and `speech serve`, whatever `--device` and `--threads` say, which are the recognition and synthesis models'. It is
  faster there in pieces, and a file and the same audio streamed are cut on one device into the same regions.

The alternatives were turned down:

- Speech started when `speech_detection_speaking()` first reports the region, 0.2 s sooner. A noise shorter than
  `min_speech_duration_ms` would then start an item that is never committed, which a client that waits for each item's
  completion waits for forever, or else be committed and recognized, where a recognizer writes words for a click, and
  the utterances would no longer be the regions of `speech asr --vad`.
- Speech started once the region has lasted its padding, `min_speech_duration_ms`, `min_silence_duration_ms` and 0.1 s,
  which the assembly could tell without the detection's word: 0.6 s later, and still wrong where the probability stays
  between the two thresholds, which keeps a region open without lengthening its speech.
- Readings at a fixed interval. It wastes time on a fast machine and falls behind on a slow one.
- Each reading's whole text as it comes, or the difference from the reading before. A delta cannot take back text, and
  each reading rewrites the end of the one before, which the end of its audio cut.
- One delta of the whole text once the utterance is recognized, as manual commits had. No text comes while someone
  speaks.
- The detection on the device of the recognizer. Metal took four times as long for pieces of 20 ms, and a file and a
  stream detected on different devices could be cut into different regions.

## Consequences

Four FLEURS ja recordings joined with 1.5 s of silence, six regions, fed in real time in pieces of 20 ms to `speech serve` with
`server_vad` at 24 kHz, on an Apple M5 with the recognizer on Metal and the detection on the CPU (2026-10-08, another
program using the GPU at times), the times counted from where speech vad finds the speech without padding:

| | reazonspeech-v2 | qwen3-asr-0.6b |
|---|---|---|
| speech started, after the speech begins | 0.25 to 0.26 s | 0.25 to 0.27 s |
| first delta, after the speech begins | 0.68 to 1.53 s | 0.48 to 1.12 s |
| commit, after the speech ends | 0.60 s | 0.60 s |
| completed, after the speech ends | 0.69 to 1.15 s | 0.72 to 1.30 s |

Two runs each, while another program synthesized speech on the GPU. The first delta waits for two readings to agree
past the mark or space before their end, about half a second of speech. The commit waits for the silence of `min_silence_duration_ms` and for what follows within twice
`speech_pad_ms`. The detection took 0.12 s per minute of 16 kHz audio and 0.15 s of 24 kHz alone, and 0.3 to 0.6 s
beside the readings, which take the GPU and the CPU it waits on.

A beginning that two readings agree on and a later reading changes stays in the deltas until the completed event
replaces it, and later readings add nothing to them; where the final text goes on from what they gave, one more delta
gives the rest before the completed event, so that the deltas join into the text.
