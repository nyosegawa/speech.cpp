# speech serve gives speakers in diarized_json by regions and in each Realtime completed, which waits for them

## Context

A program that transcribes a meeting, such as a minutes app, runs one `speech serve`, opens a Realtime session for each
source of audio (the microphone, the system's audio), and wants to know who spoke within each utterance soon after it
ends; after the meeting it sends the whole recording again for a more accurate pass. OpenAI's API at
github.com/openai/openai-openapi commit 234829e (2026-10-07) gives speakers only in `response_format` `diarized_json` of
the create transcription request, for `gpt-4o-transcribe-diarize`: segments of `type` `transcript.text.segment` with
`id`, `start`, `end`, `text` and `speaker`, a label of `known_speaker_names[]` or else `A`, `B` and on, and
`chunking_strategy` required for audio over 30 s. Its Realtime transcription has no diarization.

The library's diarization of audio as it arrives gives each segment once nothing that follows can change it
([the record of the settings](a-diarization-takes-nemos-streaming-and-postprocessing-settings-offline-for-a-whole-recording-and-low-latency-for-a-stream.md)):
in `low_latency` a chunk is diarized once 1.04 s of audio after its start has arrived. Each diarization numbers its own
speakers from 0, and cannot tell whether someone it hears is someone another diarization heard. The recognizers give no
times within a region but FastConformer's segments.

## Decision

- **A Realtime session diarizes only when asked**, by speech.cpp's own `session.audio.input.diarization`: `null`, the
  default, or an object of the diarization model's request options, `{}` for the server's defaults and then the model's,
  so that a session of the microphone and one of the system's audio choose apart. The same options in a later
  `session.update` keep the diarization under way; other options, or diarization given again after `null`, start one of
  the audio from there, and `null` ends it. An utterance keeps the diarization under way when it is committed, whose end
  settles the rest of its audio.
- **The completed event waits for the speakers and carries them**, as speech.cpp's own `speakers`: each speaker's parts
  of the utterance, from its `speech_started` to its `speech_stopped`. The wait begins once the utterance is recognized,
  so it overlaps the recognition.
- **Speakers in OpenAI's shapes are labels, in each one's units**: the Realtime events and `diarized_json` label them
  `A`, `B` and on, in the order they first speak; the Realtime events give times in milliseconds of the session's audio,
  as `audio_start_ms` does, and `diarized_json` in seconds, as its segments do. speech.cpp's own outputs (the C API,
  `speech diarize`, the text of `speech asr` and its JSON of files) keep the model's numbers from 0 and seconds.
- **A session labels its own speakers, and a diarization that takes another's place goes on after every label the ones
  before it gave**, rather than from `A` again: the same person may get a second label, which says what the model knows,
  where a label given again could name two people. Two sessions at once label theirs apart.
- **`diarized_json` needs `chunking_strategy` and gives a segment for each of its regions**, diarized in the `offline`
  mode on the whole file: the segment's `speaker` is the one who speaks the most in it, or `""` where no one does, and
  speech.cpp's own `speakers` gives every speaker's parts of it. `known_speaker_names[]` and `known_speaker_references[]`
  are refused, since the model knows no voices. `speech asr --diarize` gives each region of `--vad` its speakers in the
  same way, and for a stream the same events as the server.

The alternatives were turned down:

- A separate event of speech.cpp's own after the completed, with the speakers. A client would hold each text until its
  speakers come, or show it twice, for an event that comes only a little sooner.
- Diarization for every session of a server that holds a model. A session of one person's microphone would pay the wait
  for nothing, and would have no way to say so.
- Speakers as numbers, or times in seconds, in the Realtime events. One server would write the same speaker two ways,
  and the events' other times are milliseconds.
- Each new diarization labeling from `A`. A client could not tell the `A` before a change from the `A` after it.
- Segments of `diarized_json` cut at each change of speaker. Their text would have to be cut by times the recognizers
  do not give; a region is what the recognizer hears alone.
- `diarized_json` without `chunking_strategy` as one segment of the whole file. A segment of a whole meeting with one
  speaker says nothing, and OpenAI's own model requires `chunking_strategy` past 30 s.

## Consequences

A session with diarization gets each completed later than one without, by the time the diarization still needs after
the recognition. A client that reads the speakers of `speech asr -` from its text gets numbers, and from its JSON labels.
