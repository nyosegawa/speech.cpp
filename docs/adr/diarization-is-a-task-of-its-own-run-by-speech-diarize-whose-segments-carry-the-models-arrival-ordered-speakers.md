# Diarization is a task of its own, run by speech_diarize(), whose segments carry the model's arrival-ordered speakers

## Context

A program that transcribes a meeting, live or after it, wants to know who spoke when. Nemotron 3 Diarization gives,
every 10 ms, the probability that each of eight speakers speaks, its channels ordered by the speakers' first arrival
(the model card's "speaker ordering"), and NeMo's diarize() turns them into segments of each speaker with its
postprocessing (onset and offset thresholds, padding, short segments dropped and short pauses filled). Its inference is
always streaming: the audio is cut into chunks, and each step hears the chunk after a speaker cache and a FIFO queue
that the steps before it filled, so that a speaker keeps its channel for as long as the audio lasts. NeMo's offline
diarization is the same streaming with chunks of 27.2 s.

The C API had three tasks: synthesis, recognition and detection, whose regions are the segments of its result, without
text, in order and apart, and a detection of audio given a piece at a time (`speech_detection`) that gives
speech_detect()'s regions, each once it is certain
([its record](a-detection-of-audio-given-a-piece-at-a-time-gives-speech-detects-regions-each-once-it-is-certain.md)).
A diarization's segments overlap where two speakers speak at once.

## Decision

- **Diarization is a fourth task**, `SPEECH_TASK_DIARIZATION`, run by speech_diarize() on a request's audio, which is
  resampled to the model's rate as a recognition's is. The result's segments are its segments, each with its start and
  end, rounded to 10 ms as diarize() rounds them, the text "" and a speaker, which speech_result_segment_speaker()
  gives. They are ordered by their starts and, at the same start, by their speakers; segments of different speakers
  overlap where they speak at once. The model's information gives the most speakers it tells apart
  (speech_model_info_speakers()). The calls of the other tasks are `unsupported` on a diarization model, and
  speech_diarize() on the others.
- **A speaker is the model's channel**, numbered from 0, which the model orders by first arrival and keeps for the whole
  request; the number does not name a person. The library does not renumber the channels.
- **A diarization of audio given a piece at a time** (`speech_diarization`) is started from a request's options and the
  rate of the audio to come, as a detection is, and keeps its own speaker cache and FIFO queue, so that a speaker keeps
  its number in it and two diarizations on one model number their speakers each on its own. Each step runs once its
  chunk's audio and its right context's have arrived, and the probabilities every 10 ms, the steps and the segments are
  those of the whole audio, however it is cut into pieces: once the audio has ended, the segments are speech_diarize()'s
  with the same options. A push holds the model as a request does.
- **A segment is given once it is final**, and the diarization says up to when everything is final
  (speech_diarization_settled()): every segment that ends before that time has been given, and a segment that began
  before it and goes on keeps the start speech_diarization_speaking() gives. That time trails the audio by the
  streaming mode's latency, and by `pad_onset`, `min_duration_on` and `min_duration_off`, the reach of what later audio
  can still join, fill or keep.
- **A diarization model takes no language**: its file has neither `general.languages` nor `speech.language_use`, and
  it takes the language `auto` alone. The worker refuses it when it starts, as it refuses a detection model.
- **`speech diarize` is the command line's way in**, writing text, JSON or RTTM.

The alternatives were turned down:

- Diarization as a detection whose regions carry a speaker. A detection's regions are apart and in order, and its
  options are Silero VAD's; a diarization's segments overlap, and a caller choosing a model by its task would take one
  for the other.
- Speakers as an option of recognition. A request runs on one model and a family reads one file; the recognizers here
  take no speaker activity.
- Accessors of the result's own for a diarization's segments. A segment of a diarization is a segment without text, and
  only its speaker is new.
- Speakers renumbered by their first segment. The channels are already in order of arrival, and the numbers would differ
  from NeMo's `speaker_<k>` for the same audio.
- A stream that gives a segment only once it has ended. A speaker who speaks through several utterances of a live
  transcription would have no segment for any of them until it stopped.
- The probabilities every 10 ms through the C API, as diarize()'s include_tensor_outputs. The segments are what
  diarize() returns and what a caller acts on; the checks read the probabilities from the family.

## Consequences

A caller that wants who said what joins a diarization's segments with a recognition's utterances in the tools, as
transcription by regions joins a detection's regions with recognitions.
