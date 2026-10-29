# A recognition tells who speaks by the option speakers, whose segments carry the model's speakers and whose text drops its tags

## Context

Nemotron 3.5 Transcribe writes `<spk:N>` before each turn when its user turn asks for speaker tags, numbering speakers
from 0 in the order they first speak; the model card's prompt for "Who said what, and when" asks for them, a prompt with
no room for context terms. A tag says who speaks from that word on, and its times come only from the CTC alignment.
The C API already gives segments with times (`speech_result_segment()`), and a diarization's segments their speakers
(`speech_result_segment_speaker()`), numbered by the model in arrival order and never renumbered.

## Decision

- **A boolean request option `speakers`**, false being neutral, which a model that tells speakers by what it writes
  takes. Nemotron asks with the card's prompt for speaker tags.
- **The tags leave the text and become segment speakers.** Each tag and the whitespace after it leave the text; a
  segment ends where a tag starts a turn, besides where a sentence ends, and carries the tag's number through
  `speech_result_segment_speaker()`, -1 for words the model wrote before its first tag. Segments keep the order of the
  text, so two speakers' times may interleave.
- **`speakers` gives segments without tokens, and `timestamps` adds the tokens**: a word, or a letter or number of a
  language written without spaces, with the spaces before it and the marks after it, so that the tokens' texts join
  into the text.
- **`speakers` with a prompt is refused, out of range naming `prompt`**, before any work.
- **`speech_result_untimed()` says why a recognition that set either option has neither tokens nor segments**, with
  its text all the same; segments are not kept without times either.
- **The tools carry them as they carry a diarization's**: the worker's `end` and `speech asr`'s JSON give each segment
  its `speaker` and the result its `untimed`, `speech asr --speakers` writes FILE, START, END, SPEAKER and TEXT, and
  refuses `--vad`, whose regions would each number their speakers from 0.

The alternatives were turned down:

- The tags left in the text, as the card's request leaves them. Every caller would parse them, and their times would
  have no place.
- Speakers as a value of `prompt` or of an instruction. The tags would need a place in the result all the same, and an
  instruction of the caller's own makes a recognition whatever it asks for.
- Segments with speakers and no times where the alignment fails. Every reader of segments, the server's `diarized_json`
  and the command line's columns among them, would meet times that are not numbers.
- A category of failure for each reason. The reasons are what the alignment could not do, which a caller reports and
  does not branch on.

## Consequences

The API's minor version rises to 3.6. A caller that wants speakers across recordings longer than the model takes, or
from a model that writes none, uses a diarization model as before.
