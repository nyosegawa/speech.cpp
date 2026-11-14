# A full-duplex conversation is a task of its own, whose object gives a frame for each frame it hears

## Context

higurashi-ai's Haru-4B (Apache-2.0) listens and speaks at once. Every 133 ms it takes a frame of what it hears, 3,200
samples at 24 kHz, and gives a frame of its own audio and the text it began to say in it; the official `DuplexSession`
runs the prompt (a character card and a voice reference) at `start()`, which gives the first frame, and stays one frame
ahead of the audio it hears, so that the frame's length is the only delay it adds. A conversation lasts up to an hour in
the official code. More models of this kind are expected.

The C API had five tasks. Synthesis and enhancement pass audio to a callback; recognition, detection and diarization
give results; detection, diarization and enhancement also take audio that arrives a piece at a time through objects
that are started, pushed and ended. None gives output for input in step, and none has turns.

## Decision

- **Dialogue is a sixth task**, `SPEECH_TASK_DIALOGUE`, held through `speech_dialogue` (start, push, end), as the
  objects of audio that arrives are. A dialogue model takes no request of its own: a request carries a conversation's
  options, and nothing else of it is used.
- **A frame for each frame heard.** The start runs the prompt and passes the model's first frame to a callback that
  takes its audio and its text together. A push resamples what it is given to the model's rate and passes one frame for
  each whole frame of it before it returns; the end drops the audio short of a frame. The model's frame and the frame it
  hears have the same length, which the model information gives (`speech_model_info_frame_samples()`).
- **The caller keeps time.** The library computes a frame when its audio has arrived and does not wait for a clock,
  start a thread or drop a frame. A program that plays the frames as they come and pushes the microphone's audio as it
  comes holds a conversation; one whose device is too slow sees its frames fall behind and decides what to do.
- **Each conversation holds its own state**, its keys and values, the language model's states, the codec's and the text's
  stream, and the memory its `max_seconds` takes from its start. Several may run on one model, each frame computed while
  the model is held, as a request is. A push past `max_seconds` is `out_of_range` naming it.
- **The frames are clamped to [-1, 1]**, as the official session passes them; nothing else of the audio is changed.

The alternatives were turned down:

- OpenAI's Realtime session as the C API's shape. It runs turns, a response after the user's turn, where a full-duplex
  model has none; its events would need turns the model does not make.
- A request that takes a whole recording and gives the whole reply. It holds no conversation, which is the model's use;
  a check replays recordings through the object instead.
- The library pacing the frames on a clock of its own, or dropping late frames to keep up. The library starts no
  threads and keeps no time, and a dropped frame is audio the model never heard, a change behind the caller's back; the
  caller knows its device's clock and its buffers.
- One conversation a model, its state in the engine. A server that serves several clients would load the model again for
  each, and the state of one conversation is small beside the weights.

## Consequences

- `speech talk`, the worker and `speech serve` hold conversations, the server over a WebSocket of its own
  ([speech-serve-holds-one-conversation-at-a-time-over-its-own-websocket-at-v1-dialogue.md](speech-serve-holds-one-conversation-at-a-time-over-its-own-websocket-at-v1-dialogue.md)).
- A model of another kind with frames of another length takes the same object; the model information says the length.
