# speech serve holds one conversation at a time over its own WebSocket at /v1/dialogue

## Context

A dialogue model listens and speaks at once: each frame it hears gets a frame of its own audio and the text it began
to say in it
([a-full-duplex-conversation-is-a-task-of-its-own-whose-object-gives-a-frame-for-each-frame-it-hears.md](a-full-duplex-conversation-is-a-task-of-its-own-whose-object-gives-a-frame-for-each-frame-it-hears.md)).
Programs that speak HTTP, the page of `speech serve` first, need to hold a conversation with the model the server
holds, and `speech serve` speaks OpenAI's API wherever OpenAI has the same thing. OpenAI's Realtime sessions that speak
(`session.type` `realtime`) run turns: the server finds the end of the user's turn, a response follows it and ends with
`response.done`, and `input_audio_buffer.speech_started` tells a client to stop the response it plays. Haru's frame lasts
133 ms and took 72 ms at p50 in Q8_0 on Metal on an Apple M5 and 36 ms on Vulkan on an RTX 2080 (2026-11-15), and a
conversation holds the memory of its `max_seconds` from its start, 1.7 GB for Haru's 60 minutes.

## Decision

- **A WebSocket of speech.cpp's own at /v1/dialogue.** The client's first message is `{"type": "start"}` with the
  rate of the audio it sends and the request options as members, as the worker's `dialogue` message has them; its audio
  goes in binary messages of 16-bit PCM, and `{"type": "hang_up"}` ends the conversation. The server sends `started`
  with the model's rate and frame once the prompt has run, each frame of the model's audio as a binary message of 16-bit
  PCM with a `text` event before it where the model began to say something, `end` after a hang-up, and an `error` event
  with OpenAI's error object, after which it closes.
- **One conversation at a time.** Another is refused with a 409 (`conversation_in_progress`) until the first has
  ended, and the place is freed with the conversation's memory before the client hears `end`.
- **The server keeps no time**, as the worker does
  ([a-live-conversation-hears-every-frame-and-stops-once-it-has-fallen-a-second-behind.md](a-live-conversation-hears-every-frame-and-stops-once-it-has-fallen-a-second-behind.md)):
  it computes each frame once its audio has arrived, and the client that plays the frames judges whether they keep up.

The alternatives were turned down:

- OpenAI's Realtime session that speaks. A full-duplex model has no turns, so its frames would be fitted into responses
  the model does not make, and an OpenAI client would stop playing the model's audio whenever the user spoke, which the
  model hears and answers on its own.
- The audio in base64 in JSON events, as the worker's protocol carries it. A WebSocket carries binary messages, which
  take a third fewer bytes and no encoding, and the page plays them as they come.
- Compressed audio such as Opus. The page reaches the server on the same machine, where 48 KB a second each way costs
  nothing, and the browser would need an encoder and the server a decoder.
- Several conversations at once on one model. The frames of all of them are computed on one device, so two in Q8_0 on
  an M5 take 144 ms for each 133 ms frame of both and fall behind, and each holds its memory.

## Consequences

- OpenAI's clients do not speak /v1/dialogue; a program speaks it from its description in docs/server.md.
- The page replaces the dialogue model only after its own conversation has ended, and a conversation of another client
  holds the model until it ends.
