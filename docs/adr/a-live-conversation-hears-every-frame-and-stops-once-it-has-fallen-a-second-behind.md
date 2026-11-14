# A live conversation hears every frame and stops once it has fallen a second behind

## Context

A dialogue model gives a frame of its audio for each frame it hears, and the library computes a frame once its audio
has arrived, keeping no time of its own ([a-full-duplex-conversation-is-a-task-of-its-own-whose-object-gives-a-frame-for-each-frame-it-hears.md](a-full-duplex-conversation-is-a-task-of-its-own-whose-object-gives-a-frame-for-each-frame-it-hears.md)).
A program that holds a live conversation, `speech talk --live` or the talk panel of `speech serve`'s page, takes the
microphone's audio as it comes and plays the model's frames as they come, and a frame computed in longer than it lasts leaves it behind: Haru's frame lasts 133 ms,
and its step took 69 ms at p50 and 70 ms at p99 in Q8_0 on Metal on an Apple M5, 36 ms in Q8_0 on Vulkan on an RTX 2080,
183 ms in F32 on the M5 and 317 ms in Q8_0 on the CPU of an i9-9900K (2026-11-14). On the RTX 2080, with other programs
on its GPU, some steps took 0.2 to 2.7 s. The official client plays silence while a frame is late and, once the delay
has grown, shortens its audio in time with WSOLA to catch up.

## Decision

- **Every frame the microphone gives is heard, in order.** Nothing is dropped and nothing is compressed in time.
- **The audio queued to play absorbs a late frame.** A frame computed in longer than it lasts plays late, and the frames
  computed after it, while the model catches up, wait in the queue.
- **The delay is what waits**: the microphone's audio not yet given to the model, which on the page is the audio sent
  and not yet answered, and the model's audio not yet played.
  Once it passes 1 s, about seven frames, the conversation stops with `device`, saying that the device computes the
  frames slower than they last and that a smaller type or a faster device keeps up.
- **The worker and `speech serve` keep no time.** Neither knows whether its caller plays the frames live, so each
  computes a frame once its audio has arrived and leaves the judgement to the caller.

The alternatives were turned down:

- Dropping the user's frames to catch up. The model would answer audio it never heard, a change behind the caller's
  back.
- Shortening the model's audio in time, as the official client does. It changes what the model said, and hides a
  device too slow for the model behind audio that sounds hurried.
- Letting the delay grow without a bound. Each late frame would push every later reply further from what it answers,
  and a conversation on a device that cannot keep up would never say so.

## Consequences

- A live conversation on a device that keeps up absorbs single slow steps up to a second; a run of them stops it with
  an error that names the device's speed.
- A conversation on a WAVE file has no clock and gives the same frames as a live one that hears the same audio.
