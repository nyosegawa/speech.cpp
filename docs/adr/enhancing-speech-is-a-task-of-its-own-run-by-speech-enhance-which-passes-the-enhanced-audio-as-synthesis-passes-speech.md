# Enhancing speech is a task of its own, run by speech_enhance(), which passes the enhanced audio as synthesis passes speech

## Context

NVIDIA NemotronLabs RE-USE (OpenMDW-1.1) takes a recording and gives the same recording with its noise, reverberation,
clipping, band limits and codec artifacts removed, at the rate it was given. The official code enhances a file at once
(`inference.py`) or in chunks of 5 s that overlap by half (`inference_chunk.py`), and can resample the audio before it
enhances it (`--BWE`).

The C API had four tasks: synthesis, which passes the audio it makes to a callback as it makes it, and recognition,
detection and diarization, which take a request's audio and give text or segments in the result. The worker served
synthesis and recognition; `speech serve` held one model of each task and spoke OpenAI's audio API, which has no
endpoint for enhancement.

Enhancement in front of a recognizer does not help on every input. On ReazonSpeech's test set (5,263 utterances of
Japanese television, 2026-10-31), RE-USE's official code raised the CER of Qwen3-ASR 1.7B from 25.63% to 27.20% and of
ReazonSpeech v2 from 4.69% to 8.36%, and removed the voice of a few short replies.

RE-USE looks ahead: its Mamba runs backward as well as forward over a chunk, and its instance norms take a chunk's whole
statistics, so a chunk's output waits for the chunk's last sample. With the official chunks of 5 s a sample of audio that
arrives as it is spoken is settled 5 to 7.5 s after it arrives. OpenAI's Realtime session names noise reduction of its
input `noise_reduction`, applied before its voice activity detection; its transcription endpoint has no such field.

## Decision

- **Enhancement is a fifth task**, `SPEECH_TASK_ENHANCEMENT`, run by `speech_enhance()` on a request's audio, set with
  `speech_request_set_audio()` as for the other tasks that take audio. It passes the enhanced audio to a
  `speech_audio_callback` as synthesis does, each piece once no later chunk changes it, so that a caller writes or plays
  it as it comes; the result gives the samples passed and the stop reason, as a synthesis's does. The calls of the other
  tasks are `unsupported` on an enhancement model, and `speech_enhance()` on the others.
- **The audio comes out at the rate it is enhanced at.** An enhancement model's information has the sample rate 0, and
  its file no `speech.sample_rate`: RE-USE enhances audio at its own rate, anywhere in the range of rates it takes (8 to
  48 kHz), refusing a rate outside it unless the request names a rate to resample to, and the range is that of the
  option that names it.
- **Audio that arrives a piece at a time is enhanced through `speech_enhancement`** (start, push, end), as a detection
  and a diarization take such audio: the official chunks are computed as their audio arrives, and the samples are
  `speech_enhance()`'s for the whole audio bit for bit, each passed once no later chunk can reach it. A caller trades
  latency for context through `chunk_size_in_seconds`; nothing outside the official chunks shortens it.
- **Recognition enhances only when asked, and the tools join the two models**, as they join a detection model's regions
  and a diarization model's speakers to recognitions; the library keeps one task to a request. `speech asr --enhance
  MODEL` enhances each file whole before its regions are found and recognized. `speech serve` holding an enhancement model
  takes `noise_reduction` (`near_field` or `far_field`, both through the model it holds): in a Realtime session under
  OpenAI's name, and in a transcription's form under the same name as speech.cpp's own field. Off is the default
  everywhere.
- **Live transcription enhances each utterance whole** once its region has ended, before the utterance's last reading,
  so that the enhancement adds its computation to an utterance's text and no wait for later audio; the voice activity
  detection hears the audio as it arrives, and the readings of an utterance under way hear it unenhanced. This departs
  from OpenAI's order, which the server's page says.
- **`speech enhance` is the command line's way in**, writing 16-bit WAVE files as the audio comes. The worker takes an
  enhancement request as a recognition request's `chunk` lines followed by an `enhance` line, and answers as a synthesis
  request does, with `chunk`s of audio and an `end` that gives the samples and the stop reason. `speech serve` holds an
  enhancement model beside the others and serves `POST /v1/audio/enhancements`, whose request is a transcription's form
  (`file`, `model`, the options by name) and whose response is a speech response's audio (`wav` or `pcm`, with
  `X-Sample-Rate`). Its request option flags take the prefix `enhance.`.

The alternatives were turned down:

- Enhancement as an option of recognition in the C API. A request runs on one model and a family reads one file, and the
  measurement above shows that a recognizer can lose more than it gains from it; a caller decides per input.
- Enhancement on by default, or chosen by the tools from the audio. A recognizer loses on audio like ReazonSpeech's, and
  nothing in speech.cpp tells such audio from a crowd's.
- Live audio enhanced as it arrives, before the voice activity detection, as OpenAI orders it. The official chunks
  settle a sample 5 to 7.5 s late, and chunks short enough for a conversation hear a second of context, which RE-USE was
  not evaluated with; an utterance whole hears all of its own.
- A stream that keeps a past context and settles each sample after a short look-ahead. It is no inference of the
  official code, and its output would be checked against nothing.
- Enhanced audio in the result, read once the request has ended. A long recording would be held twice in memory, and a
  caller could not write or play the first audio until the last chunk was done.
- A sample rate for RE-USE's information, such as 16 or 48 kHz. Every input would be resampled to it, so audio at the
  other rates the model takes would lose its band or gain a resampling it does not need.
- An accessor of the range of rates of its own. The option that resamples has that range already, which
  speech_model_info_option_range() gives, and one fact would have two places.
- An endpoint path of speech.cpp's own beside OpenAI's, such as `/enhance`. The server's other endpoints sit under
  `/v1/audio/` with a plural noun, as `transcriptions` and `translations` do.

## Consequences

The model pages say on which inputs enhancement helps recognition and on which it hurts, with the measurements, and how
the accuracy of shorter chunks compares, for a caller that streams.
