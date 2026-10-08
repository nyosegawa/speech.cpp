// Transcribes a recording while it is made. The recording is cut into pieces as it grows; each finished piece is
// transcribed once, and the piece still being recorded is transcribed again as often as the model keeps up, its text
// shown as provisional until its piece is finished. A finished piece goes first: it stops the provisional request.
// The server holds no state of it: every request is a WAVE file of one piece.

import * as api from './api.js';
import { PieceCutter, Transcript } from './pieces.js';
import { wavFile } from './wav.js';

/** The shortest unfinished piece worth transcribing; Qwen3-ASR pads anything shorter with silence. */
const MIN_PENDING_SECONDS = 0.5;
/**
 * The audio that has to come in before the unfinished piece is transcribed again. Audio arrives every 0.2 s, and a
 * transcription reads the whole piece again, so a shorter step changes no text and only costs the model's time.
 */
const NEW_SECONDS = 0.2;

export class LiveTranscription {
  #rate;
  #cutter;
  #fields;
  #show;
  #transcript = new Transcript();
  #finished = [];
  #stopped = false;
  #controller = new AbortController();
  /** The request for the unfinished piece, which a finished piece or the end of the recording makes pointless. */
  #provisional = null;
  #wake = null;
  /** The samples pushed, and how many there were when the unfinished piece was last transcribed. */
  #pushed = 0;
  #pushedAtProvisional = 0;

  /**
   * `pieceSeconds` bounds each piece, `fields()` gives the form's fields when a request is sent, and
   * `show(transcript, provisional)` shows the text so far.
   */
  constructor(rate, { pieceSeconds, fields, show }) {
    this.#rate = rate;
    this.#cutter = new PieceCutter(rate, pieceSeconds);
    this.#fields = fields;
    this.#show = show;
  }

  /** Adds the samples just recorded. */
  push(samples) {
    this.#pushed += samples.length;
    const finished = this.#cutter.push(samples);
    if (finished.length) {
      this.#finished.push(...finished);
      this.#provisional?.abort();
    }
    this.#wakeUp();
  }

  /** The recording has ended; the last piece is transcribed and the run ends. */
  stop() {
    this.#stopped = true;
    this.#provisional?.abort();
    this.#wakeUp();
  }

  /** Stops every request; the run ends with an AbortError. */
  abort() {
    this.#controller.abort();
    this.#stopped = true;
    this.#wakeUp();
  }

  /** Transcribes until the recording ends and returns the transcript, or throws the first failure. */
  async run() {
    for (;;) {
      this.#controller.signal.throwIfAborted();
      if (this.#finished.length) {
        const piece = this.#finished.shift();
        const { result, stop } = await this.#transcribe(piece.samples, this.#controller.signal);
        this.#transcript.add(result, stop, piece.start);
        this.#show(this.#transcript, '');
        continue;
      }
      if (this.#stopped) {
        const last = this.#cutter.finish();
        if (!last) return this.#transcript;
        this.#finished.push(last);
        continue;
      }
      if (this.#pushed - this.#pushedAtProvisional < NEW_SECONDS * this.#rate || this.#cutter.pendingSeconds < MIN_PENDING_SECONDS) {
        await this.#sleep();
        continue;
      }
      await this.#transcribePending();
    }
  }

  async #transcribePending() {
    this.#pushedAtProvisional = this.#pushed;
    this.#provisional = new AbortController();
    const signal = AbortSignal.any([this.#controller.signal, this.#provisional.signal]);
    try {
      const { result } = await this.#transcribe(this.#cutter.pending.samples, signal);
      // A piece finished meanwhile holds what this text was of; the next one replaces it.
      if (!this.#finished.length && !this.#stopped) this.#show(this.#transcript, result.text);
    } catch (e) {
      if (!(e.name === 'AbortError' && !this.#controller.signal.aborted)) throw e;
    } finally {
      this.#provisional = null;
    }
  }

  #transcribe(samples, signal) {
    return api.transcribe(wavFile(samples, this.#rate), this.#fields(), signal);
  }

  /** Waits until samples come in or the recording ends. */
  #sleep() {
    return new Promise((resolve) => {
      this.#wake = () => {
        this.#wake = null;
        resolve();
      };
    });
  }

  #wakeUp() {
    this.#wake?.();
  }
}
