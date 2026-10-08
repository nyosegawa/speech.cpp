// Transcribes a recording while it is made. The recording is cut into pieces as it grows; each finished piece is
// transcribed once, and the piece still being recorded is transcribed again every few seconds, its text shown as
// provisional until its piece is finished. The server holds no state of it: every request is a WAVE file of one piece.

import * as api from './api.js';
import { PieceCutter, Transcript } from './pieces.js';
import { wavFile } from './wav.js';

/** The shortest unfinished piece worth transcribing; Qwen3-ASR pads anything shorter with silence. */
const MIN_PENDING_SECONDS = 0.5;

export class LiveTranscription {
  #rate;
  #cutter;
  #every;
  #fields;
  #show;
  #transcript = new Transcript();
  #finished = [];
  #stopped = false;
  #controller = new AbortController();
  /** The request for the unfinished piece, which the end of the recording makes pointless. */
  #provisional = null;
  #wake = null;
  #lastStart = -Infinity;
  #lastTook = 0;

  /**
   * `pieceSeconds` bounds each piece and `everySeconds` the time between two transcriptions of the unfinished one,
   * which is at least twice the time the last request took. `fields()` gives the form's fields when a request is
   * sent, and `show(transcript, provisional)` shows the text so far.
   */
  constructor(rate, { pieceSeconds, everySeconds, fields, show }) {
    this.#rate = rate;
    this.#cutter = new PieceCutter(rate, pieceSeconds);
    this.#every = everySeconds;
    this.#fields = fields;
    this.#show = show;
  }

  /** Adds the samples just recorded. */
  push(samples) {
    const finished = this.#cutter.push(samples);
    if (!finished.length) return;
    this.#finished.push(...finished);
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
      const wait = this.#lastStart + 1000 * Math.max(this.#every, 2 * this.#lastTook) - performance.now();
      if (wait > 0 || this.#cutter.pendingSeconds < MIN_PENDING_SECONDS) {
        await this.#sleep(Math.max(wait, 100));
        continue;
      }
      await this.#transcribePending();
    }
  }

  async #transcribePending() {
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

  async #transcribe(samples, signal) {
    this.#lastStart = performance.now();
    const answer = await api.transcribe(wavFile(samples, this.#rate), this.#fields(), signal);
    this.#lastTook = (performance.now() - this.#lastStart) / 1000;
    return answer;
  }

  #sleep(ms) {
    return new Promise((resolve) => {
      const done = () => {
        clearTimeout(timer);
        this.#wake = null;
        resolve();
      };
      const timer = setTimeout(done, ms);
      this.#wake = done;
    });
  }

  #wakeUp() {
    this.#wake?.();
  }
}
