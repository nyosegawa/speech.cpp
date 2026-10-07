// Plays speech as the server makes it, and again afterwards. Each chunk of 16-bit samples is scheduled to start where
// the one before it ends, so that the audio starts with the first chunk and plays without gaps while the chunks come
// faster than it plays; once the speech is whole it plays from any position, and pauses.

import { wavFile } from './wav.js';

/** How far ahead of now a chunk that arrives after the one before it has ended is scheduled, in seconds. */
const LEAD = 0.08;

export class Player {
  #context = null;
  #sources = new Set();
  #rate = 0;
  #chunks = [];
  #samples = 0;
  /** The context's time at which the position 0 of the speech plays, or null while it does not play. */
  #origin = null;
  #end = 0;
  #paused = 0;
  #whole = null;
  /** Whether chunks are played as they come, until the speech is paused. */
  #live = false;

  /** Called when playing starts or stops. */
  onchange = () => {};

  /** Begins a speech at `rate`, stopping the one before, and plays its chunks as they come. */
  async begin(rate) {
    this.#halt();
    this.#context ??= new AudioContext();
    await this.#context.resume();
    this.#rate = rate;
    this.#chunks = [];
    this.#samples = 0;
    this.#whole = null;
    this.#paused = 0;
    this.#origin = null;
    this.#end = 0;
    this.#live = true;
  }

  push(samples) {
    const before = this.duration;
    this.#chunks.push(samples);
    this.#samples += samples.length;
    this.#whole = null;
    if (!this.#live) return;
    const buffer = this.#buffer([samples]);
    const at = Math.max(this.#end, this.#context.currentTime + LEAD);
    // After the chunks ran out, the speech resumes where it ended.
    if (this.#origin === null) {
      this.#origin = at - before;
      this.onchange();
    }
    this.#start(buffer, at, 0);
    this.#end = at + buffer.duration;
  }

  /** Stops playing and keeps the position; chunks that come later are kept for play(). */
  pause() {
    this.#paused = this.position;
    this.#live = false;
    this.#halt();
    this.onchange();
  }

  /** Plays the whole speech from `offset` seconds, or from where it was paused. */
  play(offset = this.#paused >= this.duration ? 0 : this.#paused) {
    this.#live = false;
    this.#halt();
    this.#whole ??= this.#buffer(this.#chunks);
    this.#origin = this.#context.currentTime - offset;
    this.#start(this.#whole, this.#context.currentTime, offset);
    this.#end = this.#origin + this.duration;
    this.onchange();
  }

  /** Moves to `offset` seconds, playing on from there if it was playing. */
  seek(offset) {
    if (this.playing) this.play(offset);
    else this.#paused = offset;
  }

  get playing() {
    return this.#origin !== null;
  }

  /** The seconds of speech received. */
  get duration() {
    return this.#rate ? this.#samples / this.#rate : 0;
  }

  /** The position that plays now, in seconds. */
  get position() {
    if (this.#origin === null) return this.#paused;
    return Math.max(0, Math.min(this.duration, this.#context.currentTime - this.#origin));
  }

  /** The samples received, in order. */
  get chunks() {
    return this.#chunks;
  }

  /** The speech received as a WAVE file, or null before any. */
  wav() {
    if (!this.#samples) return null;
    const samples = new Int16Array(this.#samples);
    let at = 0;
    for (const chunk of this.#chunks) {
      samples.set(chunk, at);
      at += chunk.length;
    }
    return wavFile(samples, this.#rate);
  }

  #buffer(chunks) {
    const length = chunks.reduce((n, c) => n + c.length, 0);
    const buffer = this.#context.createBuffer(1, length, this.#rate);
    const channel = buffer.getChannelData(0);
    let at = 0;
    for (const chunk of chunks) {
      for (let i = 0; i < chunk.length; i++) channel[at + i] = chunk[i] / 32768;
      at += chunk.length;
    }
    return buffer;
  }

  #start(buffer, at, offset) {
    const source = this.#context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.#context.destination);
    source.start(at, offset);
    this.#sources.add(source);
    source.onended = () => {
      this.#sources.delete(source);
      if (this.#sources.size === 0 && this.#origin !== null) {
        this.#paused = this.duration;
        this.#origin = null;
        this.onchange();
      }
    };
  }

  #halt() {
    const sources = [...this.#sources];
    this.#sources.clear();
    this.#origin = null;
    for (const source of sources) source.stop();
  }
}
