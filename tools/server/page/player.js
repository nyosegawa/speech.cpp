// Plays speech as the server makes it: each chunk of 16-bit samples is scheduled to start where the one before it ends,
// so that the audio starts with the first chunk and plays without gaps while the chunks come faster than it plays.

import { wavFile } from './wav.js';

/** How far ahead of now a chunk that arrives after the one before it has ended is scheduled, in seconds. */
const LEAD = 0.08;

export class Player {
  #context = null;
  #sources = new Set();
  #rate = 0;
  #end = 0;
  #chunks = [];

  /** Begins a speech at `rate`, stopping the one before. */
  async begin(rate) {
    this.stop();
    this.#context ??= new AudioContext();
    await this.#context.resume();
    this.#rate = rate;
    this.#end = 0;
    this.#chunks = [];
  }

  push(samples) {
    this.#chunks.push(samples);
    const buffer = this.#context.createBuffer(1, samples.length, this.#rate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i++) channel[i] = samples[i] / 32768;
    const source = this.#context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.#context.destination);
    const at = Math.max(this.#end, this.#context.currentTime + LEAD);
    source.start(at);
    this.#end = at + buffer.duration;
    this.#sources.add(source);
    source.onended = () => this.#sources.delete(source);
  }

  stop() {
    for (const source of this.#sources) source.stop();
    this.#sources.clear();
  }

  /** The seconds of speech received. */
  get seconds() {
    return this.#rate ? this.#chunks.reduce((n, c) => n + c.length, 0) / this.#rate : 0;
  }

  /** The speech received as a WAVE file, or null before any. */
  wav() {
    if (!this.#chunks.length) return null;
    const samples = new Int16Array(this.#chunks.reduce((n, c) => n + c.length, 0));
    let at = 0;
    for (const chunk of this.#chunks) {
      samples.set(chunk, at);
      at += chunk.length;
    }
    return wavFile(samples, this.#rate);
  }
}
