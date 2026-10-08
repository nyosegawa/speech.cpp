// Audio the user gives: a file in any format the browser decodes, or a recording from the microphone, made one channel
// at the model's rate before it goes to the server.

import { pcm16, wavHeader } from './wav.js';

/** One channel at `rate` of the audio in a file's bytes, as the browser decodes and resamples it. */
export async function decode(bytes, rate) {
  // An offline context decodes at its own rate, so the browser resamples while it decodes.
  const context = new OfflineAudioContext(1, 1, rate);
  let audio;
  try {
    audio = await context.decodeAudioData(bytes);
  } catch {
    throw new Error('The browser cannot decode this file as audio. Give a WAVE, MP3, AAC, FLAC or Ogg file.');
  }
  const samples = new Float32Array(audio.length);
  for (let c = 0; c < audio.numberOfChannels; c++) {
    const channel = audio.getChannelData(c);
    for (let i = 0; i < samples.length; i++) samples[i] += channel[i] / audio.numberOfChannels;
  }
  return samples;
}

/** Records the microphone as it is, without the echo cancellation, noise suppression and gain control of calls. */
export class Recorder {
  #stream = null;
  #context = null;
  #chunks = [];

  #rate;

  /** A recorder at `rate`, which the browser resamples the microphone to, or at the microphone's own rate. */
  constructor(rate = undefined) {
    this.#rate = rate;
  }

  /** Starts recording; a failure lets go of what it took, so that the microphone is not left on without a recorder. */
  async start() {
    this.#stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    try {
      this.#context = new AudioContext(this.#rate ? { sampleRate: this.#rate } : {});
      await this.#context.audioWorklet.addModule('/page/recorder-worklet.js');
      const node = new AudioWorkletNode(this.#context, 'recorder');
      this.#chunks = [];
      node.port.onmessage = (event) => this.#chunks.push(event.data);
      this.#context.createMediaStreamSource(this.#stream).connect(node);
      // A node with no path to the destination is not run by every browser; the recorder's output is silence.
      node.connect(this.#context.destination);
    } catch (e) {
      await this.#release();
      throw e;
    }
  }

  /** The rate at which every sample is recorded. */
  get rate() {
    return this.#context.sampleRate;
  }

  /** Takes the samples recorded since the last time they were taken. */
  take() {
    const samples = new Float32Array(this.#chunks.reduce((n, c) => n + c.length, 0));
    let at = 0;
    for (const chunk of this.#chunks) {
      samples.set(chunk, at);
      at += chunk.length;
    }
    this.#chunks = [];
    return samples;
  }

  /** Stops, and returns the samples recorded since they were last taken. */
  async stop() {
    const rest = this.take();
    await this.#release();
    return rest;
  }

  /** Turns the microphone off and closes the audio context, whichever of them was taken. */
  async #release() {
    this.#stream?.getTracks().forEach((track) => track.stop());
    await this.#context?.close();
    this.#stream = this.#context = null;
  }
}

/** The seconds of 16-bit samples kept in each of a recording's blobs. */
const PART_SECONDS = 10;

/**
 * A recording as a WAVE file of 16-bit samples, kept in blobs of ten seconds, which the browser may hold on disk, so
 * that a recording of an hour or more does not fill the page's memory. The samples come in any amounts.
 */
export class RecordingFile {
  #rate;
  #parts = [];
  #part;
  #length = 0;
  #count = 0;

  constructor(rate) {
    this.#rate = rate;
    this.#part = new Float32Array(PART_SECONDS * rate);
  }

  get rate() {
    return this.#rate;
  }

  /** The samples recorded. */
  get count() {
    return this.#count;
  }

  get seconds() {
    return this.#count / this.#rate;
  }

  append(samples) {
    for (let at = 0; at < samples.length; ) {
      const n = Math.min(samples.length - at, this.#part.length - this.#length);
      this.#part.set(samples.subarray(at, at + n), this.#length);
      this.#length += n;
      this.#count += n;
      at += n;
      if (this.#length === this.#part.length) this.#flush();
    }
  }

  /** The recording so far as a WAVE file. */
  blob() {
    this.#flush();
    return new Blob([wavHeader(this.#count, this.#rate), ...this.#parts], { type: 'audio/wav' });
  }

  #flush() {
    if (!this.#length) return;
    this.#parts.push(new Blob([pcm16(this.#part.subarray(0, this.#length))]));
    this.#length = 0;
  }
}
