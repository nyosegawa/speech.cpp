// Audio the user gives: a file in any format the browser decodes, or a recording from the microphone, made one channel
// at the model's rate before it goes to the server.

import { wavFile } from './wav.js';

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

/** A WAVE file of the audio in a file's bytes, one channel at `rate`. */
export async function toWav(bytes, rate) {
  return wavFile(await decode(bytes, rate), rate);
}

/** Records the microphone as it is, without the echo cancellation, noise suppression and gain control of calls. */
export class Recorder {
  #stream = null;
  #context = null;
  #chunks = [];

  async start() {
    this.#stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    this.#context = new AudioContext();
    await this.#context.audioWorklet.addModule('/page/recorder-worklet.js');
    const node = new AudioWorkletNode(this.#context, 'recorder');
    this.#chunks = [];
    node.port.onmessage = (event) => this.#chunks.push(event.data);
    this.#context.createMediaStreamSource(this.#stream).connect(node);
    // A node with no path to the destination is not run by every browser; the recorder's output is silence.
    node.connect(this.#context.destination);
  }

  /** The seconds recorded so far. */
  get seconds() {
    return this.#context ? this.#chunks.reduce((n, c) => n + c.length, 0) / this.#context.sampleRate : 0;
  }

  /** Stops, and returns the recording as a WAVE file at the microphone's rate. */
  async stop() {
    this.#stream?.getTracks().forEach((track) => track.stop());
    const rate = this.#context.sampleRate;
    await this.#context.close();
    const samples = new Float32Array(this.#chunks.reduce((n, c) => n + c.length, 0));
    let at = 0;
    for (const chunk of this.#chunks) {
      samples.set(chunk, at);
      at += chunk.length;
    }
    this.#stream = this.#context = null;
    this.#chunks = [];
    return wavFile(samples, rate);
  }
}
