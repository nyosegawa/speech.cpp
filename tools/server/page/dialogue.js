// A conversation with speech serve's dialogue model over /v1/dialogue: the page sends the microphone as 16-bit PCM at
// the model's rate in binary messages, and the server answers each frame it hears with a frame of the model's audio in a
// binary message, the text the model began to say in it going before it as a `text` event. The frames play as they
// come, each after the one before; a frame computed late plays late, and a conversation that falls more than a second
// behind stops, as speech talk --live does.

import { ServerError } from './api.js';
import { floats, pcm16 } from './wav.js';

/** How far a conversation may fall behind, in seconds of audio sent and not yet answered and of audio not yet played. */
const BEHIND_SECONDS = 1;
/** How far ahead of now a frame that arrives after the one before it has ended is scheduled, in seconds. */
const LEAD = 0.05;

export class Dialogue {
  #socket = null;
  #context = null;
  #rate = 0;
  #frame = 0;
  #sent = 0;
  #answered = 0;
  #end = 0;
  #text = '';
  #ended = null;
  #failed;
  #said;
  #closing = false;
  #started = false;

  /**
   * `said(samples, rate, time, text)` is called with each frame of the model as it is scheduled to play at `time` on
   * the clock of performance.now() in seconds, and `failed(e)` once, on the first failure after the conversation started,
   * after which it has ended.
   */
  constructor(said, failed) {
    this.#said = said;
    this.#failed = failed;
  }

  /** The model's rate, at which push() takes the microphone's samples. */
  get rate() {
    return this.#rate;
  }

  /**
   * Starts a conversation with the request's `options`; it resolves with the model's rate once the server has run the
   * prompt and sent the model's first frame.
   */
  async open(options) {
    this.#context = new AudioContext();
    await this.#context.resume();
    const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${scheme}//${location.host}/v1/dialogue`);
    socket.binaryType = 'arraybuffer';
    this.#socket = socket;
    let opened = false;
    const started = new Promise((resolve, reject) => {
      this.#ended = { resolve: () => {}, reject };
      socket.addEventListener('message', (event) => {
        if (typeof event.data !== 'string') {
          this.#play(floats(event.data));
          return;
        }
        const message = JSON.parse(event.data);
        if (message.type === 'started') {
          this.#rate = message.sample_rate;
          this.#frame = message.frame_samples;
          this.#started = true;
          resolve(this.#rate);
        } else if (message.type === 'text') {
          this.#text = message.text;
        } else if (message.type === 'end') {
          this.#closing = true;
          this.#ended.resolve(message);
        } else if (message.type === 'error') {
          this.#fail(new ServerError(400, message.error));
        }
      });
    });
    socket.addEventListener('close', () => {
      if (this.#closing) return;
      this.#fail(new Error(opened ? 'The server closed the conversation.' : 'The server refused the conversation, as when its dialogue model is being replaced or holds another conversation.'));
    });
    socket.addEventListener('open', () => {
      opened = true;
      socket.send(JSON.stringify({ type: 'start', ...options }));
    });
    return started;
  }

  /** Sends the microphone's samples at the model's rate, and stops the conversation once it has fallen behind. */
  push(samples) {
    if (!this.#socket || this.#socket.readyState !== WebSocket.OPEN) return;
    this.#sent += samples.length;
    // The first frame answers the prompt, and each frame after it a frame heard.
    const unanswered = this.#sent - Math.max(0, this.#answered - 1) * this.#frame;
    const behind = unanswered / this.#rate + Math.max(0, this.#end - this.#context.currentTime);
    if (behind > BEHIND_SECONDS) {
      this.#fail(new Error(`The conversation has fallen ${behind.toFixed(1)} s behind the microphone, as the model computes its frames ` +
        'slower than they last on this device. Use a smaller type of the model, or a faster device.'));
      return;
    }
    this.#socket.send(pcm16(samples));
  }

  /** Ends the conversation: the server answers the audio sent and closes; resolves once it has. */
  async hangUp() {
    if (!this.#socket || this.#socket.readyState !== WebSocket.OPEN) return;
    const ended = new Promise((resolve, reject) => (this.#ended = { resolve, reject }));
    this.#socket.send(JSON.stringify({ type: 'hang_up' }));
    await ended;
    this.#release();
  }

  /** Ends the conversation at once, without waiting for the server. */
  abort() {
    this.#closing = true;
    this.#release();
  }

  #play(samples) {
    this.#answered++;
    const buffer = this.#context.createBuffer(1, samples.length, this.#rate);
    buffer.copyToChannel(samples, 0);
    const source = this.#context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.#context.destination);
    const at = Math.max(this.#end, this.#context.currentTime + LEAD);
    source.start(at);
    this.#end = at + buffer.duration;
    // The view's clock is performance.now(); the context's time maps to it by their difference now.
    const time = performance.now() / 1000 + (at - this.#context.currentTime);
    this.#said(samples, this.#rate, time, this.#text);
    this.#text = '';
  }

  #fail(e) {
    if (this.#closing) return;
    this.#closing = true;
    this.#release();
    this.#ended?.reject(e);
    // A failure before the conversation started rejects open() alone.
    if (this.#started) this.#failed(e);
  }

  #release() {
    this.#socket?.close();
    this.#socket = null;
    this.#context?.close();
    this.#context = null;
  }
}
