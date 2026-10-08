// A transcription of the microphone over speech serve's /v1/realtime, OpenAI's Realtime transcription: the page sends the
// audio as 16-bit PCM at 24 kHz, the server's detection model finds where someone speaks (server_vad), and each utterance
// comes back as deltas while it is said, which only add to its end, then as its final text, which replaces them.

import { ServerError } from './api.js';
import { join } from './pieces.js';
import { pcm16 } from './wav.js';

/** The one rate of OpenAI's audio/pcm. */
export const RATE = 24000;

/** `buffer` in base64, a slice at a time, since a string of a whole second's bytes is too many arguments at once. */
function base64(buffer) {
  const bytes = new Uint8Array(buffer);
  let text = '';
  for (let at = 0; at < bytes.length; at += 0x8000) text += String.fromCharCode(...bytes.subarray(at, at + 0x8000));
  return btoa(text);
}

/** The error of an `error` event or of a failed transcription, in the page's form. */
function serverError(error) {
  return new ServerError(400, error);
}

export class RealtimeTranscription {
  #socket = null;
  #show;
  #failed;
  /** The utterances in the order they were heard, by item id: {text, deltas, final, committed}. */
  #items = new Map();
  #languages = [];
  #stop = 'complete';
  #seconds = 0;
  #waiting = null;
  #failure = null;

  /**
   * `show(transcript, provisional)` is called whenever the text changes, and `failed(e)` once, on the first failure while
   * the audio is being sent, after which the session ends.
   */
  constructor(show, failed) {
    this.#show = show;
    this.#failed = failed;
  }

  /**
   * Opens a session that transcribes with `model`, given the form's `fields` (language, prompt), and finds speech with
   * the server's detection model; it resolves once the server has taken the configuration.
   */
  async open(model, fields) {
    const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${scheme}//${location.host}/v1/realtime?model=${encodeURIComponent(model)}`);
    this.#socket = socket;
    const configured = new Promise((resolve, reject) => {
      this.#waiting = { resolve, reject, until: 'session.updated' };
    });
    socket.addEventListener('message', (event) => this.#receive(JSON.parse(event.data)));
    let opened = false;
    socket.addEventListener('close', () =>
      this.#fail(new Error(opened ? 'The server closed the connection.' : 'The server refused the session, as when its recognition model is being replaced.')));
    socket.addEventListener('open', () => {
      opened = true;
      const transcription = { model, ...fields };
      socket.send(JSON.stringify({
        type: 'session.update',
        session: {
          type: 'transcription',
          audio: { input: { format: { type: 'audio/pcm', rate: RATE }, transcription, turn_detection: { type: 'server_vad' } } },
        },
      }));
    });
    await configured;
  }

  /** Sends samples at RATE. */
  push(samples) {
    if (!samples.length || this.#socket?.readyState !== WebSocket.OPEN) return;
    this.#seconds += samples.length / RATE;
    this.#socket.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: base64(pcm16(samples)) }));
  }

  /**
   * Ends the session once the audio has been sent: an utterance still under way is committed as it is, and it resolves
   * with the whole text once every utterance has its final text.
   */
  async finish() {
    if (this.#failure) throw this.#failure;
    if ([...this.#items.values()].some((item) => !item.committed)) {
      this.#socket.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
    }
    if (!this.#settled()) {
      await new Promise((resolve, reject) => {
        this.#waiting = { resolve, reject, until: 'settled' };
      });
    }
    this.#close();
    return this.#transcript();
  }

  /** Ends the session without waiting for the text. */
  abort() {
    this.#close();
  }

  #close() {
    this.#waiting = null;
    // A session the page ends is not a failure.
    this.#failure ??= new Error('The session has ended.');
    if (this.#socket && this.#socket.readyState <= WebSocket.OPEN) this.#socket.close();
  }

  /** Whether every utterance heard has its final text, and none is under way. */
  #settled() {
    return [...this.#items.values()].every((item) => item.final);
  }

  #item(id) {
    if (!this.#items.has(id)) this.#items.set(id, { text: '', deltas: '', final: false, committed: false });
    return this.#items.get(id);
  }

  #receive(event) {
    switch (event.type) {
      case 'session.updated':
        if (this.#waiting?.until === 'session.updated') this.#waiting.resolve();
        return;
      case 'input_audio_buffer.speech_started':
        this.#item(event.item_id);
        break;
      case 'input_audio_buffer.committed':
        this.#item(event.item_id).committed = true;
        break;
      case 'conversation.item.input_audio_transcription.delta':
        this.#item(event.item_id).deltas += event.delta;
        break;
      case 'conversation.item.input_audio_transcription.completed': {
        const item = this.#item(event.item_id);
        Object.assign(item, { text: event.transcript, deltas: '', final: true, committed: true });
        for (const { code } of event.languages ?? []) if (this.#languages.at(-1) !== code) this.#languages.push(code);
        if (event.stop === 'model_limit') this.#stop = 'model_limit';
        break;
      }
      case 'conversation.item.input_audio_transcription.failed':
        Object.assign(this.#item(event.item_id), { deltas: '', final: true, committed: true });
        this.#fail(serverError(event.error));
        break;
      case 'error':
        this.#fail(serverError(event.error));
        return;
      default:
        return;
    }
    this.#show(this.#transcript(), this.#provisional());
    if (this.#waiting?.until === 'settled' && this.#settled()) this.#waiting.resolve();
  }

  /** Ends what waits with `e`; the first failure is the one shown. */
  #fail(e) {
    if (this.#failure) return;
    this.#failure = e;
    if (this.#waiting) this.#waiting.reject(e);
    else this.#failed(e);
    this.#waiting = null;
    this.#close();
  }

  /** The final texts so far, in the order the utterances were heard, in the shape the transcript view takes. */
  #transcript() {
    let text = '';
    for (const item of this.#items.values()) if (item.final) text = join(text, item.text);
    return { text, segments: [], languages: this.#languages, seconds: this.#seconds, stop: this.#stop };
  }

  /** The text of the utterances not yet final, as their deltas give it. */
  #provisional() {
    let text = '';
    for (const item of this.#items.values()) if (!item.final) text = join(text, item.deltas);
    return text;
  }
}
