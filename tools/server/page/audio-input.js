// A place to give audio: drop a file on it, choose one, or record the microphone. Once audio is given it shows its name,
// its length and a small player in place of the prompt to drop one. The speak panel's voice maker and the transcribe
// panel each have one.

import { decode, Recorder, RecordingFile } from './audio.js';
import { PieceCutter } from './pieces.js';
import { floats, pcmLayout, wavFile } from './wav.js';

/** How often a recording shows its length and hands its new samples on. */
const TICK_MS = 200;

function seconds(value) {
  return `${value.toFixed(1)} s`;
}

export class AudioInput {
  #file = null;
  /** Where the 16-bit samples of the audio given are, when it is a recording made here: pcmLayout's form. */
  #recorded = null;
  #recorder = null;
  #recording = null;
  #listener;
  #listening = false;
  #ticking = 0;
  #preview = null;

  /**
   * Fills `slot` from the page's audio-input template; `changed` is called whenever the audio given changes.
   * `listener`, when given, hears a recording as it is made: `started(rate)` returns whether it wants the samples,
   * `samples(samples)` gets them as they come, and `stopped()` follows the last of them.
   */
  constructor(slot, changed = () => {}, listener = null) {
    slot.append(document.querySelector('#audio-input').content.cloneNode(true));
    const $ = (selector) => slot.querySelector(selector);
    this.drop = $('.drop');
    this.empty = $('.drop-empty');
    this.chosen = $('.drop-chosen');
    this.recordingView = $('.drop-recording');
    this.recordingTime = $('.recording-time');
    this.input = $('.audio-file');
    this.choose = $('.audio-choose');
    this.record = $('.audio-record');
    this.remove = $('.audio-remove');
    this.name = $('.audio-name');
    this.audio = $('.audio-preview');
    this.error = $('.error');
    this.changed = changed;
    this.#listener = listener;

    this.choose.addEventListener('click', () => this.input.click());
    this.input.addEventListener('change', () => {
      if (this.input.files[0]) this.use(this.input.files[0], this.input.files[0].name);
      this.input.value = '';
    });
    this.record.addEventListener('click', () => (this.#recorder ? this.#stopRecording() : this.#startRecording()));
    this.remove.addEventListener('click', () => this.#clear());
    this.audio.addEventListener('loadedmetadata', () => {
      if (Number.isFinite(this.audio.duration)) this.name.textContent = `${this.label}, ${seconds(this.audio.duration)}`;
    });
    this.drop.addEventListener('dragover', (event) => {
      event.preventDefault();
      this.drop.classList.add('over');
    });
    this.drop.addEventListener('dragleave', () => this.drop.classList.remove('over'));
    this.drop.addEventListener('drop', (event) => {
      event.preventDefault();
      this.drop.classList.remove('over');
      const file = event.dataTransfer.files[0];
      if (file) this.use(file, file.name);
    });
    this.#show('empty');
  }

  /** Whether audio is given. */
  get given() {
    return this.#file !== null;
  }

  /** Whether the microphone is being recorded. */
  get recording() {
    return this.#recorder !== null;
  }

  /** The audio given as a WAVE file of one channel at `rate`. */
  async wav(rate) {
    return wavFile(await decode(await this.#file.arrayBuffer(), rate), rate);
  }

  /**
   * The audio given in pieces of at most `maxSeconds`, each {samples, rate, start, total} with its start and the whole
   * audio's length in seconds. A WAVE file of 16-bit PCM, a recording made here among them, is read ten seconds at a
   * time at its own rate, which the server resamples, so that an hour of it never sits in the page's memory; any other
   * file is decoded whole at `rate`, the model's.
   */
  async *pieces(rate, maxSeconds) {
    // Audio given while the pieces are read does not change what they are read from.
    const file = this.#file;
    const layout = this.#recorded ?? (await pcmLayout(file));
    if (layout) {
      const { rate: own, channels, offset, count } = layout;
      const cutter = new PieceCutter(own, maxSeconds);
      const frame = 2 * channels;
      for (let at = 0; at < count; at += 10 * own) {
        const end = Math.min(count, at + 10 * own);
        const samples = floats(await file.slice(offset + frame * at, offset + frame * end).arrayBuffer(), channels);
        for (const piece of cutter.push(samples)) yield { ...piece, rate: own, total: count / own };
      }
      const last = cutter.finish();
      if (last) yield { ...last, rate: own, total: count / own };
      return;
    }
    const samples = await decode(await file.arrayBuffer(), rate);
    const cutter = new PieceCutter(rate, maxSeconds);
    const total = samples.length / rate;
    // Pushed a slice at a time, so that each piece is copied out only when it is wanted.
    for (let at = 0; at < samples.length; at += rate) {
      for (const piece of cutter.push(samples.subarray(at, at + rate))) yield { ...piece, rate, total };
    }
    const last = cutter.finish();
    if (last) yield { ...last, rate, total };
  }

  /** Takes `file`, a File or a Blob of audio, shown as `label`. */
  use(file, label) {
    this.#recorded = null;
    this.#take(file, label);
  }

  /** Shows a failure of the audio given, the browser's or as the server words it. */
  fail(message) {
    this.error.textContent = message;
    this.error.hidden = false;
  }

  #take(file, label) {
    this.#file = file;
    this.label = label;
    this.error.hidden = true;
    if (this.#preview) URL.revokeObjectURL(this.#preview);
    this.#preview = URL.createObjectURL(file);
    this.audio.src = this.#preview;
    this.name.textContent = label;
    this.#show('chosen');
    this.changed();
  }

  #show(state) {
    this.empty.hidden = state !== 'empty';
    this.chosen.hidden = state !== 'chosen';
    this.recordingView.hidden = state !== 'recording';
    this.remove.hidden = state !== 'chosen';
    this.choose.textContent = state === 'chosen' ? 'Another file' : 'Choose a file';
    this.choose.disabled = state === 'recording';
    this.drop.dataset.state = state;
  }

  #clear() {
    this.#file = this.#recorded = null;
    this.audio.removeAttribute('src');
    if (this.#preview) URL.revokeObjectURL(this.#preview);
    this.#preview = null;
    this.#show('empty');
    this.changed();
  }

  async #startRecording() {
    this.error.hidden = true;
    const recorder = new Recorder();
    try {
      await recorder.start();
    } catch (e) {
      this.fail(e.name === 'NotAllowedError' ? 'The browser was not allowed to use the microphone.' : e.message);
      return;
    }
    this.#recorder = recorder;
    this.#recording = new RecordingFile(recorder.rate);
    this.#listening = this.#listener?.started(recorder.rate) ?? false;
    this.record.setAttribute('aria-pressed', 'true');
    this.record.textContent = 'Stop recording';
    this.#show('recording');
    this.changed();
    const tick = () => {
      this.#hand(recorder.take());
      this.recordingTime.textContent = `Recording, ${seconds(this.#recording.seconds)}`;
    };
    tick();
    this.#ticking = setInterval(tick, TICK_MS);
  }

  async #stopRecording() {
    clearInterval(this.#ticking);
    const recorder = this.#recorder;
    this.#recorder = null;
    this.record.setAttribute('aria-pressed', 'false');
    this.record.textContent = 'Record';
    this.#hand(await recorder.stop());
    if (this.#listening) this.#listener.stopped();
    this.#listening = false;
    const recording = this.#recording;
    this.#recording = null;
    this.#recorded = { rate: recording.rate, channels: 1, offset: 44, count: recording.count };
    this.#take(recording.blob(), 'Recording');
  }

  /** Keeps new samples of the recording, and hands them to the listener when it wants them. */
  #hand(samples) {
    if (!samples.length) return;
    this.#recording.append(samples);
    if (this.#listening) this.#listener.samples(samples);
  }
}
