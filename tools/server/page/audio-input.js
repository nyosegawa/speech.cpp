// A place to give audio: drop a file on it, choose one, or record the microphone, with the audio given played back.
// The speak panel's voice maker and the transcribe panel each have one.

import { Recorder, toWav } from './audio.js';

export class AudioInput {
  #file = null;
  #recorder = null;
  #ticking = 0;
  #preview = null;

  /** Fills `slot` from the page's audio-input template; `changed` is called whenever the audio given changes. */
  constructor(slot, changed = () => {}) {
    slot.append(document.querySelector('#audio-input').content.cloneNode(true));
    const $ = (selector) => slot.querySelector(selector);
    this.drop = $('.drop');
    this.input = $('.audio-file');
    this.choose = $('.audio-choose');
    this.record = $('.audio-record');
    this.chosen = $('.audio-chosen');
    this.name = $('.audio-name');
    this.audio = $('.audio-preview');
    this.error = $('.error');
    this.changed = changed;

    this.choose.addEventListener('click', () => this.input.click());
    this.input.addEventListener('change', () => this.input.files[0] && this.#take(this.input.files[0], this.input.files[0].name));
    this.record.addEventListener('click', () => (this.#recorder ? this.#stopRecording() : this.#startRecording()));
    this.drop.addEventListener('dragover', (event) => {
      event.preventDefault();
      this.drop.classList.add('over');
    });
    this.drop.addEventListener('dragleave', () => this.drop.classList.remove('over'));
    this.drop.addEventListener('drop', (event) => {
      event.preventDefault();
      this.drop.classList.remove('over');
      const file = event.dataTransfer.files[0];
      if (file) this.#take(file, file.name);
    });
  }

  /** Whether audio is given. */
  get given() {
    return this.#file !== null;
  }

  /** The audio given as a WAVE file of one channel at `rate`. */
  async wav(rate) {
    return toWav(await this.#file.arrayBuffer(), rate);
  }

  #take(file, label) {
    this.#file = file;
    this.error.hidden = true;
    if (this.#preview) URL.revokeObjectURL(this.#preview);
    this.#preview = URL.createObjectURL(file);
    this.audio.src = this.#preview;
    this.name.textContent = label;
    this.chosen.hidden = false;
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
    this.record.setAttribute('aria-pressed', 'true');
    this.choose.disabled = true;
    const tick = () => (this.record.textContent = `Stop recording (${recorder.seconds.toFixed(1)} s)`);
    tick();
    this.#ticking = setInterval(tick, 200);
  }

  async #stopRecording() {
    clearInterval(this.#ticking);
    const recorder = this.#recorder;
    this.#recorder = null;
    this.record.setAttribute('aria-pressed', 'false');
    this.record.textContent = 'Record';
    this.choose.disabled = false;
    const seconds = recorder.seconds;
    this.#take(await recorder.stop(), `Recording, ${seconds.toFixed(1)} s`);
  }

  /** Shows a failure of the audio given, the browser's or as the server words it. */
  fail(message) {
    this.error.textContent = message;
    this.error.hidden = false;
  }
}
