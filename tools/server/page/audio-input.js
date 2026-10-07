// A place to give audio: drop a file on it, choose one, or record the microphone. Once audio is given it shows its name,
// its length and a small player in place of the prompt to drop one. The speak panel's voice maker and the transcribe
// panel each have one.

import { Recorder, toWav } from './audio.js';

function seconds(value) {
  return `${value.toFixed(1)} s`;
}

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
    this.empty = $('.drop-empty');
    this.chosen = $('.drop-chosen');
    this.recording = $('.drop-recording');
    this.recordingTime = $('.recording-time');
    this.input = $('.audio-file');
    this.choose = $('.audio-choose');
    this.record = $('.audio-record');
    this.remove = $('.audio-remove');
    this.name = $('.audio-name');
    this.audio = $('.audio-preview');
    this.error = $('.error');
    this.changed = changed;

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

  /** The audio given as a WAVE file of one channel at `rate`. */
  async wav(rate) {
    return toWav(await this.#file.arrayBuffer(), rate);
  }

  /** Takes `file`, a File or a Blob of audio, shown as `label`. */
  use(file, label) {
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

  /** Shows a failure of the audio given, the browser's or as the server words it. */
  fail(message) {
    this.error.textContent = message;
    this.error.hidden = false;
  }

  #show(state) {
    this.empty.hidden = state !== 'empty';
    this.chosen.hidden = state !== 'chosen';
    this.recording.hidden = state !== 'recording';
    this.remove.hidden = state !== 'chosen';
    this.choose.textContent = state === 'chosen' ? 'Another file' : 'Choose a file';
    this.choose.disabled = state === 'recording';
    this.drop.dataset.state = state;
  }

  #clear() {
    this.#file = null;
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
    this.record.setAttribute('aria-pressed', 'true');
    this.record.textContent = 'Stop recording';
    this.#show('recording');
    const tick = () => (this.recordingTime.textContent = `Recording, ${seconds(recorder.seconds)}`);
    tick();
    this.#ticking = setInterval(tick, 200);
  }

  async #stopRecording() {
    clearInterval(this.#ticking);
    const recorder = this.#recorder;
    this.#recorder = null;
    this.record.setAttribute('aria-pressed', 'false');
    this.record.textContent = 'Record';
    this.use(await recorder.stop(), 'Recording');
  }
}
