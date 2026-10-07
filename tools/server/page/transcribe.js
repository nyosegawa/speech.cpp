// The transcribe panel: audio dropped, chosen, recorded or handed over from the speak panel, the options the
// transcription form takes, and the text as a card with the language the model heard, the time it took, a copy button
// and the segments with their times, where the model gives them.

import * as api from './api.js';
import { AudioInput } from './audio-input.js';
import { languageName } from './languages.js';
import { OptionForm } from './options.js';

const $ = (id) => document.getElementById(id);

/** The options of the model that OpenAI's transcription form carries; timestamps follow from verbose_json. */
const FORM_OPTIONS = new Set(['language', 'prompt', 'decoding']);

function seconds(value) {
  return `${value.toFixed(2)} s`;
}

export class TranscribePanel {
  #held = null;
  #options = null;
  #controller = null;
  #audio = new AudioInput($('transcribe-audio'), () => {
    $('transcribe-error').hidden = true;
    this.#ready();
  });

  constructor() {
    $('transcribe-form').addEventListener('submit', (event) => {
      event.preventDefault();
      this.#transcribe();
    });
    $('transcribe-stop').addEventListener('click', () => this.#controller?.abort());
    $('transcript-copy').addEventListener('click', async () => {
      const button = $('transcript-copy');
      try {
        await navigator.clipboard.writeText($('transcript-text').textContent);
        button.textContent = 'Copied';
      } catch {
        // A browser that keeps the clipboard from the page still lets the user copy the text, selected for them.
        getSelection().selectAllChildren($('transcript-text'));
        button.textContent = 'Selected; copy it';
      }
      setTimeout(() => (button.textContent = 'Copy'), 2000);
    });
  }

  /** Shows the recognition model held, `held` of GET /speech/models, or that there is none. */
  show(held) {
    this.#held = held;
    if (held) {
      const kept = this.#options?.values() ?? {};
      this.#options = new OptionForm(held.model, $('transcribe-options'), null, FORM_OPTIONS);
      this.#options.restore(kept);
    } else {
      $('transcribe-options').replaceChildren();
      this.#options = null;
    }
    this.#ready();
  }

  /** Takes audio from elsewhere on the page, the speak panel's speech. */
  use(file, label) {
    this.#audio.use(file, label);
    $('transcribe').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  /** The button transcribes once there is a model and audio, and says what is missing until then. */
  #ready() {
    const start = $('transcribe-start');
    start.disabled = !this.#held || !this.#audio.given || this.#controller !== null;
    start.title = !this.#held ? 'Choose a model first' : !this.#audio.given ? 'Give the audio first' : '';
  }

  #status(text) {
    $('transcribe-status').textContent = text;
  }

  /** Transcribes the audio given, one transcription at a time. */
  async #transcribe() {
    if (this.#controller) return;
    const error = $('transcribe-error');
    error.hidden = true;
    this.#options.clearErrors();
    this.#controller = new AbortController();
    this.#ready();
    $('transcribe-stop').hidden = false;
    try {
      this.#status('Preparing the audio');
      const recording = await this.#audio.wav(this.#held.model.sample_rate);
      this.#status('Transcribing');
      const started = performance.now();
      const { result, stop } = await api.transcribe(recording, this.#options.values(), this.#controller.signal);
      this.#render(result, stop, (performance.now() - started) / 1000);
    } catch (e) {
      if (e.name === 'AbortError') {
        // Stopped by the user, who sees the button come back.
      } else if (e.param === 'file') {
        this.#audio.fail(e.message);
      } else if (!(e.param && this.#options.showError(e.param, e.message))) {
        error.textContent = e.message;
        error.hidden = false;
      }
    } finally {
      this.#status('');
      this.#controller = null;
      $('transcribe-stop').hidden = true;
      this.#ready();
    }
  }

  #render(result, stop, took) {
    $('transcript').hidden = false;
    $('transcript-text').textContent = result.text || '(no speech)';
    $('transcript-copy').hidden = !result.text;
    const meta = [];
    if (result.language) meta.push(result.language.split(',').map(languageName).join(', '));
    meta.push(`${seconds(result.duration)} of audio in ${seconds(took)}`);
    meta.push(this.#held.model.name);
    if (stop === 'model_limit') meta.push('stopped at the most text the model writes');
    $('transcript-meta').textContent = meta.join(' · ');
    const segments = $('segments');
    const rows = segments.querySelector('tbody');
    rows.replaceChildren();
    for (const segment of result.segments ?? []) {
      const row = rows.insertRow();
      row.insertCell().textContent = seconds(segment.start);
      row.insertCell().textContent = seconds(segment.end);
      row.insertCell().textContent = segment.text;
    }
    segments.querySelector('summary').textContent = `${rows.rows.length} segments with their times`;
    segments.hidden = rows.rows.length === 0;
  }
}
