// The transcribe panel: audio dropped, chosen or recorded, the options the transcription form takes, and the text with
// the language the model heard and the segments with their times, where the model gives them.

import * as api from './api.js';
import { AudioInput } from './audio-input.js';
import { OptionForm, languageName } from './options.js';

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
  #audio = new AudioInput($('transcribe-audio'), () => ($('transcribe-error').hidden = true));

  constructor() {
    $('transcribe-form').addEventListener('submit', (event) => {
      event.preventDefault();
      this.#transcribe();
    });
    $('transcribe-stop').addEventListener('click', () => this.#controller?.abort());
  }

  /** Shows the recognition model held, `held` of GET /speech/models, or that there is none. */
  show(held) {
    this.#held = held;
    $('transcribe-empty').hidden = held !== null;
    $('transcribe-form').hidden = held === null;
    if (!held) return;
    const kept = this.#options?.values() ?? {};
    const { model } = held;
    $('transcribe-model').textContent = `${model.name} on ${model.device}`;
    this.#options = new OptionForm(model, $('transcribe-options'), null, FORM_OPTIONS);
    this.#options.restore(kept);
  }

  #status(text) {
    $('transcribe-status').textContent = text;
  }

  async #transcribe() {
    const error = $('transcribe-error');
    error.hidden = true;
    this.#options.clearErrors();
    if (!this.#audio.given) {
      this.#audio.fail('Drop, choose or record the audio to transcribe first.');
      return;
    }
    this.#controller = new AbortController();
    $('transcribe-start').disabled = true;
    $('transcribe-stop').disabled = false;
    try {
      this.#status('Preparing the audio');
      const recording = await this.#audio.wav(this.#held.model.sample_rate);
      this.#status('Transcribing');
      const started = performance.now();
      const { result, stop } = await api.transcribe(recording, this.#options.values(), this.#controller.signal);
      this.#status('');
      this.#render(result, stop, (performance.now() - started) / 1000);
    } catch (e) {
      this.#status(e.name === 'AbortError' ? 'Stopped.' : '');
      if (e.name === 'AbortError') return;
      if (e.param === 'file') this.#audio.fail(e.message);
      else if (!(e.param && this.#options.showError(e.param, e.message))) {
        error.textContent = e.message;
        error.hidden = false;
      }
    } finally {
      this.#controller = null;
      $('transcribe-start').disabled = false;
      $('transcribe-stop').disabled = true;
    }
  }

  #render(result, stop, took) {
    $('transcript').hidden = false;
    $('transcript-text').textContent = result.text || '(no speech)';
    const meta = [];
    if (result.language) meta.push(`Heard ${result.language.split(',').map(languageName).join(' and ')}.`);
    meta.push(`${seconds(result.duration)} of audio in ${seconds(took)}.`);
    if (stop === 'model_limit') meta.push('It stopped at the most text the model writes.');
    $('transcript-meta').textContent = meta.join(' ');
    const table = $('segments');
    const rows = table.tBodies[0];
    rows.replaceChildren();
    for (const segment of result.segments ?? []) {
      const row = rows.insertRow();
      row.insertCell().textContent = seconds(segment.start);
      row.insertCell().textContent = seconds(segment.end);
      row.insertCell().textContent = segment.text;
    }
    table.hidden = rows.rows.length === 0;
  }
}
