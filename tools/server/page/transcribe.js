// The transcribe panel: audio dropped, chosen, recorded or handed over from the speak panel, the options the
// transcription form takes, and the text as a card. Audio goes to the server in pieces, so that an hour of it costs
// what a minute does per request.

import * as api from './api.js';
import { AudioInput } from './audio-input.js';
import { OptionForm } from './options.js';
import { Transcript } from './pieces.js';
import { length, TranscriptView } from './transcript.js';
import { wavFile } from './wav.js';

const $ = (id) => document.getElementById(id);

/** The options of the model that OpenAI's transcription form carries; timestamps follow from verbose_json. */
export const FORM_OPTIONS = new Set(['language', 'prompt', 'decoding']);
const SETTINGS_KEY = 'speech.cpp transcribe settings';

/** The value of a number field, within its bounds, or its default when it holds no number. */
export function bounded(input) {
  const value = Number(input.value);
  return Number.isFinite(value) ? Math.min(Number(input.max), Math.max(Number(input.min), value)) : Number(input.defaultValue);
}

export class TranscribePanel {
  #held = null;
  #options = null;
  #controller = null;
  #view = new TranscriptView($('transcribe-transcript'));
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
    try {
      const kept = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}');
      if (Number.isFinite(kept.piece)) $('transcribe-piece').value = kept.piece;
    } catch {
      // Storage the browser refuses, or a value of another shape, leaves the setting as the page gives it.
    }
    $('transcribe-piece').addEventListener('change', () => {
      try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify({ piece: bounded($('transcribe-piece')) }));
      } catch {
        // Without storage the setting lasts as long as the page.
      }
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
  }

  /** The button transcribes once there is a model and audio, and says what is missing until then. */
  #ready() {
    const start = $('transcribe-start');
    start.disabled = !this.#held || !this.#audio.given || this.#controller !== null || this.#audio.recording;
    start.title = !this.#held ? 'Choose a model first' : !this.#audio.given ? 'Give the audio first' : '';
    $('transcribe-stop').hidden = this.#controller === null;
  }

  #status(text) {
    $('transcribe-status').textContent = text;
  }

  /** Shows a failure next to its cause: the audio, an option's field, or below the form. */
  #fail(e) {
    if (e.name === 'AbortError') {
      // Stopped by the user, who sees the button come back.
    } else if (e.param === 'file') {
      this.#audio.fail(e.message);
    } else if (!(e.param && this.#options?.showError(e.param, e.message))) {
      $('transcribe-error').textContent = e.message;
      $('transcribe-error').hidden = false;
    }
  }

  /** Transcribes the audio given, a piece at a time, showing the text as each piece comes back. */
  async #transcribe() {
    if (this.#controller) return;
    $('transcribe-error').hidden = true;
    this.#options.clearErrors();
    this.#controller = new AbortController();
    this.#ready();
    const model = this.#held.model;
    const transcript = new Transcript();
    const started = performance.now();
    try {
      this.#status('Preparing the audio');
      for await (const piece of this.#audio.pieces(model.sample_rate, bounded($('transcribe-piece')))) {
        this.#status(`Transcribing, ${length(piece.start)} of ${length(piece.total)} done`);
        const wav = wavFile(piece.samples, piece.rate);
        const { result, stop } = await api.transcribe(wav, this.#options.values(), this.#controller.signal);
        transcript.add(result, stop, piece.start);
        this.#view.render(transcript, model, { done: false });
      }
      this.#view.render(transcript, model, { took: (performance.now() - started) / 1000 });
    } catch (e) {
      this.#fail(e);
    } finally {
      this.#status('');
      this.#controller = null;
      this.#ready();
    }
  }
}
