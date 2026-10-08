// The transcribe panel: audio dropped, chosen, recorded or handed over from the speak panel, the options the
// transcription form takes, and the text as a card with the language the model heard, the time it took, a copy button
// and the segments with their times, where the model gives them. Audio goes to the server in pieces, so that an hour of
// it costs what a minute does per request; a recording can be transcribed as it is made.

import * as api from './api.js';
import { AudioInput } from './audio-input.js';
import { languageName } from './languages.js';
import { LiveTranscription } from './live.js';
import { OptionForm } from './options.js';
import { join, Transcript } from './pieces.js';
import { wavFile } from './wav.js';

const $ = (id) => document.getElementById(id);

/** The options of the model that OpenAI's transcription form carries; timestamps follow from verbose_json. */
const FORM_OPTIONS = new Set(['language', 'prompt', 'decoding']);
const SETTINGS_KEY = 'speech.cpp transcribe settings';

function seconds(value) {
  return `${value.toFixed(2)} s`;
}

/** A length as a user reads it: "12.3 s" under a minute, "1:02:05" or "4:07" above. */
function length(value) {
  if (value < 60) return `${value.toFixed(1)} s`;
  const s = Math.floor(value % 60).toString().padStart(2, '0');
  const m = Math.floor(value / 60) % 60;
  return value < 3600 ? `${m}:${s}` : `${Math.floor(value / 3600)}:${m.toString().padStart(2, '0')}:${s}`;
}

function bounded(input) {
  const value = Number(input.value);
  return Number.isFinite(value) ? Math.min(Number(input.max), Math.max(Number(input.min), value)) : Number(input.defaultValue);
}

export class TranscribePanel {
  #held = null;
  #options = null;
  #controller = null;
  #live = null;
  #audio = new AudioInput(
    $('transcribe-audio'),
    () => {
      $('transcribe-error').hidden = true;
      this.#ready();
    },
    {
      started: (rate) => this.#startLive(rate),
      samples: (samples) => this.#live?.push(samples),
      stopped: () => this.#live?.stop(),
    },
  );

  constructor() {
    $('transcribe-form').addEventListener('submit', (event) => {
      event.preventDefault();
      this.#transcribe();
    });
    $('transcribe-stop').addEventListener('click', () => {
      this.#controller?.abort();
      this.#live?.abort();
    });
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
    this.#restoreSettings();
    for (const id of ['transcribe-when', 'transcribe-every', 'transcribe-piece']) {
      $(id).addEventListener('change', () => this.#saveSettings());
    }
  }

  /** Shows the recognition model held, `held` of GET /speech/models, or that there is none. */
  show(held) {
    this.#live?.abort();
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

  /** The settings of how audio is sent: live while recording, every few seconds, in pieces of at most so long. */
  #settings() {
    return {
      live: $('transcribe-when').value === 'live',
      every: bounded($('transcribe-every')),
      piece: bounded($('transcribe-piece')),
    };
  }

  #restoreSettings() {
    try {
      const kept = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}');
      if (kept.when === 'live' || kept.when === 'after') $('transcribe-when').value = kept.when;
      if (Number.isFinite(kept.every)) $('transcribe-every').value = kept.every;
      if (Number.isFinite(kept.piece)) $('transcribe-piece').value = kept.piece;
    } catch {
      // Storage the browser refuses, or a value of another shape, leaves the settings as the page gives them.
    }
    this.#showSettings();
  }

  #saveSettings() {
    this.#showSettings();
    const { every, piece } = this.#settings();
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify({ when: $('transcribe-when').value, every, piece }));
    } catch {
      // Without storage the settings last as long as the page.
    }
  }

  #showSettings() {
    $('transcribe-every-field').hidden = !this.#settings().live;
  }

  /** The button transcribes once there is a model and audio, and says what is missing until then. */
  #ready() {
    const start = $('transcribe-start');
    const busy = this.#controller !== null || this.#live !== null || this.#audio.recording;
    start.disabled = !this.#held || !this.#audio.given || busy;
    start.title = !this.#held ? 'Choose a model first' : !this.#audio.given ? 'Give the audio first' : '';
    $('transcribe-stop').hidden = this.#controller === null && this.#live === null;
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

  #begin() {
    $('transcribe-error').hidden = true;
    this.#options.clearErrors();
  }

  /** Transcribes the audio given, a piece at a time, showing the text as each piece comes back. */
  async #transcribe() {
    if (this.#controller || this.#live) return;
    this.#begin();
    this.#controller = new AbortController();
    this.#ready();
    const transcript = new Transcript();
    const started = performance.now();
    const took = () => (performance.now() - started) / 1000;
    try {
      this.#status('Preparing the audio');
      for await (const piece of this.#audio.pieces(this.#held.model.sample_rate, this.#settings().piece)) {
        this.#status(`Transcribing, ${length(piece.start)} of ${length(piece.total)} done`);
        const wav = wavFile(piece.samples, piece.rate);
        const { result, stop } = await api.transcribe(wav, this.#options.values(), this.#controller.signal);
        transcript.add(result, stop, piece.start);
        this.#render(transcript, { done: false });
      }
      this.#render(transcript, { took: took() });
    } catch (e) {
      this.#fail(e);
    } finally {
      this.#status('');
      this.#controller = null;
      this.#ready();
    }
  }

  /** Starts transcribing a recording as it is made, when the panel is set to and nothing else runs. */
  #startLive(rate) {
    const { live: wanted, every, piece } = this.#settings();
    if (!wanted || !this.#held || this.#controller || this.#live) return false;
    this.#begin();
    this.#live = new LiveTranscription(rate, {
      pieceSeconds: piece,
      everySeconds: every,
      fields: () => this.#options.values(),
      show: (transcript, provisional) => this.#render(transcript, { provisional, done: false }),
    });
    this.#ready();
    this.#status('Listening');
    this.#render(new Transcript(), { done: false });
    const live = this.#live;
    live
      .run()
      .then((transcript) => this.#render(transcript, { live: true }))
      .catch((e) => this.#fail(e))
      .finally(() => {
        this.#status('');
        if (this.#live === live) this.#live = null;
        this.#ready();
      });
    return true;
  }

  /**
   * Shows `transcript`, with `provisional`, the text of the piece still being recorded, after it in the muted colour;
   * `done` is false while pieces are still to come.
   */
  #render(transcript, { provisional = '', done = true, took = null, live = false }) {
    $('transcript').hidden = false;
    const text = $('transcript-text');
    text.replaceChildren(transcript.text);
    if (provisional) {
      const rest = document.createElement('span');
      rest.className = 'provisional';
      rest.textContent = join(transcript.text, provisional).slice(transcript.text.length);
      text.append(rest);
    }
    if (!transcript.text && !provisional) text.textContent = done ? '(no speech)' : '…';
    $('transcript-copy').hidden = !done || !transcript.text;
    const meta = [];
    if (transcript.languages.length) meta.push(transcript.languages.map(languageName).join(', '));
    if (transcript.seconds) meta.push(`${length(transcript.seconds)} of audio${took === null ? '' : ` in ${length(took)}`}`);
    if (live) meta.push('transcribed as it was recorded');
    meta.push(this.#held.model.name);
    if (transcript.stop === 'model_limit') meta.push('a piece stopped at the most text the model writes');
    $('transcript-meta').textContent = meta.join(' · ');
    const segments = $('segments');
    const rows = segments.querySelector('tbody');
    rows.replaceChildren();
    for (const segment of transcript.segments) {
      const row = rows.insertRow();
      row.insertCell().textContent = seconds(segment.start);
      row.insertCell().textContent = seconds(segment.end);
      row.insertCell().textContent = segment.text;
    }
    segments.querySelector('summary').textContent = `${rows.rows.length} segments with their times`;
    segments.hidden = rows.rows.length === 0;
  }
}
