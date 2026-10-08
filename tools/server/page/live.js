// The live panel: transcribes the microphone while the user speaks, the text that can still change shown in the muted
// colour until it is final, and the whole text once the user stops.

import { Recorder } from './audio.js';
import { LiveTranscription } from './live-transcription.js';
import { OptionForm } from './options.js';
import { Transcript } from './pieces.js';
import { bounded, FORM_OPTIONS } from './transcribe.js';
import { length, TranscriptView } from './transcript.js';

const $ = (id) => document.getElementById(id);

/** How often the samples recorded go to the transcription and the time listened shows. */
const TICK_MS = 200;
const SETTINGS_KEY = 'speech.cpp live settings';

export class LivePanel {
  #held = null;
  #options = null;
  #recorder = null;
  /** The recorder of a run whose microphone the browser has not yet given; a change of model ends it. */
  #starting = null;
  #live = null;
  #ticking = 0;
  #busy;
  #view = new TranscriptView($('live-transcript'));

  /** `busy(running)` follows the start and the end of each run, which the panel's tab shows. */
  constructor(busy) {
    this.#busy = busy;
    $('live-form').addEventListener('submit', (event) => {
      event.preventDefault();
      if (this.#recorder) this.#stop();
      else this.#start();
    });
    try {
      const kept = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}');
      if (Number.isFinite(kept.piece)) $('live-piece').value = kept.piece;
    } catch {
      // Storage the browser refuses, or a value of another shape, leaves the setting as the page gives it.
    }
    $('live-piece').addEventListener('change', () => {
      try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify({ piece: bounded($('live-piece')) }));
      } catch {
        // Without storage the setting lasts as long as the page.
      }
    });
  }

  /** Shows the recognition model held, `held` of GET /speech/models, or that there is none; a run going on ends. */
  show(held) {
    this.#starting = null;
    if (this.#live) this.#abort();
    this.#held = held;
    if (held) {
      const kept = this.#options?.values() ?? {};
      this.#options = new OptionForm(held.model, $('live-options'), null, FORM_OPTIONS);
      this.#options.restore(kept);
    } else {
      $('live-options').replaceChildren();
      this.#options = null;
    }
    this.#ready();
  }

  /** The button starts once there is a model, stops while listening, and waits while the last words are transcribed. */
  #ready() {
    const button = $('live-start');
    button.textContent = this.#recorder ? 'Stop' : 'Start';
    button.classList.toggle('primary', !this.#recorder);
    button.classList.toggle('stop', this.#recorder !== null);
    button.disabled = !this.#held || this.#starting !== null || (this.#live !== null && !this.#recorder);
    button.title = this.#held ? '' : 'Choose a model first';
    $('live-dot').hidden = !this.#recorder;
  }

  #status(text) {
    $('live-status').textContent = text;
  }

  /** Shows a failure next to its cause: an option's field, or below the form. */
  #fail(e) {
    if (e.name === 'AbortError') return;
    if (!(e.param && this.#options?.showError(e.param, e.message))) {
      $('live-error').textContent = e.message;
      $('live-error').hidden = false;
    }
  }

  async #start() {
    if (this.#live || this.#starting || !this.#held) return;
    $('live-error').hidden = true;
    this.#options.clearErrors();
    const recorder = new Recorder();
    this.#starting = recorder;
    this.#ready();
    try {
      await recorder.start();
    } catch (e) {
      if (this.#starting !== recorder) return;
      this.#starting = null;
      this.#fail(e.name === 'NotAllowedError' ? new Error('The browser was not allowed to use the microphone.') : e);
      this.#ready();
      return;
    }
    if (this.#starting !== recorder) {
      await recorder.stop();
      return;
    }
    this.#starting = null;
    const model = this.#held.model;
    const live = new LiveTranscription(recorder.rate, {
      pieceSeconds: bounded($('live-piece')),
      fields: () => this.#options.values(),
      show: (transcript, provisional) => this.#view.render(transcript, model, { provisional, done: false }),
    });
    this.#recorder = recorder;
    this.#live = live;
    this.#busy(true);
    this.#ready();
    this.#view.render(new Transcript(), model, { done: false });
    let heard = 0;
    const tick = () => {
      const samples = recorder.take();
      heard += samples.length;
      live.push(samples);
      this.#status(`Listening, ${length(heard / recorder.rate)}`);
    };
    tick();
    this.#ticking = setInterval(tick, TICK_MS);
    try {
      this.#view.render(await live.run(), model, { live: true });
    } catch (e) {
      this.#fail(e);
    } finally {
      // A run that failed while listening lets go of the microphone too.
      if (this.#recorder === recorder) await this.#release();
      if (this.#live === live) this.#live = null;
      this.#status('');
      this.#busy(false);
      this.#ready();
    }
  }

  /** Ends the recording; the run transcribes the last words and ends. */
  async #stop() {
    const live = this.#live;
    live.push(await this.#release());
    live.stop();
    this.#status('Transcribing the last words');
    this.#ready();
  }

  /** Ends the run without the last words. */
  #abort() {
    this.#live.abort();
    this.#live = null;
    if (this.#recorder) this.#release();
  }

  /** Turns the microphone off, and returns the samples recorded since the last tick. */
  async #release() {
    clearInterval(this.#ticking);
    const recorder = this.#recorder;
    this.#recorder = null;
    return recorder.stop();
  }
}
