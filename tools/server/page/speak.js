// The speak panel: a text and the synthesis model's options, the speech played as the server makes it, which can be
// stopped and saved as a WAVE file, and for a model that takes voice files, a voice made from a recording.

import * as api from './api.js';
import { AudioInput } from './audio-input.js';
import { OptionForm } from './options.js';
import { Player } from './player.js';
import { save } from './wav.js';

const $ = (id) => document.getElementById(id);

const STOPS = {
  max_seconds: 'It stopped at the length that max seconds sets.',
  model_limit: 'It stopped at the longest speech the model makes.',
};

export class SpeakPanel {
  #held = null;
  #options = null;
  #player = new Player();
  #controller = null;
  #voice = new AudioInput($('voice-audio'));

  constructor() {
    $('speak-form').addEventListener('submit', (event) => {
      event.preventDefault();
      this.#speak();
    });
    $('speak-text').addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) $('speak-form').requestSubmit();
    });
    $('speak-stop').addEventListener('click', () => {
      this.#controller?.abort();
      this.#player.stop();
    });
    $('speak-save').addEventListener('click', () => save(this.#player.wav(), 'speech.wav'));
    $('voice-form').addEventListener('submit', (event) => {
      event.preventDefault();
      this.#addVoice();
    });
  }

  /** Shows the synthesis model held, `held` of GET /speech/models, or that there is none. */
  show(held) {
    this.#held = held;
    $('speak-empty').hidden = held !== null;
    $('speak-form').hidden = held === null;
    $('voice-maker').hidden = !held?.model.voice_files;
    if (!held) return;
    const kept = this.#options?.values() ?? {};
    const { model } = held;
    $('speak-model').textContent = `${model.name} on ${model.device}, ${model.sample_rate} Hz`;
    this.#options = new OptionForm(model, $('speak-options'), $('speak-more'));
    this.#options.restore(kept);
  }

  #status(text) {
    $('speak-status').textContent = text;
  }

  /** Shows a failure at the field of the input it names, or under the form. */
  #fail(error) {
    if (error.param === 'input') {
      $('speak-text').setAttribute('aria-invalid', 'true');
      $('speak-text-error').textContent = error.message;
      $('speak-text-error').hidden = false;
      return;
    }
    if (error.param && this.#options.showError(error.param, error.message)) return;
    $('speak-error').textContent = error.message;
    $('speak-error').hidden = false;
  }

  async #speak() {
    const text = $('speak-text');
    this.#options.clearErrors();
    text.removeAttribute('aria-invalid');
    $('speak-text-error').hidden = true;
    $('speak-error').hidden = true;
    if (!text.value.trim()) {
      this.#fail({ param: 'input', message: 'Write the text to speak.' });
      text.focus();
      return;
    }
    this.#controller = new AbortController();
    $('speak-start').disabled = true;
    $('speak-stop').disabled = false;
    $('speak-save').disabled = true;
    this.#status('Starting');
    const started = performance.now();
    let firstAudio = 0;
    try {
      for await (const event of api.speak({ input: text.value, ...this.#options.values() }, this.#controller.signal)) {
        if (event.type === 'start') {
          await this.#player.begin(event.rate);
        } else if (event.type === 'audio') {
          firstAudio ||= performance.now();
          this.#player.push(event.samples);
          this.#status(`Speaking: ${this.#player.seconds.toFixed(1)} s of audio`);
        } else {
          const first = ((firstAudio - started) / 1000).toFixed(2);
          this.#status(`${this.#player.seconds.toFixed(2)} s of audio, the first after ${first} s; seed ${event.seed}. ${STOPS[event.stop] ?? ''}`);
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') this.#status(`Stopped after ${this.#player.seconds.toFixed(2)} s of audio.`);
      else {
        this.#status('');
        this.#fail(error);
      }
    } finally {
      this.#controller = null;
      $('speak-start').disabled = false;
      $('speak-stop').disabled = true;
      $('speak-save').disabled = this.#player.seconds === 0;
    }
  }

  async #addVoice() {
    const name = $('voice-name');
    const error = $('voice-error');
    error.hidden = true;
    name.removeAttribute('aria-invalid');
    const fail = (message) => {
      error.textContent = message;
      error.hidden = false;
    };
    if (!this.#voice.given) return fail('Drop, choose or record a recording of the voice first.');
    if (!name.value.trim()) {
      name.setAttribute('aria-invalid', 'true');
      name.focus();
      return fail('Name the voice, so that you can choose it.');
    }
    $('voice-add').disabled = true;
    $('voice-status').textContent = 'Making the voice';
    try {
      const recording = await this.#voice.wav(this.#held.model.sample_rate);
      this.show(await api.addVoice(name.value.trim(), recording));
      this.#options.select('voice', name.value.trim());
      $('voice-status').textContent = `Added the voice ${name.value.trim()}, chosen above.`;
      name.value = '';
    } catch (e) {
      $('voice-status').textContent = '';
      if (e.param === 'file') this.#voice.fail(e.message);
      else fail(e.message);
    } finally {
      $('voice-add').disabled = false;
    }
  }
}
