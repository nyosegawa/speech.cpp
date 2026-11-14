// The talk panel: a conversation with a model that listens and speaks at once. Before it starts, the panel shows the
// dialogue model's options (a character card, where the model takes one) and makes a voice from a recording; while it
// runs, the settings fold away and the conversation fills the panel (talk-view.js), heard through the microphone and
// played on the speakers (dialogue.js).

import * as api from './api.js';
import { Recorder } from './audio.js';
import { AudioInput } from './audio-input.js';
import { Dialogue } from './dialogue.js';
import { OptionForm, withServerDefaults } from './options.js';
import { TalkView } from './talk-view.js';

const $ = (id) => document.getElementById(id);

/** How often the microphone's samples go to the server. */
const TICK_MS = 50;

function clock(seconds) {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

export class TalkPanel {
  #held = null;
  #options = null;
  #view = new TalkView($('talk-stage'));
  #voice = new AudioInput($('talk-voice-audio'));
  #recorder = null;
  #dialogue = null;
  /** The recorder of a conversation whose microphone or prompt is not there yet; a change of model ends it. */
  #starting = null;
  #ticking = 0;
  #busy;

  /** `busy(running)` follows the start and the end of each conversation, which the panel's tab shows. */
  constructor(busy) {
    this.#busy = busy;
    $('talk-start').addEventListener('click', () => (this.#dialogue || this.#starting ? this.#hangUp() : this.#start()));
    $('talk-voice-form').addEventListener('submit', (event) => {
      event.preventDefault();
      this.#addVoice();
    });
  }

  /** Shows the dialogue model held, `held` of GET /speech/models, or that there is none; a conversation going on ends. */
  show(held) {
    this.#end();
    this.#held = held;
    $('talk-form').hidden = held === null;
    $('talk-voice-maker').hidden = !held?.model.voice_recordings;
    if (held) {
      const kept = this.#options?.values() ?? {};
      this.#options = new OptionForm(withServerDefaults(held), $('talk-options'), $('talk-more'));
      this.#options.restore(kept);
    } else {
      this.#options = null;
    }
    this.#ready();
  }

  #ready() {
    const talking = this.#dialogue !== null || this.#starting !== null;
    const button = $('talk-start');
    button.textContent = talking ? 'Hang up' : 'Start talking';
    button.classList.toggle('primary', !talking);
    button.classList.toggle('stop', talking);
    button.disabled = !this.#held;
    button.title = this.#held ? '' : 'Choose a model first';
    $('talk-setup').hidden = talking;
    $('talk-dot').hidden = this.#dialogue === null;
  }

  #status(text) {
    $('talk-status').textContent = text;
  }

  /** Shows a failure next to its cause: an option's field, or below the conversation. */
  #fail(e) {
    $('talk-setup').hidden = false;
    const field = e.param?.split('.').pop();
    if (!(field && this.#options?.showError(field, e.message))) {
      $('talk-error').textContent = e.name === 'NotAllowedError' ? 'The browser was not allowed to use the microphone.' : e.message;
      $('talk-error').hidden = false;
    }
  }

  async #start() {
    if (!this.#held) return;
    $('talk-error').hidden = true;
    this.#options.clearErrors();
    const options = this.#options.values();
    const rate = this.#held.model.sample_rate;
    const recorder = new Recorder(rate);
    const dialogue = new Dialogue(
      (samples, rate, time, text) => this.#view.said(samples, rate, time, text),
      (e) => {
        this.#end();
        this.#fail(e);
        this.#ready();
      },
    );
    this.#starting = recorder;
    this.#ready();
    $('talk-stage').hidden = false;
    this.#view.start(options.name || this.#held.model.basename || this.#held.model.name);
    this.#status('Starting the conversation');
    const [microphone, session] = await Promise.allSettled([recorder.start(), dialogue.open(options)]);
    if (microphone.status === 'rejected' || session.status === 'rejected' || this.#starting !== recorder) {
      dialogue.abort();
      await recorder.stop();
      if (this.#starting !== recorder) return;
      this.#starting = null;
      this.#view.stop();
      this.#fail(microphone.status === 'rejected' ? microphone.reason : session.reason);
      this.#status('');
      this.#ready();
      return;
    }
    // The conversation begins with the model's first frame: what the microphone heard while the prompt ran came before it.
    recorder.take();
    this.#starting = null;
    this.#recorder = recorder;
    this.#dialogue = dialogue;
    this.#busy(true);
    this.#ready();
    const began = TalkView.now();
    const tick = () => {
      const samples = recorder.take();
      if (!samples.length) return;
      this.#view.heard(samples, rate, TalkView.now() - samples.length / rate);
      dialogue.push(samples);
      this.#status(clock(TalkView.now() - began));
    };
    this.#ticking = setInterval(tick, TICK_MS);
  }

  /** Ends the conversation once the server has answered what it heard, and leaves it on the stage. */
  async #hangUp() {
    if (this.#starting) {
      this.#starting = null;
      this.#ready();
      return;
    }
    const dialogue = this.#dialogue;
    clearInterval(this.#ticking);
    const rest = await this.#recorder.stop();
    this.#recorder = null;
    dialogue.push(rest);
    this.#status('Hanging up');
    try {
      await dialogue.hangUp();
    } catch (e) {
      this.#fail(e);
    }
    if (this.#dialogue === dialogue) this.#dialogue = null;
    this.#view.stop();
    this.#status('');
    this.#busy(false);
    this.#ready();
  }

  /** Ends a conversation that is starting or going on, without waiting for the server. */
  #end() {
    this.#starting = null;
    clearInterval(this.#ticking);
    this.#recorder?.stop();
    this.#recorder = null;
    if (this.#dialogue) {
      this.#dialogue.abort();
      this.#dialogue = null;
      this.#busy(false);
    }
    this.#view.stop();
    this.#status('');
  }

  async #addVoice() {
    const name = $('talk-voice-name');
    const error = $('talk-voice-error');
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
    $('talk-voice-add').disabled = true;
    $('talk-voice-status').textContent = 'Making the voice';
    try {
      const recording = await this.#voice.wav(this.#held.model.sample_rate);
      this.show(await api.addVoice(name.value.trim(), recording, 'dialogue'));
      this.#options.select('voice', name.value.trim());
      $('talk-voice-status').textContent = `Added ${name.value.trim()}, chosen above.`;
      name.value = '';
    } catch (e) {
      $('talk-voice-status').textContent = '';
      if (e.param === 'file') this.#voice.fail(e.message);
      else fail(e.message);
    } finally {
      $('talk-voice-add').disabled = false;
    }
  }
}
