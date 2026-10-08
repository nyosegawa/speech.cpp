// The text a panel transcribed, as a card: the text with a copy button, the language the model heard, the length of the
// audio and the time it took, and the segments with their times where the model gives them. The transcribe panel and
// the live panel each have one.

import { languageName } from './languages.js';
import { join } from './pieces.js';

function seconds(value) {
  return `${value.toFixed(2)} s`;
}

/** A length as a user reads it: "12.3 s" under a minute, "1:02:05" or "4:07" above. */
export function length(value) {
  if (value < 60) return `${value.toFixed(1)} s`;
  const s = Math.floor(value % 60).toString().padStart(2, '0');
  const m = Math.floor(value / 60) % 60;
  return value < 3600 ? `${m}:${s}` : `${Math.floor(value / 3600)}:${m.toString().padStart(2, '0')}:${s}`;
}

export class TranscriptView {
  /** Fills `slot` from the page's transcript template; the card stays hidden until there is something to show. */
  constructor(slot) {
    slot.append(document.querySelector('#transcript-view').content.cloneNode(true));
    const $ = (selector) => slot.querySelector(selector);
    this.card = $('.transcript');
    this.text = $('.transcript-text');
    this.copy = $('.transcript-copy');
    this.meta = $('.meta');
    this.segments = $('.segments');
    const heading = $('h3');
    heading.id = `${slot.id}-heading`;
    this.card.setAttribute('aria-labelledby', heading.id);

    this.copy.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(this.text.textContent);
        this.copy.textContent = 'Copied';
      } catch {
        // A browser that keeps the clipboard from the page still lets the user copy the text, selected for them.
        getSelection().selectAllChildren(this.text);
        this.copy.textContent = 'Selected; copy it';
      }
      setTimeout(() => (this.copy.textContent = 'Copy'), 2000);
    });
  }

  /**
   * Shows `transcript` of `model`, with `provisional`, the text not yet final, after it in the muted colour; `done` is
   * false while more text is to come, `took` is the seconds it took, and `live` says it was transcribed as it was said.
   */
  render(transcript, model, { provisional = '', done = true, took = null, live = false }) {
    this.card.hidden = false;
    this.text.replaceChildren(transcript.text);
    if (provisional) {
      const rest = document.createElement('span');
      rest.className = 'provisional';
      rest.textContent = join(transcript.text, provisional).slice(transcript.text.length);
      this.text.append(rest);
    }
    if (!transcript.text && !provisional) this.text.textContent = done ? '(no speech)' : '…';
    this.copy.hidden = !done || !transcript.text;
    const meta = [];
    if (transcript.languages.length) meta.push(transcript.languages.map(languageName).join(', '));
    if (transcript.seconds) meta.push(`${length(transcript.seconds)} of audio${took === null ? '' : ` in ${length(took)}`}`);
    if (live) meta.push('transcribed as it was said');
    meta.push(model.name);
    if (transcript.stop === 'model_limit') meta.push('a piece stopped at the most text the model writes');
    this.meta.textContent = meta.join(' · ');
    const rows = this.segments.querySelector('tbody');
    rows.replaceChildren();
    for (const segment of transcript.segments) {
      const row = rows.insertRow();
      row.insertCell().textContent = seconds(segment.start);
      row.insertCell().textContent = seconds(segment.end);
      row.insertCell().textContent = segment.text;
    }
    const count = rows.rows.length;
    this.segments.querySelector('summary').textContent = `${count} segment${count === 1 ? '' : 's'} with their times`;
    this.segments.hidden = rows.rows.length === 0;
  }
}
