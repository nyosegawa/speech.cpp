// Long audio goes to the server in pieces, so that a request's size, the model's memory and the time of each request
// stay the same however long the audio is: a piece ends at the quietest tenth of a second of its last few seconds, where
// a cut is least likely to split a word, and the texts of the pieces are joined in order.

/** The stretch at the end of a full piece in which it is cut, and the stretch whose loudness is compared. */
const WINDOW_SECONDS = 8;
const FRAME_SECONDS = 0.1;

/** Cuts samples that arrive in any amounts into pieces of at most `maxSeconds`. */
export class PieceCutter {
  #rate;
  #max;
  #window;
  #buffer;
  #length = 0;
  /** The sample, counted from the beginning of the audio, at which the unfinished piece starts. */
  #start = 0;

  constructor(rate, maxSeconds) {
    this.#rate = rate;
    this.#max = Math.round(maxSeconds * rate);
    this.#window = Math.min(Math.round(WINDOW_SECONDS * rate), this.#max >> 1);
    this.#buffer = new Float32Array(this.#max);
  }

  /** Adds samples, and returns the pieces they finish, each {samples, start} with its start in seconds. */
  push(samples) {
    const finished = [];
    for (let at = 0; at < samples.length; ) {
      const n = Math.min(samples.length - at, this.#max - this.#length);
      this.#buffer.set(samples.subarray(at, at + n), this.#length);
      this.#length += n;
      at += n;
      if (this.#length === this.#max) finished.push(this.#take(this.#quietest()));
    }
    return finished;
  }

  /** The seconds of the unfinished piece. */
  get pendingSeconds() {
    return this.#length / this.#rate;
  }

  /** A copy of the unfinished piece, {samples, start}. */
  get pending() {
    return { samples: this.#buffer.slice(0, this.#length), start: this.#start / this.#rate };
  }

  /** Ends the unfinished piece and returns it, or null when there is none. */
  finish() {
    return this.#length ? this.#take(this.#length) : null;
  }

  /** The middle of the quietest frame in the window at the end of the buffer. */
  #quietest() {
    const frame = Math.max(1, Math.round(FRAME_SECONDS * this.#rate));
    let best = this.#length;
    let quietest = Infinity;
    for (let from = this.#length - this.#window; from + frame <= this.#length; from += frame) {
      let energy = 0;
      for (let i = from; i < from + frame; i++) energy += this.#buffer[i] * this.#buffer[i];
      if (energy < quietest) {
        quietest = energy;
        best = from + (frame >> 1);
      }
    }
    return best;
  }

  #take(end) {
    const piece = { samples: this.#buffer.slice(0, end), start: this.#start / this.#rate };
    this.#buffer.copyWithin(0, end, this.#length);
    this.#length -= end;
    this.#start += end;
    return piece;
  }
}

/** Scripts written without spaces between words, between whose texts a join adds none. */
const UNSPACED = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}　-〿＀-￯]/u;

/** Two texts as one: a space between them unless either side is written without spaces or already has one. */
export function join(a, b) {
  if (!a || !b || /\s$/.test(a) || /^\s/.test(b) || UNSPACED.test(a.at(-1)) || UNSPACED.test(b[0])) return a + b;
  return `${a} ${b}`;
}

/** The results of the pieces of one audio, joined in order. */
export class Transcript {
  text = '';
  segments = [];
  /** The languages heard, in the order of the audio, one for each run of pieces in the same language. */
  languages = [];
  seconds = 0;
  stop = 'complete';

  /** Adds the result of the piece that starts `start` seconds into the audio, and its stop reason. */
  add(result, stop, start) {
    this.text = join(this.text, result.text);
    for (const segment of result.segments ?? []) {
      this.segments.push({ ...segment, start: segment.start + start, end: segment.end + start });
    }
    for (const language of result.language ? result.language.split(',') : []) {
      if (this.languages.at(-1) !== language) this.languages.push(language);
    }
    this.seconds = start + result.duration;
    if (stop === 'model_limit') this.stop = stop;
  }
}
