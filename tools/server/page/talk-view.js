// The conversation as the talk panel shows it: a lane for each side, the user above and the model below, each with an
// orb that swells with its voice and a band of beads, a column for each 150 ms of its audio, stacked symmetrically
// about the band's middle as high as the audio is loud. New audio enters at the right and flows left, both lanes on one
// clock, so that where the two speak at once, and where one answers the other, line up; silence is a dotted line. The
// model's text goes below in large letters, an utterance at a time.

/** The audio whose peak each peak kept stands for. */
const PEAK_SECONDS = 0.05;
/** The peaks a column of beads stands for. */
const PER_COLUMN = 3;
const COLUMN_SECONDS = PEAK_SECONDS * PER_COLUMN;
/** The seconds of the conversation a band shows. */
const SHOWN_SECONDS = 8;
/**
 * A column is silence, drawn as a dot, below the larger of this and FLOOR_TIMES the lane's noise floor: the microphone
 * is recorded without the gain control of calls, so a laptop's hums at 0.002 to 0.01 and its speech may peak at 0.02.
 */
const QUIET = 0.002;
const FLOOR_TIMES = 2.5;
/** The peak that fills a band's height; quieter audio takes the square root of its share, so that soft speech shows. */
const FULL = 0.3;
/** The pause in the model's audio after which its next text starts a new utterance. */
const PAUSE_SECONDS = 1.2;

/** The audio of one side: the peak of each 50 ms of it, at the time its end is heard. */
class Lane {
  #element;
  #canvas;
  #orb;
  /** [time, peak] of each 50 ms, oldest first, as far back as a band shows. */
  #peaks = [];
  #level = 0;

  constructor(element) {
    this.#element = element;
    this.#canvas = element.querySelector('canvas');
    this.#orb = element.querySelector('.talk-orb');
  }

  clear() {
    this.#peaks = [];
    this.#level = 0;
  }

  /** Adds `samples` at `rate` whose first one is heard at `time` seconds of the view's clock. */
  add(samples, rate, time) {
    const per = Math.round(PEAK_SECONDS * rate);
    for (let at = 0; at < samples.length; at += per) {
      let peak = 0;
      const end = Math.min(samples.length, at + per);
      for (let i = at; i < end; i++) peak = Math.max(peak, Math.abs(samples[i]));
      this.#peaks.push([time + end / rate, peak]);
    }
  }

  /** The peak below which the side is silent: above the lowest tenth of its peaks of the last seconds, its noise floor. */
  quiet(now) {
    const recent = this.#peaks.filter(([t]) => t > now - SHOWN_SECONDS && t <= now).map(([, peak]) => peak).sort((a, b) => a - b);
    return Math.max(QUIET, FLOOR_TIMES * (recent[Math.floor(recent.length / 10)] ?? 0));
  }

  /** Whether the side has made a sound above silence between `from` and `to`. */
  heardSince(from, to) {
    const quiet = this.quiet(to);
    return this.#peaks.some(([t, peak]) => t > from && t <= to && peak >= quiet);
  }

  draw(now) {
    while (this.#peaks.length && this.#peaks[0][0] < now - SHOWN_SECONDS - 1) this.#peaks.shift();
    const canvas = this.#canvas;
    const scale = devicePixelRatio;
    const width = Math.round(canvas.clientWidth * scale);
    const height = Math.round(canvas.clientHeight * scale);
    if (!width || !height) return;
    if (canvas.width !== width || canvas.height !== height) Object.assign(canvas, { width, height });
    const g = canvas.getContext('2d');
    g.clearRect(0, 0, width, height);
    const style = getComputedStyle(this.#element);
    const slot = (width * COLUMN_SECONDS) / SHOWN_SECONDS;
    const middle = height / 2;
    // Column k holds the audio from k * COLUMN_SECONDS of the clock, so that a column stays where it is as the audio of
    // its time arrives in pieces, and the dots of silence keep the same grid; it sits where its end's time puts it.
    const x = (k) => width - ((now - (k + 1) * COLUMN_SECONDS) / SHOWN_SECONDS) * width - slot / 2;
    const columns = new Map();
    let latest = 0;
    for (const [t, peak] of this.#peaks) {
      if (t > now || t <= now - SHOWN_SECONDS - COLUMN_SECONDS) continue;
      const k = Math.floor(Math.round(t / PEAK_SECONDS - 1) / PER_COLUMN);
      columns.set(k, Math.max(columns.get(k) ?? 0, peak));
      if (now - t < 3 * PEAK_SECONDS) latest = Math.max(latest, peak);
    }
    const quiet = this.quiet(now);
    const shown = [...columns.entries()].filter(([, peak]) => peak >= quiet).sort((a, b) => a[0] - b[0]);
    const level = (peak) => Math.min(1, Math.sqrt(peak / FULL));
    // Each column's height takes a quarter from each neighbour that sounds, so that the beads rise and fall smoothly.
    const heights = shown.map(([k, peak], i) => {
      const before = shown[i - 1]?.[0] === k - 1 ? shown[i - 1][1] : peak;
      const after = shown[i + 1]?.[0] === k + 1 ? shown[i + 1][1] : peak;
      return level(0.5 * peak + 0.25 * before + 0.25 * after) * height * 0.8;
    });
    const colour = style.getPropertyValue('--lane').trim();
    const light = style.getPropertyValue('--orb-light').trim();
    const bead = slot * 0.66;
    const step = bead * 1.25;
    shown.forEach(([k], i) => {
      // An odd count, so that a bead sits on the middle line, the outer half of them in the lighter colour.
      let count = Math.max(1, Math.round(heights[i] / step));
      if (count % 2 === 0) count += 1;
      const half = (count - 1) / 2;
      for (let j = -half; j <= half; j++) {
        g.fillStyle = half && Math.abs(j) / half > 0.5 ? light : colour;
        g.beginPath();
        g.arc(x(k), middle + j * step, bead / 2, 0, 2 * Math.PI);
        g.fill();
      }
    });
    const sounding = new Set(shown.map(([k]) => k));
    g.fillStyle = style.getPropertyValue('--lane-dots').trim();
    const last = Math.floor(now / COLUMN_SECONDS);
    for (let k = last; k > last - SHOWN_SECONDS / COLUMN_SECONDS; k--) {
      if (sounding.has(k)) continue;
      g.beginPath();
      g.arc(x(k), middle, 1.4 * scale, 0, 2 * Math.PI);
      g.fill();
    }
    // The orb follows the loudest of the last peaks heard, rising at once and falling slowly.
    this.#level = Math.max(latest >= quiet ? level(latest) : 0, this.#level * 0.9);
    this.#orb.style.setProperty('--level', this.#level.toFixed(3));
  }
}

export class TalkView {
  #you;
  #model;
  #name;
  #text;
  #frame = 0;
  /** The utterances of the model, the last one still growing. */
  #said = [];
  #lastText = -Infinity;

  /** Takes the stage of the talk panel: its two lanes, the model's name and the place of its text. */
  constructor(stage) {
    this.#you = new Lane(stage.querySelector('.talk-lane.talk-you'));
    this.#model = new Lane(stage.querySelector('.talk-lane.talk-model'));
    this.#name = stage.querySelectorAll('.talk-model-name');
    this.#text = stage.querySelector('.talk-text');
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.#draw());
  }

  /** The seconds of the view's clock, which both lanes are drawn on. */
  static now() {
    return performance.now() / 1000;
  }

  /** Starts a conversation with the model called `name`, clearing the last one. */
  start(name) {
    for (const element of this.#name) element.textContent = name;
    this.#you.clear();
    this.#model.clear();
    this.#said = [];
    this.#lastText = -Infinity;
    this.#text.replaceChildren();
    cancelAnimationFrame(this.#frame);
    const tick = () => {
      this.#draw();
      this.#frame = requestAnimationFrame(tick);
    };
    tick();
  }

  /** Stops moving, and leaves the conversation as it ended. */
  stop() {
    cancelAnimationFrame(this.#frame);
    this.#frame = 0;
  }

  /** The user's audio, its first sample heard at `time`. */
  heard(samples, rate, time) {
    this.#you.add(samples, rate, time);
  }

  /** The model's audio, its first sample played at `time`, with the text it began to say in it. */
  said(samples, rate, time, text) {
    this.#model.add(samples, rate, time);
    if (!text) return;
    // A pause in the model's speech before this text, or a while without text, starts a new utterance.
    const paused = !this.#model.heardSince(time - PAUSE_SECONDS, time) || time - this.#lastText > PAUSE_SECONDS * 2;
    if (paused || this.#said.length === 0) this.#said.push('');
    this.#said[this.#said.length - 1] += text;
    this.#lastText = time;
    // The text shows when its audio plays.
    setTimeout(() => this.#showText(), Math.max(0, (time - TalkView.now()) * 1000));
  }

  #showText() {
    const shown = this.#said.slice(-2);
    const lines = shown.map((utterance, i) => {
      const line = document.createElement('p');
      line.className = i === shown.length - 1 ? 'talk-line current' : 'talk-line';
      line.textContent = utterance.trim();
      return line;
    });
    this.#text.replaceChildren(...lines);
  }

  #draw() {
    const now = TalkView.now();
    this.#you.draw(now);
    this.#model.draw(now);
  }
}
