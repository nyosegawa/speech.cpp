// The waveform of a speech, drawn as bars of its peaks with the part played in the accent colour; a range input laid
// over it moves the position, by pointer or by keyboard, and gives it its accessible name.

const BAR = 3;
const GAP = 1;

export class Waveform {
  #canvas;
  #seek;
  #chunks = [];
  #duration = 0;
  #position = 0;
  /** The peak of each bar, kept while the samples and the width stay the same, since the position moves each frame. */
  #peaks = null;
  #peaksOf = '';

  /** `onseek(seconds)` is called when the user moves the position. */
  constructor(canvas, seek, onseek) {
    this.#canvas = canvas;
    this.#seek = seek;
    seek.addEventListener('input', () => onseek(Number(seek.value) * this.#duration));
    new ResizeObserver(() => this.draw()).observe(canvas);
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.draw());
  }

  /** Shows the samples received, 16-bit chunks of a speech `duration` seconds long, and the position played. */
  update(chunks, duration, position) {
    this.#chunks = chunks;
    this.#duration = duration;
    this.#position = position;
    this.#seek.value = duration ? String(position / duration) : '0';
    this.#seek.setAttribute('aria-valuetext', `${position.toFixed(1)} of ${duration.toFixed(1)} seconds`);
    this.draw();
  }

  draw() {
    const canvas = this.#canvas;
    const scale = devicePixelRatio;
    const width = Math.round(canvas.clientWidth * scale);
    const height = Math.round(canvas.clientHeight * scale);
    if (!width || !height) return;
    if (canvas.width !== width || canvas.height !== height) Object.assign(canvas, { width, height });
    const g = canvas.getContext('2d');
    g.clearRect(0, 0, width, height);
    const style = getComputedStyle(canvas);
    const played = style.getPropertyValue('--wave-played').trim();
    const rest = style.getPropertyValue('--wave-rest').trim();
    const total = this.#chunks.reduce((n, c) => n + c.length, 0);
    const bars = Math.floor(width / ((BAR + GAP) * scale));
    if (!total || !bars) return;
    if (this.#peaksOf !== `${total}/${bars}`) {
      this.#peaks = this.#measure(total, bars);
      this.#peaksOf = `${total}/${bars}`;
    }
    const playedBars = this.#duration ? (this.#position / this.#duration) * bars : 0;
    // Bars are drawn against the loudest one, so that quiet speech fills the height as loud speech does.
    const loudest = Math.max(1, ...this.#peaks);
    for (let b = 0; b < bars; b++) {
      const h = Math.max(scale, Math.sqrt(this.#peaks[b] / loudest) * height);
      g.fillStyle = b < playedBars ? played : rest;
      g.fillRect(b * (BAR + GAP) * scale, (height - h) / 2, BAR * scale, h);
    }
  }

  #measure(total, bars) {
    const peaks = new Float32Array(bars);
    const perBar = total / bars;
    let i = 0;
    for (const chunk of this.#chunks) {
      for (const v of chunk) {
        const b = Math.min(bars - 1, Math.floor(i++ / perBar));
        peaks[b] = Math.max(peaks[b], Math.abs(v));
      }
    }
    return peaks;
  }
}
