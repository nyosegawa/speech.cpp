// WAVE files of one channel of 16-bit samples, the form the server reads and the page saves.

/** The 44 bytes that begin a WAVE file of `count` 16-bit samples of one channel at `rate` samples a second. */
export function wavHeader(count, rate) {
  const bytes = new DataView(new ArrayBuffer(44));
  const text = (at, s) => [...s].forEach((c, i) => bytes.setUint8(at + i, c.charCodeAt(0)));
  text(0, 'RIFF');
  bytes.setUint32(4, 36 + 2 * count, true);
  text(8, 'WAVEfmt ');
  bytes.setUint32(16, 16, true);
  bytes.setUint16(20, 1, true);
  bytes.setUint16(22, 1, true);
  bytes.setUint32(24, rate, true);
  bytes.setUint32(28, 2 * rate, true);
  bytes.setUint16(32, 2, true);
  bytes.setUint16(34, 16, true);
  text(36, 'data');
  bytes.setUint32(40, 2 * count, true);
  return bytes.buffer;
}

/** `samples`, 16-bit integers or floats from -1 to 1, as little-endian 16-bit samples. */
export function pcm16(samples) {
  const bytes = new DataView(new ArrayBuffer(2 * samples.length));
  const integers = samples instanceof Int16Array;
  for (let i = 0; i < samples.length; i++) {
    const v = integers ? samples[i] : Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767);
    bytes.setInt16(2 * i, v, true);
  }
  return bytes.buffer;
}

/** Little-endian 16-bit frames of `channels` channels as one channel of floats, read as the server reads them. */
export function floats(buffer, channels = 1) {
  const bytes = new DataView(buffer);
  const samples = new Float32Array(Math.floor(buffer.byteLength / (2 * channels)));
  for (let i = 0; i < samples.length; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) sum += bytes.getInt16(2 * (i * channels + c), true);
    samples[i] = sum / channels / 32768;
  }
  return samples;
}

/**
 * Where the samples of a WAVE file of 16-bit PCM are, {rate, channels, offset, count} with `count` frames from byte
 * `offset`, so that a long file can be read a slice at a time; null for any other file, which is decoded whole.
 */
export async function pcmLayout(file) {
  const head = new DataView(await file.slice(0, 1 << 16).arrayBuffer());
  const tag = (at) => String.fromCharCode(...new Uint8Array(head.buffer, at, 4));
  if (head.byteLength < 12 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') return null;
  let format = null;
  for (let at = 12; at + 8 <= head.byteLength; ) {
    const size = head.getUint32(at + 4, true);
    if (tag(at) === 'fmt ' && at + 24 <= head.byteLength) {
      const code = head.getUint16(at + 8, true);
      // WAVE_FORMAT_EXTENSIBLE names PCM by the first two bytes of its sub-format's GUID.
      const extensible = code === 0xfffe && size >= 40 && at + 34 <= head.byteLength;
      const pcm = code === 1 || (extensible && head.getUint16(at + 32, true) === 1);
      const channels = head.getUint16(at + 10, true);
      if (!pcm || head.getUint16(at + 22, true) !== 16 || channels === 0) return null;
      format = { rate: head.getUint32(at + 12, true), channels };
    } else if (tag(at) === 'data') {
      if (!format) return null;
      // A file written to a pipe gives its data's size as 0xFFFFFFFF; the data then runs to the end of the file.
      const bytes = Math.min(size, file.size - (at + 8));
      return { ...format, offset: at + 8, count: Math.floor(bytes / (2 * format.channels)) };
    }
    at += 8 + size + (size & 1);
  }
  return null;
}

/** A WAVE file of `samples`, 16-bit integers or floats from -1 to 1, at `rate` samples a second. */
export function wavFile(samples, rate) {
  return new Blob([wavHeader(samples.length, rate), pcm16(samples)], { type: 'audio/wav' });
}

/** Offers a file to the user to save. */
export function save(blob, name) {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 0);
}
