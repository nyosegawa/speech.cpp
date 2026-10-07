// WAVE files of one channel of 16-bit samples, the form the server reads and the page saves.

/** A WAVE file of `samples`, 16-bit integers or floats from -1 to 1, at `rate` samples a second. */
export function wavFile(samples, rate) {
  const bytes = new DataView(new ArrayBuffer(44 + 2 * samples.length));
  const text = (at, s) => [...s].forEach((c, i) => bytes.setUint8(at + i, c.charCodeAt(0)));
  text(0, 'RIFF');
  bytes.setUint32(4, 36 + 2 * samples.length, true);
  text(8, 'WAVEfmt ');
  bytes.setUint32(16, 16, true);
  bytes.setUint16(20, 1, true);
  bytes.setUint16(22, 1, true);
  bytes.setUint32(24, rate, true);
  bytes.setUint32(28, 2 * rate, true);
  bytes.setUint16(32, 2, true);
  bytes.setUint16(34, 16, true);
  text(36, 'data');
  bytes.setUint32(40, 2 * samples.length, true);
  const integers = samples instanceof Int16Array;
  for (let i = 0; i < samples.length; i++) {
    const v = integers ? samples[i] : Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767);
    bytes.setInt16(44 + 2 * i, v, true);
  }
  return new Blob([bytes.buffer], { type: 'audio/wav' });
}

/** Offers a file to the user to save. */
export function save(blob, name) {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 0);
}
