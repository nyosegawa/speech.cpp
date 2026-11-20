// Reads a WAVE file into mono float32 samples as `speech` reads one (src/common/wav.cpp): 16-, 24- and 32-bit PCM and
// 32-bit float, every channel averaged in float32, so that a page gives the library the samples the command line gives
// it for the same file, and the same results.
//
// usage:
//   import { readWav } from './wav.mjs';
//   const { sampleRate, samples } = readWav(await file.arrayBuffer(), file.name);

const FORMAT_PCM = 1, FORMAT_FLOAT = 3, FORMAT_EXTENSIBLE = 0xfffe;

/** The mono samples and the sample rate of the WAVE file in `buffer`, an ArrayBuffer; `name` names it in errors. */
export function readWav(buffer, name = 'the audio') {
  const view = new DataView(buffer);
  const tag = (at) => String.fromCharCode(view.getUint8(at), view.getUint8(at + 1), view.getUint8(at + 2), view.getUint8(at + 3));
  if (buffer.byteLength < 12 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') throw new Error(`${name} is not a WAVE file`);
  let format = 0, bits = 0, channels = 0, sampleRate = 0, data = -1, dataSize = 0;
  for (let at = 12; at + 8 <= buffer.byteLength;) {
    const size = view.getUint32(at + 4, true);
    if (at + 8 + size > buffer.byteLength) throw new Error(`${name} ends inside a chunk`);
    if (tag(at) === 'fmt ' && size >= 16) {
      format = view.getUint16(at + 8, true);
      channels = view.getUint16(at + 10, true);
      sampleRate = view.getUint32(at + 12, true);
      bits = view.getUint16(at + 22, true);
      if (format === FORMAT_EXTENSIBLE && size >= 40) format = view.getUint16(at + 32, true);
    } else if (tag(at) === 'data') {
      data = at + 8;
      dataSize = size;
    }
    at += 8 + size + (size & 1);
  }
  if (data < 0 || channels === 0) throw new Error(`${name} has no format or no data`);
  const pcm = format === FORMAT_PCM && (bits === 16 || bits === 24 || bits === 32);
  const ieee = format === FORMAT_FLOAT && bits === 32;
  if (!pcm && !ieee) {
    throw new Error(`${name} is neither 16-, 24- or 32-bit PCM nor 32-bit float WAVE (format ${format}, ${bits} bits)`);
  }
  const width = bits / 8;
  const frames = Math.floor(Math.floor(dataSize / width) / channels);
  const sample = (i) => {
    const at = data + i * width;
    if (ieee) {
      const x = view.getFloat32(at, true);
      if (!Number.isFinite(x)) throw new Error(`${name} holds a sample that is not a finite number, sample ${i}; write the recording again from its source`);
      return x;
    }
    if (bits === 16) return view.getInt16(at, true) / 32768;
    if (bits === 24) return ((view.getUint8(at) << 8 | view.getUint8(at + 1) << 16 | view.getUint8(at + 2) << 24) >> 8) / 8388608;
    return Math.fround(view.getInt32(at, true) / 2147483648);
  };
  const samples = new Float32Array(frames);
  for (let f = 0; f < frames; f++) {
    if (channels === 1) {
      samples[f] = sample(f);
      continue;
    }
    let sum = 0;
    for (let c = 0; c < channels; c++) sum = Math.fround(sum + sample(f * channels + c));
    samples[f] = sum / channels;
  }
  return { sampleRate, samples };
}
