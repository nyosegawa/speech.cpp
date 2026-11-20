// The Worker of the page in index.html: runs speech.mjs on the page's messages, one at a time, and answers each with
// its value or its error. A message is { id, op, ...arguments }: "open" loads the library, the build with threads or
// the one without; "load" loads a model from a Blob, freeing the one before; "detect" reads a WAVE Blob as `speech`
// reads the file and finds the regions where someone speaks. Each answer carries the milliseconds the library took and
// the bytes of its memory.
import { openSpeech } from './speech.mjs';
import { readWav } from './wav.mjs';

let speech = null;
let model = null;

const ops = {
  async open({ threads }) {
    const start = performance.now();
    speech = await openSpeech({ threads });
    return { version: speech.version, ms: performance.now() - start };
  },
  load({ blob, threads }) {
    model?.free();
    model = null;
    const start = performance.now();
    model = speech.load(blob, { threads });
    return { ms: performance.now() - start, info: model.info };
  },
  async detect({ blob, options }) {
    const { sampleRate, samples } = readWav(await blob.arrayBuffer(), blob.name);
    const start = performance.now();
    const regions = model.detect(samples, sampleRate, options);
    return { regions, seconds: samples.length / sampleRate, ms: performance.now() - start };
  },
};

onmessage = async ({ data: { id, op, ...args } }) => {
  try {
    const value = await ops[op](args);
    postMessage({ id, value: { ...value, memory: speech?.memory ?? 0 } });
  } catch (e) {
    postMessage({ id, error: { name: e.name, status: e.status ?? null, message: e.message, option: e.option ?? null } });
  }
};
