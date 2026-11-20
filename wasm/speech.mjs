// speech.cpp's C API in a Worker, over the library compiled to WebAssembly: libspeech.js and libspeech.wasm. A model is
// read from a Blob or File through Emscripten's WORKERFS, which reads the file a piece at a time as the library asks for
// it, so that the weights are held once, in the module's memory. Each call blocks until the library returns, as the C
// API's do, and WORKERFS reads with FileReaderSync, which a Worker alone has: the module runs in a Worker.
//
// usage, in a module Worker:
//   import { openSpeech } from './speech.mjs';
//   const speech = await openSpeech();
//   const model = speech.load(file);                            // a Blob or File of a GGUF model
//   const regions = model.detect(samples, 16000, { threshold: 0.5 });   // [{ start, end }] in seconds
//   model.free();

const OK = 0;
const TYPE_STRING = 0, TYPE_INT = 1, TYPE_FLOAT = 2, TYPE_BOOL = 3;

/** A failure the library reports: its status ("invalid_argument", "model_file", ...), message and input at fault. */
export class SpeechError extends Error {
  constructor(status, message, option) {
    super(message);
    this.name = 'SpeechError';
    this.status = status;
    this.option = option;
  }
}

/** The library, loaded from libspeech.js beside this file, or from `url`. */
export async function openSpeech({ url } = {}) {
  const module = url ?? new URL('./libspeech.js', import.meta.url).href;
  const { default: create } = await import(module);
  return new Speech(await create());
}

/** Throws the library's error for `status` unless it is SPEECH_OK. */
function check(m, status) {
  if (status === OK) return;
  const option = m._speech_last_error_option();
  throw new SpeechError(m.UTF8ToString(m._speech_status_name(status)), m.UTF8ToString(m._speech_last_error()),
                        option ? m.UTF8ToString(option) : null);
}

/**
 * Runs `call` with the address of a word that the library writes, as the C API's out parameters, and returns the word:
 * a pointer, which a module of more than 2 GB holds above 2^31, read unsigned.
 */
function out(m, call) {
  const at = m._malloc(4) >>> 0;
  try {
    check(m, call(at));
    return m.HEAPU32[at >>> 2];
  } finally {
    m._free(at);
  }
}

/** Runs `call` with `text` in the module's memory as a UTF-8 C string. */
function withString(m, text, call) {
  const at = m.stringToNewUTF8(text) >>> 0;
  try {
    return call(at);
  } finally {
    m._free(at);
  }
}

/** Sets the request option `name` to `value`, a JavaScript value of the option's type. */
function setOption(m, request, name, value) {
  const option = withString(m, name, (text) => out(m, (at) => m._speech_option_from_name(text, at)));
  const type = m._speech_option_type(option);
  const refuse = (wanted) => {
    throw new SpeechError('invalid_argument', `the option ${name} takes ${wanted}, not ${typeof value} ${String(value)}`, name);
  };
  if (type === TYPE_STRING) {
    if (typeof value !== 'string') refuse('a string');
    withString(m, value, (text) => check(m, m._speech_request_set_string(request, option, text)));
  } else if (type === TYPE_INT) {
    if (typeof value !== 'bigint' && !Number.isInteger(value)) refuse('an integer');
    check(m, m._speech_request_set_int(request, option, BigInt(value)));
  } else if (type === TYPE_FLOAT) {
    if (typeof value !== 'number') refuse('a number');
    check(m, m._speech_request_set_float(request, option, value));
  } else if (type === TYPE_BOOL) {
    if (typeof value !== 'boolean') refuse('true or false');
    check(m, m._speech_request_set_bool(request, option, value ? 1 : 0));
  } else {
    throw new Error(`the option ${name} is of a type ${type} this module does not know`);
  }
}

let mounts = 0;

/** The library: loads models from Blobs. */
class Speech {
  #m;

  constructor(m) {
    this.#m = m;
  }

  /** The release of speech.cpp, as speech_version() gives it. */
  get version() {
    return this.#m.UTF8ToString(this.#m._speech_version());
  }

  /** The bytes of the module's memory, which grows as the library asks and never shrinks: the most it has used. */
  get memory() {
    return this.#m.HEAPU8.length;
  }

  /**
   * Loads the GGUF model in `blob` with the threads given, or the library's default, as speech_model_load() does. The
   * Blob is read as the library reads the file and must not change while the model is loaded.
   */
  load(blob, { threads } = {}) {
    const m = this.#m;
    const folder = `/models/${++mounts}`;
    const name = blob.name || 'model.gguf';
    m.FS.mkdirTree(folder);
    m.FS.mount(m.WORKERFS, { blobs: [{ name, data: blob }] }, folder);
    const unmount = () => {
      m.FS.unmount(folder);
      m.FS.rmdir(folder);
    };
    try {
      const params = out(m, (at) => m._speech_load_params_new(at));
      try {
        if (threads !== undefined) check(m, m._speech_load_params_set_threads(params, threads));
        const model = withString(m, `${folder}/${name}`, (path) => out(m, (at) => m._speech_model_load(path, params, at)));
        return new Model(m, model, unmount);
      } finally {
        m._speech_load_params_free(params);
      }
    } catch (e) {
      unmount();
      throw e;
    }
  }
}

/** A loaded model, which serves one request at a time. */
class Model {
  #m;
  #model;
  #unmount;

  constructor(m, model, unmount) {
    this.#m = m;
    this.#model = model;
    this.#unmount = unmount;
  }

  /** The model's information, as speech_model_info_json() writes it. */
  get info() {
    const m = this.#m;
    const info = out(m, (at) => m._speech_model_get_info(this.#live(), at));
    try {
      return JSON.parse(m.UTF8ToString(m._speech_model_info_json(info)));
    } finally {
      m._speech_model_info_free(info);
    }
  }

  /**
   * The regions where someone speaks in `samples`, a Float32Array of mono audio at `sampleRate` Hz, with the request
   * options in `options` by their names: [{ start, end }] in seconds, as speech_detect() gives them.
   */
  detect(samples, sampleRate, options = {}) {
    const m = this.#m;
    return this.#request(samples, sampleRate, options, (request) => m._speech_detect(request), (result) => {
      const n = m._speech_result_segment_count(result) >>> 0;
      const regions = [];
      const at = m._malloc(24) >>> 0;
      try {
        for (let i = 0; i < n; i++) {
          check(m, m._speech_result_segment(result, i, at, at + 8, at + 16));
          regions.push({ start: m.HEAPF64[at >>> 3], end: m.HEAPF64[(at >>> 3) + 1] });
        }
      } finally {
        m._free(at);
      }
      return regions;
    });
  }

  /** Frees the model; it cannot be used again. */
  free() {
    if (!this.#model) return;
    this.#m._speech_model_free(this.#model);
    this.#model = 0;
    this.#unmount();
  }

  #live() {
    if (!this.#model) throw new Error('the model has been freed');
    return this.#model;
  }

  /** Runs a request of `samples` with `options` through `run`, and reads its result with `read` before freeing it. */
  #request(samples, sampleRate, options, run, read) {
    const m = this.#m;
    if (!(samples instanceof Float32Array)) throw new TypeError('the samples are a Float32Array');
    const request = out(m, (at) => m._speech_request_new(this.#live(), at));
    try {
      const audio = m._malloc(Math.max(1, samples.length) * 4) >>> 0;
      try {
        m.HEAPF32.set(samples, audio >>> 2);
        check(m, m._speech_request_set_audio(request, audio, samples.length, sampleRate));
      } finally {
        m._free(audio);
      }
      for (const [name, value] of Object.entries(options)) setOption(m, request, name, value);
      check(m, run(request));
      return read(m._speech_request_result(request) >>> 0);
    } finally {
      m._speech_request_free(request);
    }
  }
}
