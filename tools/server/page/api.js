// The server's endpoints as the page calls them: its own under /speech/, and OpenAI's audio endpoints, which any client
// calls. Every failure is thrown as a ServerError carrying OpenAI's error object, so that the page shows the server's own
// words next to their cause.

/** A failure the server answered: OpenAI's error object, whose param names the input at fault. */
export class ServerError extends Error {
  constructor(status, error) {
    super(error?.message ?? `The server answered ${status}.`);
    this.name = 'ServerError';
    this.status = status;
    this.param = error?.param ?? null;
    this.code = error?.code ?? null;
  }
}

async function failure(response) {
  let error = null;
  try {
    error = (await response.json()).error;
  } catch {
    // A body that is not JSON leaves the status alone to tell what failed.
  }
  return new ServerError(response.status, error);
}

async function call(path, init = {}) {
  const response = await fetch(path, init);
  if (!response.ok) throw await failure(response);
  return response;
}

/** The objects of a stream of server-sent events, each from its "data:" line. */
async function* events(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value;
    for (let end = buffer.indexOf('\n\n'); end >= 0; end = buffer.indexOf('\n\n')) {
      const data = buffer.slice(0, end).split('\n').filter((line) => line.startsWith('data: ')).map((line) => line.slice(6)).join('\n');
      buffer = buffer.slice(end + 2);
      if (data) yield JSON.parse(data);
    }
  }
}

/** The catalog, where each of its files is and how much of it is there, and the model held for each task. */
export async function models() {
  return (await call('/speech/models')).json();
}

/**
 * Fetches a model of the catalog and loads it in place of its task's, yielding {type: "fetch", done, total},
 * {type: "note", message}, {type: "load"} and lastly {type: "loaded", task, held}. Aborting `signal` stops the fetch,
 * which the next load resumes.
 */
export async function* load(name, signal) {
  const response = await call('/speech/load', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: name }),
    signal,
  });
  for await (const event of events(response)) {
    if (event.type === 'error') throw new ServerError(500, event.error);
    yield event;
  }
}

/** Adds a voice made from a WAVE recording to the synthesis model, and returns the model as the server now holds it. */
export async function addVoice(name, recording) {
  const form = new FormData();
  form.append('name', name);
  form.append('file', recording, 'recording.wav');
  return (await (await call('/speech/voices', { method: 'POST', body: form })).json()).held;
}

/**
 * Speaks `request`, a create speech request without its format, as a stream of events: {type: "start", rate}, then
 * {type: "audio", samples} with each chunk of 16-bit samples as it is made, and lastly {type: "done", seed, samples,
 * stop}. Aborting `signal` stops the request.
 */
export async function* speak(request, signal) {
  const response = await call('/v1/audio/speech', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...request, response_format: 'pcm', stream_format: 'sse' }),
    signal,
  });
  yield { type: 'start', rate: Number(response.headers.get('X-Sample-Rate')) };
  for await (const event of events(response)) {
    if (event.type === 'error') throw new ServerError(500, event.error);
    if (event.type === 'speech.audio.delta') yield { type: 'audio', samples: pcm(event.audio) };
    if (event.type === 'speech.audio.done') {
      yield { type: 'done', seed: event.seed, samples: event.samples, stop: event.stop };
      return;
    }
  }
  throw new ServerError(500, { message: 'The speech broke off before its end.' });
}

/** 16-bit little-endian samples from base64. */
function pcm(base64) {
  const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  const samples = new Int16Array(bytes.length >> 1);
  for (let i = 0; i < samples.length; i++) samples[i] = view.getInt16(2 * i, true);
  return samples;
}

/**
 * Transcribes a WAVE file with the form's `fields` (language, prompt, decoding), as verbose_json, which carries the
 * language heard and the segments where the model gives them.
 */
export async function transcribe(recording, fields, signal) {
  const form = new FormData();
  form.append('file', recording, 'recording.wav');
  for (const [name, value] of Object.entries(fields)) form.append(name, String(value));
  form.append('response_format', 'verbose_json');
  const response = await call('/v1/audio/transcriptions', { method: 'POST', body: form, signal });
  return { result: await response.json(), stop: response.headers.get('X-Speech-Stop') };
}
