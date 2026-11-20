// The page in index.html: finds the regions where someone speaks in a WAVE file with a detection model, both chosen by
// the person, through speech.mjs in a Worker (worker.mjs), and shows the regions, the load time, the RTFx and the module's
// memory. window.speechPage.run() does the same for a script, such as checks/smoke/wasm_browser_smoke.mjs, from URLs or
// Blobs: run({ model, audio: [...], threads, threaded, options }) loads the model once and detects in each audio in
// turn, in the build with threads where `threaded` says, by default where more than one thread is asked for. A link with
// ?model=URL&audio=URL&threads=N runs on those files as the page opens, as on a phone, where a script cannot reach the
// page. The build with threads needs the page cross-origin isolated.

const $ = (id) => document.getElementById(id);

let worker = null;
let opened = null;
let threaded = null;
let next = 0;
const waiting = new Map();

/** Sends `op` to the Worker and resolves with its value, or rejects with its error. */
function call(op, args = {}) {
  return new Promise((resolve, reject) => {
    const id = ++next;
    waiting.set(id, { resolve, reject });
    worker.postMessage({ id, op, ...args });
  });
}

function answered({ data: { id, value, error } }) {
  const { resolve, reject } = waiting.get(id);
  waiting.delete(id);
  if (!error) return resolve(value);
  const e = new Error(error.message);
  Object.assign(e, error);
  reject(e);
}

/** The Worker with the library loaded, the build with threads or the one without, started again for the other build. */
function library(threads) {
  if (opened && threaded !== threads) {
    worker.terminate();
    opened = null;
  }
  if (!opened) {
    worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });
    worker.onmessage = answered;
    threaded = threads;
    opened = call('open', { threads });
  }
  return opened;
}

/** A Blob of `source`, a Blob as it is or a URL fetched, named by the URL's last part. */
async function blobOf(source) {
  if (source instanceof Blob) return source;
  const response = await fetch(source);
  if (!response.ok) throw new Error(`${source}: ${response.status} ${response.statusText}`);
  return new File([await response.blob()], decodeURIComponent(new URL(source, location.href).pathname.split('/').pop()));
}

/** Loads `model` and detects speech in each of `audio`, with the threads and the request options given. */
async function runNow({ model, audio, threads, threaded = threads > 1, options = {} }) {
  const open = await library(threaded);
  const loaded = await call('load', { blob: await blobOf(model), threads });
  const detections = [];
  for (const source of audio) detections.push(await call('detect', { blob: await blobOf(source), options }));
  return { version: open.version, open: open.ms, load: loaded.ms, info: loaded.info, detections, memory: detections.at(-1)?.memory ?? loaded.memory };
}

let queue = Promise.resolve();

/**
 * runNow() once the runs before it have ended: the Worker holds one model, which a run that loads another while one
 * detects would replace or free under it.
 */
function run(args) {
  const result = queue.then(() => runNow(args));
  queue = result.catch(() => {});
  return result;
}

const seconds = (s) => `${s.toFixed(3)} s`;

function show(result) {
  const detection = result.detections[0];
  $('version').textContent = `speech.cpp ${result.version}`;
  $('stats').replaceChildren(...[
    ['Model', `${result.info.name} (${result.info.weight_type})`],
    ['Threads', String(result.info.threads)],
    ['Load time', `${result.load.toFixed(1)} ms`],
    ['Audio', seconds(detection.seconds)],
    ['Detection', `${detection.ms.toFixed(1)} ms`],
    ['RTFx', (detection.seconds / (detection.ms / 1000)).toFixed(0)],
    ['Memory', `${(result.memory / 2 ** 20).toFixed(1)} MiB`],
  ].flatMap(([term, value]) => {
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = term;
    dd.textContent = value;
    return [dt, dd];
  }));
  $('regions').replaceChildren(...detection.regions.map((r, i) => {
    const row = document.createElement('tr');
    for (const text of [String(i + 1), seconds(r.start), seconds(r.end)]) {
      const cell = document.createElement('td');
      cell.textContent = text;
      row.append(cell);
    }
    return row;
  }));
  $('result').hidden = false;
}

/** Runs on the model and audio given, or the files chosen, with the threads chosen, showing the result or the error. */
async function detect({ model = $('model').files[0], audio = $('audio').files[0] } = {}) {
  $('error').hidden = true;
  $('detect').disabled = true;
  $('status').textContent = 'Detecting…';
  try {
    const options = $('options').value.trim() ? JSON.parse($('options').value) : {};
    show(await run({ model, audio: [audio], threads: Number($('threads').value), options }));
  } catch (e) {
    $('error').textContent = e.status ? `${e.status}${e.option ? ` (${e.option})` : ''}: ${e.message}` : e.message;
    $('error').hidden = false;
  } finally {
    $('status').textContent = '';
    $('detect').disabled = false;
  }
}

const ready = () => ($('detect').disabled = !($('model').files.length && $('audio').files.length));
$('model').addEventListener('change', ready);
$('audio').addEventListener('change', ready);
$('form').addEventListener('submit', (event) => {
  event.preventDefault();
  detect();
});
ready();

window.speechPage = { run };
$('threads').max = navigator.hardwareConcurrency;
$('isolation').textContent = crossOriginIsolated
  ? `More than 1 runs the build with threads, up to this browser's ${navigator.hardwareConcurrency}.`
  : 'This page is not cross-origin isolated, so it runs on 1 thread.';
$('threads').disabled = !crossOriginIsolated;
const linked = new URLSearchParams(location.search);
if (linked.has('threads')) $('threads').value = linked.get('threads');
if (linked.has('model') && linked.has('audio')) detect({ model: linked.get('model'), audio: linked.get('audio') });
