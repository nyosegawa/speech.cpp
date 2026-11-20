// Runs the WebAssembly build of speech.cpp in a headless Chrome through its page (wasm/index.html), as a script drives
// it, and fails on a defect: the regions where someone speaks in each WAVE file must be those `speech vad` finds in it on
// the CPU, number for number; the build must refuse more threads than it starts, a model file cut short and an integer
// option past what the C API takes, as errors naming the input rather than an abort or a value of another size, and
// answer the next request after them; a run must keep its model while another loads one. It prints the time to load
// the model, the RTFx of each file and the most memory the module held. The page is served from 127.0.0.1 with COOP and
// COEP, which the build with threads needs, and the files given at /files/. With --threads N the page runs the build
// with threads on N threads, and audio of more than its 1 GB must be refused as out_of_memory before the module writes
// it, after which the module still answers. Chrome is the one at $CHROME, or macOS's; the server and Chrome are stopped
// however the script ends.
// usage: node checks/smoke/wasm_browser_smoke.mjs <speech> <wasm build dir> <detection model> <audio.wav>... [--threads N]
import { spawn, execFileSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';

const argv = process.argv.slice(2);
const threadsAt = argv.indexOf('--threads');
const threads = threadsAt >= 0 ? Number(argv[threadsAt + 1]) : null;
if (threadsAt >= 0) argv.splice(threadsAt, 2);
const [speech, build, model, ...audio] = argv;
if (!audio.length || (threadsAt >= 0 && !(threads > 1))) {
  console.error('usage: node checks/smoke/wasm_browser_smoke.mjs <speech> <wasm build dir> <detection model> <audio.wav>... [--threads N, 2 or more]');
  process.exit(2);
}
const library = threads ? 'libspeech-threads' : 'libspeech';
const CHROME = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

/** A child process, and a promise that rejects when it cannot start or exits before it is stopped. */
function start(command, args, options) {
  const child = spawn(command, args, options);
  let stopping = false;
  const failed = new Promise((_, reject) => {
    child.on('error', (e) => reject(new Error(`${command} could not start: ${e.message}`)));
    child.on('exit', (code) => stopping || reject(new Error(`${command} exited with ${code}`)));
  });
  failed.catch(() => {});
  return { child, failed, stop: () => ((stopping = true), child.kill()) };
}

/** `promise`, or the first failure of `processes`, or a timeout. */
function racing(promise, processes, ms, what) {
  let timer;
  const timeout = new Promise((_, reject) => (timer = setTimeout(() => reject(new Error(`timed out waiting for ${what}`)), ms)));
  return Promise.race([promise, timeout, ...processes.map((p) => p.failed)]).finally(() => clearTimeout(timer));
}

const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm' };

/** Serves the build's files at / and `files` at /files/<index>/<name>, cross-origin isolated. */
function serve(files) {
  const root = path.resolve(build);
  return http.createServer((request, response) => {
    const url = decodeURIComponent(new URL(request.url, 'http://x').pathname);
    const given = /^\/files\/(\d+)\//.exec(url);
    const file = given ? files[Number(given[1])] : path.join(root, url === '/' ? 'index.html' : url);
    if (!file || (!given && !file.startsWith(root + path.sep)) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, {
      'Content-Type': types[path.extname(file)] ?? 'application/octet-stream',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cache-Control': 'no-store',
    });
    fs.createReadStream(file).pipe(response);
  });
}

/** Where `got`, the page's regions of each file, departs from `expected`, `speech vad`'s, or null where it does not. */
function departure(got, expected) {
  for (let f = 0; f < expected.length; f++) {
    const [g, e] = [got[f].regions, expected[f]];
    for (let i = 0; i < Math.max(g.length, e.length); i++) {
      if (g[i]?.start !== e[i]?.start || g[i]?.end !== e[i]?.end) {
        return `${audio[f]}: region ${i + 1} is ${JSON.stringify(g[i] ?? null)} in the page and ${JSON.stringify(e[i] ?? null)} in speech vad `
               + `(${g.length} and ${e.length} regions)`;
      }
    }
  }
  return null;
}

const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'wasm-smoke-chrome-'));
const running = [];
let ws = null;
let server = null;
try {
  for (const file of [`${library}.js`, `${library}.wasm`, 'speech.mjs', 'index.html']) {
    if (!fs.existsSync(path.join(build, file))) {
      throw new Error(`${build} has no ${file}; build it with emcmake cmake${threads ? ' -DSPEECH_WASM_THREADS=ON' : ''} and cmake --build`);
    }
  }
  const lines = execFileSync(speech, ['vad', model, '--format', 'json', ...audio], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'] });
  const native = lines.trim().split('\n').map((line) => JSON.parse(line).regions);

  const files = [model, ...audio].map((f) => path.resolve(f));
  server = serve(files);
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  const urls = files.map((f, i) => `/files/${i}/${encodeURIComponent(path.basename(f))}`);

  const chrome = start(CHROME, ['--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--no-first-run'], { stdio: 'ignore' });
  running.push(chrome);
  const port = await racing((async () => {
    for (;;) {
      try {
        return fs.readFileSync(path.join(profile, 'DevToolsActivePort'), 'utf8').split('\n')[0];
      } catch {
        // Chrome writes the file once it listens.
        await wait(100);
      }
    }
  })(), [chrome], 30000, 'Chrome to listen');
  const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
  await racing(new Promise((resolve) => ws.addEventListener('open', resolve)), [chrome], 30000, 'Chrome\'s page');
  let id = 0;
  const pending = new Map();
  const errors = [];
  ws.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
    } else if (message.method === 'Runtime.exceptionThrown') {
      errors.push(message.params.exceptionDetails.exception?.description ?? message.params.exceptionDetails.text);
    }
  });
  const send = (method, params = {}) => racing(new Promise((resolve) => {
    const n = ++id;
    pending.set(n, resolve);
    ws.send(JSON.stringify({ id: n, method, params }));
  }), running, 300000, method);
  /** The value of `body`, an async function's body, run in the page; a run's error comes back as { error }. */
  const page = async (body) => {
    const expression = `(async () => {
      const failure = (e) => ({ error: { status: e.status ?? null, option: e.option ?? null, message: e.message } });
      ${body}
    })()`;
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description ?? JSON.stringify(r.result.exceptionDetails));
    return r.result?.result?.value;
  };

  await send('Runtime.enable');
  await send('Page.enable');
  await send('Page.navigate', { url: `${origin}/index.html` });
  await racing((async () => {
    while (!(await page('return !!window.speechPage;'))) await wait(100);
  })(), running, 30000, 'the page');
  const isolated = await page('return crossOriginIsolated;');
  if (!isolated) throw new Error('the page is not cross-origin isolated, though the server sends COOP and COEP');

  const run = (args) => page(`return await speechPage.run(${args}).catch(failure);`);
  const on = threads ? `, threads: ${threads}` : '';
  const all = `{ model: ${JSON.stringify(urls[0])}, audio: ${JSON.stringify(urls.slice(1))}${on} }`;
  const first = await run(all);
  if (first.error) throw new Error(`the page failed: ${JSON.stringify(first.error)}`);
  const differs = departure(first.detections, native);
  if (differs) throw new Error(differs);
  if (first.info.threads !== (threads ?? 1)) throw new Error(`the model computes with ${first.info.threads} threads, where ${threads ?? 1} were asked for`);
  console.log(`speech.cpp ${first.version}, ${library}: the module started in ${first.open.toFixed(1)} ms, and ${first.info.name} `
              + `${first.info.weight_type} loaded in ${first.load.toFixed(1)} ms on ${first.info.threads} threads`);
  first.detections.forEach((d, f) => {
    console.log(`${path.basename(audio[f])}: ${d.regions.length} regions as speech vad finds them, ${d.seconds.toFixed(2)} s of audio in `
                + `${d.ms.toFixed(1)} ms, RTFx ${(d.seconds / (d.ms / 1000)).toFixed(0)}`);
  });
  console.log(`the module's memory: ${(first.memory / 2 ** 20).toFixed(1)} MiB at most`);

  const one = `audio: ${JSON.stringify(urls.slice(1, 2))}${on}`;
  // One thread more than the build starts: 2 without threads, or one past the browser's logical processors with them.
  const most = threads ? await page('return navigator.hardwareConcurrency;') : 1;
  const many = await run(`{ model: ${JSON.stringify(urls[0])}, audio: ${JSON.stringify(urls.slice(1, 2))}, threads: ${most + 1}, threaded: ${!!threads} }`);
  if (many.error?.status !== 'invalid_argument' || many.error.option !== 'threads') {
    throw new Error(`${most + 1} threads in ${library}: ${JSON.stringify(many)}, where invalid_argument naming threads was due`);
  }
  console.log(`${most + 1} threads refused: ${many.error.status} (${many.error.option}): ${many.error.message}`);
  // The model cut in half is loaded while a run of the whole one is under way, which must keep its own model. Both are
  // Blobs at hand, so that the second load reaches the Worker before the first run's detection.
  const [whole, cut] = await page(`
    const whole = await (await fetch(${JSON.stringify(urls[0])})).blob();
    const model = new File([whole.slice(0, Math.floor(whole.size / 2))], 'cut.gguf');
    return await Promise.all([
      speechPage.run({ model: new File([whole], 'whole.gguf'), ${one} }).catch(failure),
      speechPage.run({ model, ${one} }).catch(failure),
    ]);
  `);
  if (cut.error?.status !== 'model_file') throw new Error(`a model cut in half: ${JSON.stringify(cut)}, where model_file was due`);
  if (whole.error) throw new Error(`a run with a model cut in half after it failed: ${JSON.stringify(whole.error)}`);
  const differsBeside = departure(whole.detections, native.slice(0, 1));
  if (differsBeside) throw new Error(`a run with a model cut in half after it: ${differsBeside}`);
  console.log(`a model cut in half refused: ${cut.error.status}: ${cut.error.message}; the run before it kept its own model`);
  // A BigInt past 64 bits would reach the library modulo 2^64, as 0, and a Number past 2^53 as another integer.
  for (const [value, status] of [['-(2n ** 64n)', 'out_of_range'], ['2 ** 60', 'invalid_argument']]) {
    const wide = await run(`{ model: ${JSON.stringify(urls[0])}, ${one}, options: { min_speech_duration_ms: ${value} } }`);
    if (wide.error?.status !== status || wide.error.option !== 'min_speech_duration_ms') {
      throw new Error(`min_speech_duration_ms of ${value}: ${JSON.stringify(wide)}, where ${status} naming it was due`);
    }
    console.log(`min_speech_duration_ms of ${value} refused: ${wide.error.status}: ${wide.error.message}`);
  }
  if (threads) {
    // Audio past the 1 GB of the build with threads, in a Worker of its own, since the page reads audio from WAVE files:
    // malloc() gives 0 for it, and the module must refuse it before it writes there, then answer the next request.
    const big = await page(`
      const source = 'import { openSpeech } from ' + JSON.stringify(location.origin + '/speech.mjs') + ';'
        + 'onmessage = async ({ data: { blob, threads } }) => {'
        + '  const speech = await openSpeech({ threads: true });'
        + '  const model = speech.load(blob, { threads });'
        + '  const failed = (e) => ({ status: e.status ?? null, message: e.message });'
        + '  let big;'
        + '  try { big = { regions: model.detect(new Float32Array(2 ** 28 + 2 ** 20), 16000) }; } catch (e) { big = failed(e); }'
        + '  let after;'
        + '  try { after = { regions: model.detect(new Float32Array(16000), 16000) }; } catch (e) { after = failed(e); }'
        + '  postMessage({ big, after, memory: speech.memory });'
        + '};';
      const worker = new Worker(URL.createObjectURL(new Blob([source], { type: 'text/javascript' })), { type: 'module' });
      const blob = await (await fetch(${JSON.stringify(urls[0])})).blob();
      const answer = new Promise((resolve, reject) => {
        worker.onmessage = ({ data }) => resolve(data);
        worker.onerror = (e) => reject(new Error(e.message));
      });
      worker.postMessage({ blob, threads: ${threads} });
      try {
        return await answer;
      } finally {
        worker.terminate();
      }
    `);
    if (big.big.status !== 'out_of_memory' || !big.after.regions || big.after.regions.length) {
      throw new Error(`audio past 1 GB: ${JSON.stringify(big)}, where out_of_memory and then a second of silence without regions were due`);
    }
    console.log(`audio past 1 GB refused: ${big.big.status}: ${big.big.message}; a second of silence after it: no regions`);
  }
  const again = await run(`{ model: ${JSON.stringify(urls[0])}, ${one} }`);
  if (again.error) throw new Error(`the page failed after the errors: ${JSON.stringify(again.error)}`);
  const differsAgain = departure(again.detections, native.slice(0, 1));
  if (differsAgain) throw new Error(`after the errors, ${differsAgain}`);
  console.log('the next request after them: the same regions');

  if (errors.length) throw new Error(`the page threw: ${errors.join('; ')}`);
  console.log('ok');
} finally {
  ws?.close();
  server?.close();
  for (const child of running.reverse()) child.stop();
  await wait(500);
  fs.rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}
