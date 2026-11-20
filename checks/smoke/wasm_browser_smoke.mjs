// Runs the WebAssembly build of speech.cpp in a headless Chrome through its page (wasm/index.html), as a script drives
// it, and fails on a defect: the regions where someone speaks in each WAVE file must be those `speech vad` finds in it on
// the CPU, number for number; the build must refuse more threads than it starts, and a model file cut short, as errors
// of the C API naming the input rather than an abort, and answer the next request after both. It prints the time to load
// the model, the RTFx of each file and the most memory the module held. The page is served from 127.0.0.1 with COOP and
// COEP, which the build with threads needs, and the files given at /files/. Chrome is the one at $CHROME, or macOS's;
// the server and Chrome are stopped however the script ends.
// usage: node checks/smoke/wasm_browser_smoke.mjs <speech> <wasm build dir> <detection model> <audio.wav>...
import { spawn, execFileSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';

const [speech, build, model, ...audio] = process.argv.slice(2);
if (!audio.length) {
  console.error('usage: node checks/smoke/wasm_browser_smoke.mjs <speech> <wasm build dir> <detection model> <audio.wav>...');
  process.exit(2);
}
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
  for (const file of ['libspeech.js', 'libspeech.wasm', 'speech.mjs', 'index.html']) {
    if (!fs.existsSync(path.join(build, file))) throw new Error(`${build} has no ${file}; build it with emcmake cmake and cmake --build`);
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
  const all = `{ model: ${JSON.stringify(urls[0])}, audio: ${JSON.stringify(urls.slice(1))} }`;
  const first = await run(all);
  if (first.error) throw new Error(`the page failed: ${JSON.stringify(first.error)}`);
  const differs = departure(first.detections, native);
  if (differs) throw new Error(differs);
  console.log(`speech.cpp ${first.version}: the module started in ${first.open.toFixed(1)} ms, and ${first.info.name} `
              + `${first.info.weight_type} loaded in ${first.load.toFixed(1)} ms`);
  first.detections.forEach((d, f) => {
    console.log(`${path.basename(audio[f])}: ${d.regions.length} regions as speech vad finds them, ${d.seconds.toFixed(2)} s of audio in `
                + `${d.ms.toFixed(1)} ms, RTFx ${(d.seconds / (d.ms / 1000)).toFixed(0)}`);
  });
  console.log(`the module's memory: ${(first.memory / 2 ** 20).toFixed(1)} MiB at most`);

  const one = `audio: ${JSON.stringify(urls.slice(1, 2))}`;
  const threads = await run(`{ model: ${JSON.stringify(urls[0])}, ${one}, threads: 2 }`);
  if (threads.error?.status !== 'invalid_argument' || threads.error.option !== 'threads') {
    throw new Error(`two threads in the build without threads: ${JSON.stringify(threads)}, where invalid_argument naming threads was due`);
  }
  console.log(`two threads refused: ${threads.error.status} (${threads.error.option}): ${threads.error.message}`);
  const cut = await page(`
    const whole = await (await fetch(${JSON.stringify(urls[0])})).blob();
    const model = new File([whole.slice(0, Math.floor(whole.size / 2))], 'cut.gguf');
    return await speechPage.run({ model, ${one} }).catch(failure);
  `);
  if (cut.error?.status !== 'model_file') throw new Error(`a model cut in half: ${JSON.stringify(cut)}, where model_file was due`);
  console.log(`a model cut in half refused: ${cut.error.status}: ${cut.error.message}`);
  const again = await run(`{ model: ${JSON.stringify(urls[0])}, ${one} }`);
  if (again.error) throw new Error(`the page failed after the errors: ${JSON.stringify(again.error)}`);
  const differsAgain = departure(again.detections, native.slice(0, 1));
  if (differsAgain) throw new Error(`after the errors, ${differsAgain}`);
  console.log('the next request after both: the same regions');

  if (errors.length) throw new Error(`the page threw: ${errors.join('; ')}`);
  console.log('ok');
} finally {
  ws?.close();
  server?.close();
  for (const child of running.reverse()) child.stop();
  await wait(500);
  fs.rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}
