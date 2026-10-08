// Drives the page of speech serve in a headless Chrome as a person does, and fails on a defect: the tabs, with the model
// pickers moving to the panel in view; the transcribe panel without a detection model, which transcribes audio up to a
// minute whole, as the API does, and asks for one for longer audio; the detection model chosen in its picker and loaded
// from the catalog; longer audio then transcribed by its regions, giving the text chunking_strategy "auto" gives; and the
// live panel, which waits for a detection model, transcribing a microphone that plays the dumps with pauses between
// them over /v1/realtime with the pause set in its options: grey text while an utterance is said, and at the end a text
// within a CER of 10% of the API's for the same audio and pause (the browser resamples the microphone to 24 kHz, so the
// samples are not the file's). The audio is
// 16-bit WAVE made of the dumps of reference/fastconformer/dump.py or reference/qwen3-asr/dump.py. Chrome is the one at
// $CHROME, or macOS's; the server and Chrome are stopped however the script ends.
// usage: node tools/server_page_browser_smoke.mjs <speech> <work dir> <recognition model> <detection model's catalog name> <dump folder>...
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';

const [speech, work, recognition, detection, ...dumps] = process.argv.slice(2);
if (!dumps.length) {
  console.error('usage: node tools/server_page_browser_smoke.mjs <speech> <work dir> <recognition model> <detection model\'s catalog name> <dump folder>...');
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

async function freePort() {
  const server = net.createServer();
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address();
  await new Promise((resolve) => server.close(resolve));
  return port;
}

/** The float32 samples of a dump's audio.npy. */
function readNpy(file) {
  const data = fs.readFileSync(file);
  const headerLength = data.readUInt16LE(8);
  const header = data.subarray(10, 10 + headerLength).toString();
  if (!header.includes("'<f4'")) throw new Error(`${file} is not float32`);
  const body = data.subarray(10 + headerLength);
  return new Float32Array(body.buffer.slice(body.byteOffset, body.byteOffset + body.length));
}

/** The character error rate of `got` against `want`, both without punctuation, symbols or spaces after NFKC. */
function cer(got, want) {
  const plain = (text) => [...text.normalize('NFKC').toLowerCase().replace(/[\p{P}\p{S}\s]/gu, '')];
  const a = plain(got), b = plain(want);
  let row = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const next = [i];
    for (let j = 1; j <= b.length; j++) next[j] = Math.min(row[j] + 1, next[j - 1] + 1, row[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    row = next;
  }
  return row[b.length] / Math.max(1, b.length);
}

/** A WAVE file of one channel of 16-bit samples, as the page writes one. */
function wav(samples, rate) {
  const out = Buffer.alloc(44 + 2 * samples.length);
  out.write('RIFF', 0);
  out.writeUInt32LE(36 + 2 * samples.length, 4);
  out.write('WAVEfmt ', 8);
  out.writeUInt32LE(16, 16);
  out.writeUInt16LE(1, 20);
  out.writeUInt16LE(1, 22);
  out.writeUInt32LE(rate, 24);
  out.writeUInt32LE(2 * rate, 28);
  out.writeUInt16LE(2, 32);
  out.writeUInt16LE(16, 34);
  out.write('data', 36);
  out.writeUInt32LE(2 * samples.length, 40);
  samples.forEach((x, i) => out.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(x * 32768))), 44 + 2 * i));
  return out;
}

const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'page-smoke-chrome-'));
const running = [];
let ws = null;
try {
  // Audio: the first dump alone, under a minute, and every dump with a second of silence after each, repeated past one.
  const rate = 16000;
  const first = readNpy(path.join(dumps[0], 'audio.npy'));
  if (first.length > 50 * rate) throw new Error(`${dumps[0]} is longer than 50 s; give a shorter dump first`);
  const once = [];
  for (const d of dumps) once.push(readNpy(path.join(d, 'audio.npy')), new Float32Array(rate));
  const spoken = new Float32Array(once.reduce((n, p) => n + p.length, 0));
  once.reduce((at, p) => (spoken.set(p, at), at + p.length), 0);
  const parts = [];
  for (let n = 0; n < 75 * rate; ) {
    for (const d of dumps) {
      const samples = readNpy(path.join(d, 'audio.npy'));
      parts.push(samples, new Float32Array(rate));
      n += samples.length + rate;
    }
  }
  const joined = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  parts.reduce((at, p) => (joined.set(p, at), at + p.length), 0);
  fs.mkdirSync(work, { recursive: true });
  const short = path.join(work, 'page-smoke-short.wav');
  const long = path.join(work, 'page-smoke-long.wav');
  fs.writeFileSync(short, wav(first, rate));
  fs.writeFileSync(long, wav(joined, rate));
  const microphone = path.join(work, 'page-smoke-microphone.wav');
  fs.writeFileSync(microphone, wav(spoken, rate));

  const server = start(speech, ['serve', recognition, '--port', String(await freePort())], { stdio: ['ignore', 'pipe', 'pipe'] });
  running.push(server);
  let log = '';
  const address = await racing(new Promise((resolve) => {
    const read = (chunk) => {
      log += chunk;
      const m = /is at (http:\/\/127\.0\.0\.1:\d+\/)/.exec(log);
      if (m) resolve(m[1]);
    };
    server.child.stdout.on('data', read);
    server.child.stderr.on('data', read);
  }), [server], 120000, 'speech serve to listen').catch((e) => {
    throw new Error(`${e.message}\n${log}`);
  });
  const origin = new URL(address).origin;

  /** The text of the API for `file`, with the form's other fields. */
  const api = async (file, fields = {}) => {
    const form = new FormData();
    form.append('file', new Blob([fs.readFileSync(file)], { type: 'audio/wav' }), 'x.wav');
    for (const [name, value] of Object.entries(fields)) form.append(name, value);
    const response = await fetch(`${origin}/v1/audio/transcriptions`, { method: 'POST', body: form });
    const body = await response.json();
    if (!response.ok) throw new Error(`the API answered ${response.status}: ${JSON.stringify(body)}`);
    return body.text;
  };

  const chrome = start(CHROME, [
    '--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--no-first-run',
    '--hide-scrollbars', '--lang=en-US', '--accept-lang=en-US', '--window-size=960,900',
    // A click the script makes is no gesture of a person's, which an audio context needs to run.
    '--autoplay-policy=no-user-gesture-required',
  ], { stdio: 'ignore' });
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
  /** Runs `body`, an async function's body, in the page with $ and until(check, what) defined. */
  const page = async (body) => {
    const expression = `(async () => {
      const $ = (id) => document.getElementById(id);
      const until = async (check, what, ms = 240000) => {
        const t = Date.now();
        while (!check()) {
          if (Date.now() - t > ms) throw new Error('timed out waiting for ' + what);
          await new Promise((r) => setTimeout(r, 100));
        }
      };
      ${body}
    })()`;
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description ?? JSON.stringify(r.result.exceptionDetails));
    return r.result?.result?.value;
  };
  /** Gives `file` to the transcribe panel's file input, as a person choosing it does. */
  const choose = async (file) => {
    const input = await send('Runtime.evaluate', { expression: "document.querySelector('#transcribe-audio .audio-file')" });
    await send('DOM.setFileInputFiles', { files: [path.resolve(file)], objectId: input.result.result.objectId });
  };
  /**
   * Transcribes the audio given in the transcribe panel and returns the card's text, or that the panel refused it, with
   * the requests it sent.
   */
  const transcribe = () => page(`
    await until(() => !$('transcribe-start').disabled, 'the transcribe button');
    window.__transcriptions.length = 0;
    $('transcribe-start').click();
    const card = document.querySelector('#transcribe-transcript .transcript');
    await until(() => !$('transcribe-start').disabled && !$('transcribe-status').textContent, 'the transcription');
    const requests = [...window.__transcriptions];
    return $('transcribe-error').hidden ? { text: card.querySelector('.transcript-text').textContent, requests } : { refused: true, requests };
  `);

  await send('Runtime.enable');
  await send('Page.enable');
  // The page's transcription requests, each with the fields of its form other than the file.
  // The microphone plays the dumps once, in real time, and then silence.
  await send('Page.addScriptToEvaluateOnNewDocument', { source: `
    navigator.mediaDevices.getUserMedia = async () => {
      const bytes = Uint8Array.from(atob(${JSON.stringify(fs.readFileSync(microphone).toString('base64'))}), (c) => c.charCodeAt(0));
      const context = new AudioContext();
      const source = context.createBufferSource();
      source.buffer = await context.decodeAudioData(bytes.buffer);
      const out = context.createMediaStreamDestination();
      source.connect(out);
      source.start();
      return out.stream;
    };
  ` });
  await send('Page.addScriptToEvaluateOnNewDocument', { source: `
    window.__sessions = [];
    const sent = WebSocket.prototype.send;
    WebSocket.prototype.send = function (data) {
      const event = JSON.parse(data);
      if (event.type === 'session.update') window.__sessions.push(event.session);
      return sent.call(this, data);
    };
    window.__transcriptions = [];
    const fetched = window.fetch;
    window.fetch = (url, init) => {
      if (String(url).endsWith('/v1/audio/transcriptions') && init?.body instanceof FormData) {
        window.__transcriptions.push(Object.fromEntries([...init.body].filter(([name]) => name !== 'file')));
      }
      return fetched(url, init);
    };
  ` });
  const pageLoaded = new Promise((resolve) => ws.addEventListener('message', (event) => {
    if (JSON.parse(event.data).method === 'Page.loadEventFired') resolve();
  }));
  await send('Page.navigate', { url: address.replace('127.0.0.1', 'localhost') });
  // Page.navigate answers once the new document is committed, which can be before the page's modules have run and
  // given the tabs their clicks; a click then is lost.
  await racing(pageLoaded, running, 60000, 'the page to load');
  // A picker sits only in the panel in view.
  await page(`
    $('tab-transcribe').click();
    await until(() => document.querySelector('#transcribe .picker-slot[data-task="recognition"] .picker-button:not(.empty)'), 'the recognition model');
  `);

  // The tabs: each picker sits in the panel in view, and the transcribe panel has the detection model's, empty.
  const tabs = await page(`
    const where = (task) => document.querySelector('.picker-slot[data-task="' + task + '"] .picker')?.closest('.panel')?.id ?? null;
    const seen = {};
    for (const tab of ['speak', 'transcribe', 'live']) {
      $('tab-' + tab).click();
      seen[tab] = { recognition: where('recognition'), detection: where('detection'), synthesis: where('synthesis'), hidden: $(tab).hidden };
    }
    $('tab-transcribe').click();
    const picker = document.querySelector('#transcribe .picker-slot[data-task="detection"] .picker');
    return { seen, empty: picker.querySelector('.picker-button').classList.contains('empty'), liveStarts: !$('live-start').disabled };
  `);
  if (tabs.seen.transcribe.recognition !== 'transcribe' || tabs.seen.live.recognition !== 'live' || tabs.seen.transcribe.detection !== 'transcribe'
      || tabs.seen.speak.synthesis !== 'speak' || Object.values(tabs.seen).some((s) => s.hidden)) {
    throw new Error(`the pickers did not follow the tabs: ${JSON.stringify(tabs.seen)}`);
  }
  if (!tabs.empty) throw new Error('the detection picker shows a model the server does not hold');
  if (tabs.liveStarts) throw new Error('the live panel starts without a detection model');
  console.log('tabs: the recognition picker in the transcribe and live panels, the detection picker in the transcribe panel, empty');

  // Without a detection model: audio under a minute whole, as the API transcribes it; longer audio refused.
  await choose(short);
  const whole = await transcribe();
  const wantWhole = await api(short);
  if (whole.text !== wantWhole || whole.requests.length !== 1 || 'chunking_strategy' in whole.requests[0]) {
    throw new Error(`the page without a detection model: ${JSON.stringify(whole)}, where the API gives ${JSON.stringify(wantWhole)} for one request without chunking_strategy`);
  }
  console.log(`transcribe without a detection model: ${(first.length / rate).toFixed(1)} s whole, the API's text: ${wantWhole.slice(0, 40)}`);
  await choose(long);
  const refused = await transcribe();
  if (!refused.refused || refused.requests.length) throw new Error(`audio over a minute without a detection model: ${JSON.stringify(refused)}`);
  console.log(`transcribe without a detection model: ${(joined.length / rate).toFixed(1)} s refused, asking for one`);

  // The detection model chosen in its picker, fetched already, and loaded in its place.
  const loaded = await page(`
    const picker = document.querySelector('#transcribe .picker-slot[data-task="detection"] .picker');
    picker.querySelector('.picker-button').click();
    const use = [...picker.querySelectorAll('.model')].find((m) => m.querySelector('.model-name').textContent === ${JSON.stringify(detection)})?.querySelector('.model-use');
    if (!use) throw new Error('the detection picker does not list ${detection}');
    use.click();
    await until(() => picker.querySelector('.picker-name').textContent === ${JSON.stringify(detection)} && picker.querySelector('.picker-progress').hidden, 'the detection model');
    return !picker.querySelector('.picker-button').classList.contains('empty');
  `);
  if (!loaded) throw new Error('the detection picker shows no model after loading one');
  console.log(`the detection picker loaded ${detection}`);

  // With it, the longer audio by its regions, as chunking_strategy "auto" gives it.
  const regions = await transcribe();
  const wantRegions = await api(long, { chunking_strategy: 'auto' });
  if (regions.text !== wantRegions || !regions.requests.length || regions.requests.some((r) => r.chunking_strategy !== 'auto')) {
    throw new Error(`the page by regions: ${JSON.stringify(regions)}, where the API gives ${JSON.stringify(wantRegions)} with chunking_strategy "auto"`);
  }
  console.log(`transcribe with ${detection}: ${(joined.length / rate).toFixed(1)} s by its regions, the text of chunking_strategy "auto": ${wantRegions.slice(0, 40)}`);
  // The live panel: the microphone transcribed as it plays, grey text while an utterance is said, then the whole text.
  const live = await page(`
    $('tab-live').click();
    await until(() => !$('live-start').disabled, 'the live panel to start');
    $('live-silence').value = '400';
    $('live-silence').dispatchEvent(new Event('change'));
    $('live-start').click();
    const card = document.querySelector('#live-transcript .transcript');
    let grey = 0;
    const end = Date.now() + ${Math.ceil((spoken.length / rate) * 1000) + 2500};
    while (Date.now() < end) {
      if (!$('live-error').hidden) return { error: $('live-error').textContent };
      if (card.querySelector('.provisional')?.textContent) grey++;
      await new Promise((r) => setTimeout(r, 100));
    }
    $('live-start').click();
    await until(() => !$('live-start').disabled && !$('live-status').textContent, 'the last words');
    if (!$('live-error').hidden) return { error: $('live-error').textContent };
    return {
      text: card.querySelector('.transcript-text').textContent, grey, provisional: card.querySelector('.provisional')?.textContent ?? '',
      // Stop's session.update changes nothing and carries no audio.
      turns: window.__sessions.filter((s) => s.audio).map((s) => s.audio.input.turn_detection),
    };
  `);
  const wantLive = await api(microphone, { 'chunking_strategy[type]': 'server_vad', 'chunking_strategy[silence_duration_ms]': '400' });
  if (live.error || !live.grey || live.provisional || cer(live.text, wantLive) > 0.1
      || JSON.stringify(live.turns) !== JSON.stringify([{ type: 'server_vad', silence_duration_ms: 400 }])) {
    throw new Error(`the live panel: ${JSON.stringify(live)}, where the API gives ${JSON.stringify(wantLive)} for the same audio`);
  }
  console.log(`live: ${(spoken.length / rate).toFixed(1)} s from the microphone, grey text in ${live.grey} of its looks, CER ${(100 * cer(live.text, wantLive)).toFixed(1)}% against the API: ${live.text.slice(0, 40)}`);
  if (errors.length) throw new Error(`the page threw: ${errors.join('; ')}`);
  console.log('ok');
} finally {
  ws?.close();
  for (const child of running.reverse()) child.stop();
  await wait(500);
  fs.rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}
