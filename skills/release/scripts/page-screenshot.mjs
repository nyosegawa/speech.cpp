// Takes the README's screenshots of the page of speech serve, one for each tab: starts the server with a synthesis, a
// recognition and a detection model on a free port, opens the page in a headless Chrome in English, speaks the text
// with the voice (Speak), hands the speech to Transcribe and transcribes it (Transcribe), then transcribes a microphone
// that says the text a sentence at a time with pauses (Live), taken while the first sentence is final and the next one
// is still grey. Each is saved at twice its CSS pixels, down to the tab's panel, as OUT-speak.png, OUT-transcribe.png
// and OUT-live.png. The server listens on 8080, the port the page names, unless $PORT says another. Chrome is the one at
// $CHROME, or macOS's. The server and Chrome are stopped however the script ends.
// A sampling model speaks differently each time; give the seed of a take whose transcript reads right.
// usage: node skills/release/scripts/page-screenshot.mjs <speech> <OUT> <synthesis model> <recognition model>
//        <detection model> <text> <voice> [seed]
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const [speech, out, synthesis, recognition, detection, text, voice, seed] = process.argv.slice(2);
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

/**
 * `promise`, or the first failure of `processes`, or a timeout. The timer is cleared once it settles: a timer left
 * running keeps Node from exiting until it fires, minutes after the screenshots are saved.
 */
function racing(promise, processes, ms, what) {
  let timer;
  const timeout = new Promise((_, reject) => (timer = setTimeout(() => reject(new Error(`timed out waiting for ${what}`)), ms)));
  return Promise.race([promise, timeout, ...processes.map((p) => p.failed)]).finally(() => clearTimeout(timer));
}


/** The rate and the 16-bit samples of a WAVE file that speech tts wrote. */
function readWave(file) {
  const bytes = fs.readFileSync(file);
  const rate = bytes.readUInt32LE(24);
  for (let at = 12; at + 8 <= bytes.length; at += 8 + bytes.readUInt32LE(at + 4)) {
    if (bytes.toString('ascii', at, at + 4) === 'data') return { rate, pcm: bytes.subarray(at + 8, at + 8 + bytes.readUInt32LE(at + 4)) };
  }
  throw new Error(`${file} has no data chunk`);
}

/** A WAVE file of the sentences of `text` spoken one at a time, with a pause before, between and after them. */
function microphone(folder) {
  const parts = [];
  let rate = 0;
  const silence = (seconds) => Buffer.alloc(2 * Math.round(seconds * rate));
  const sentences = text.split(/(?<=[.!?。！？])\s*/).filter((s) => s.trim());
  sentences.forEach((sentence, i) => {
    const file = path.join(folder, `sentence-${i}.wav`);
    const r = spawnSync(speech, ['tts', synthesis, '--voice', voice, ...(seed ? ['--seed', seed] : []), '-o', file, sentence],
                        { stdio: ['ignore', 'ignore', 'pipe'] });
    if (r.status !== 0) throw new Error(`speech tts failed: ${r.stderr}`);
    const wave = readWave(file);
    rate = wave.rate;
    parts.push(silence(i === 0 ? 0.6 : 1.5), wave.pcm);
  });
  parts.push(silence(2.5));
  const pcm = Buffer.concat(parts);
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write('WAVEfmt ', 8);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(rate, 24);
  header.writeUInt32LE(rate * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write('data', 36);
  header.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([header, pcm]);
}

const WIDTH = 960;
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'shot-chrome-'));
const running = [];
let ws = null;
try {
  const spoken = microphone(profile);
  const server = start(speech, ['serve', synthesis, recognition, detection, '--port', process.env.PORT ?? '8080'], { stdio: ['ignore', 'pipe', 'pipe'] });
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
  console.log('server', address);

  const chrome = start(CHROME, [
    '--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--no-first-run',
    '--hide-scrollbars', '--lang=en-US', '--accept-lang=en-US', `--window-size=${WIDTH},900`,
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
  ws.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
    }
  });
  const send = (method, params = {}) => racing(new Promise((resolve) => {
    const n = ++id;
    pending.set(n, resolve);
    ws.send(JSON.stringify({ id: n, method, params }));
  }), running, 180000, method);
  /** Runs `body`, an async function's body, in the page with $ and until(check, what) defined. */
  const page = async (body) => {
    const r = await send('Runtime.evaluate', {
      expression: `(async () => {
        const $ = (id) => document.getElementById(id);
        const until = async (check, what, ms = 120000) => {
          const t = Date.now();
          while (!check()) {
            if (Date.now() - t > ms) throw new Error('timed out waiting for ' + what);
            await new Promise((r) => setTimeout(r, 100));
          }
        };
        ${body}
      })()`,
      awaitPromise: true,
      returnByValue: true,
    });
    if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails));
    return r.result?.result?.value;
  };
  /** Saves the page from its top to a margin below the tab's panel, at twice its CSS pixels. */
  const shoot = async (file) => {
    await send('Emulation.setDeviceMetricsOverride', { width: WIDTH, height: 900, deviceScaleFactor: 2, mobile: false });
    await page('window.scrollTo(0, 0);');
    const height = await page(`return Math.ceil(document.querySelector('section.panel:not([hidden])').getBoundingClientRect().bottom + 48);`);
    await send('Emulation.setDeviceMetricsOverride', { width: WIDTH, height, deviceScaleFactor: 2, mobile: false });
    await wait(300);
    const shot = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync(file, Buffer.from(shot.result.data, 'base64'));
    console.log('saved', file);
  };

  await send('Page.enable');
  // The microphone says the sentences, once, in real time.
  await send('Page.addScriptToEvaluateOnNewDocument', { source: `
    navigator.mediaDevices.getUserMedia = async () => {
      const bytes = Uint8Array.from(atob(${JSON.stringify(spoken.toString('base64'))}), (c) => c.charCodeAt(0));
      const context = new AudioContext();
      const source = context.createBufferSource();
      source.buffer = await context.decodeAudioData(bytes.buffer);
      const out = context.createMediaStreamDestination();
      source.connect(out);
      source.start();
      return out.stream;
    };
  ` });
  await send('Emulation.setDeviceMetricsOverride', { width: WIDTH, height: 900, deviceScaleFactor: 2, mobile: false });
  const loaded = new Promise((resolve) => ws.addEventListener('message', (event) => {
    if (JSON.parse(event.data).method === 'Page.loadEventFired') resolve();
  }));
  await send('Page.navigate', { url: address.replace('127.0.0.1', 'localhost') });
  // Page.navigate answers once the new document is committed, which can be before the page's modules have given the
  // tabs their clicks.
  await racing(loaded, running, 60000, 'the page to load');

  await page(`
    await until(() => document.querySelector('#speak-options select[name="voice"]'), 'the models');
    $('speak-text').value = ${JSON.stringify(text)};
    $('speak-text').dispatchEvent(new Event('input'));
    const voice = document.querySelector('#speak-options select[name="voice"]');
    voice.value = ${JSON.stringify(voice)};
    voice.dispatchEvent(new Event('change'));
    const seed = document.querySelector('#speak-options input[name="seed"]');
    if (${JSON.stringify(seed ?? '')} && seed) {
      seed.value = ${JSON.stringify(seed ?? '')};
      seed.dispatchEvent(new Event('input'));
    }
    $('speak-start').click();
    await until(() => !$('speech-result').hidden && !$('speak-status').textContent, 'the speech');
  `);
  await wait(800);
  await shoot(`${out}-speak.png`);

  console.log('transcribe:', await page(`
    $('speech-transcribe').click();
    await until(() => !$('transcribe-start').disabled, 'the transcription form');
    $('transcribe-start').click();
    const card = document.querySelector('#transcribe-transcript .transcript');
    await until(() => !card.hidden && !$('transcribe-status').textContent, 'the text');
    return card.querySelector('.transcript-text').textContent;
  `));
  await wait(300);
  await shoot(`${out}-transcribe.png`);

  // The live panel while a sentence is final and the next one is grey. The grey text grows by the words two readings
  // agree on and is replaced once the sentence ends, so the page is taken again each time more of it is grey, and the
  // last of those takes is kept.
  await page(`
    $('tab-live').click();
    await until(() => !$('live-start').disabled, 'the live panel to start');
    $('live-start').click();
  `);
  const said = () => page(`
    const text = document.querySelector('#live-transcript .transcript .transcript-text');
    const grey = text?.querySelector('.provisional')?.textContent ?? '';
    return { final: text ? text.textContent.slice(0, text.textContent.length - grey.length).trim() : '', grey: grey.trim() };
  `);
  let best = '';
  for (const t = Date.now(); Date.now() - t < 60000;) {
    const now = await said();
    if (best && !now.grey) break;
    if (now.final && now.grey.length > best.length) {
      best = now.grey;
      await shoot(`${out}-live.png`);
      console.log('live:', JSON.stringify(now));
    }
    await wait(100);
  }
  if (!best) throw new Error('the live panel never showed a final sentence with grey text after it');
  await page(`
    if (!$('live-start').disabled) $('live-start').click();
    await until(() => !$('live-start').disabled && !$('live-status').textContent, 'the last words');
  `);
} finally {
  ws?.close();
  for (const child of running.reverse()) child.stop();
  await wait(500);
  fs.rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}
