// Takes the README's screenshot of the page of speech serve: starts the server with a synthesis and a recognition
// model on a free port, opens the page in a headless Chrome in English, speaks the text with the voice, hands the speech
// to the Transcribe tab, transcribes it, and saves the whole page at twice its CSS pixels. Chrome is the one at $CHROME, or macOS's. The server
// and Chrome are stopped however the script ends.
// usage: node skills/release/scripts/page-screenshot.mjs <speech> <out.png> <synthesis model> <recognition model> <text> <voice>
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';

const [speech, out, synthesis, recognition, text, voice] = process.argv.slice(2);
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
  const timeout = new Promise((_, reject) => setTimeout(() => reject(new Error(`timed out waiting for ${what}`)), ms));
  return Promise.race([promise, timeout, ...processes.map((p) => p.failed)]);
}

/** A port that nothing listens on now. */
async function freePort() {
  const server = net.createServer();
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address();
  await new Promise((resolve) => server.close(resolve));
  return port;
}

const WIDTH = 960;
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'shot-chrome-'));
const running = [];
let ws = null;
try {
  const server = start(speech, ['serve', synthesis, recognition, '--port', String(await freePort())], { stdio: ['ignore', 'pipe', 'pipe'] });
  running.push(server);
  let log = '';
  const address = await racing(new Promise((resolve) => {
    const read = (chunk) => {
      log += chunk;
      const m = /(http:\/\/127\.0\.0\.1:\d+\/#token=[0-9a-f]+)/.exec(log);
      if (m) resolve(m[1]);
    };
    server.child.stdout.on('data', read);
    server.child.stderr.on('data', read);
  }), [server], 120000, 'speech serve to listen').catch((e) => {
    throw new Error(`${e.message}\n${log}`);
  });
  console.log('server', address.replace(/token=.*/, 'token=…'));

  const chrome = start(CHROME, [
    '--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--no-first-run',
    '--hide-scrollbars', '--lang=en-US', '--accept-lang=en-US', `--window-size=${WIDTH},900`,
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
  const evaluate = async (expression) => {
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails));
    return r.result?.result?.value;
  };

  await send('Emulation.setDeviceMetricsOverride', { width: WIDTH, height: 900, deviceScaleFactor: 2, mobile: false });
  await send('Page.navigate', { url: address.replace('127.0.0.1', 'localhost') });
  // The page builds each model's fields once it has asked the server for the models, after the load event.
  console.log(await evaluate(`(async () => {
    const $ = (id) => document.getElementById(id);
    const until = async (check, what, ms = 60000) => {
      const t = Date.now();
      while (!check()) {
        if (Date.now() - t > ms) throw new Error('timed out waiting for ' + what);
        await new Promise((r) => setTimeout(r, 100));
      }
    };
    await until(() => document.querySelector('#speak-options select[name="voice"]') && $('transcribe-options').children.length, 'the models');
    $('speak-text').value = ${JSON.stringify(text)};
    $('speak-text').dispatchEvent(new Event('input'));
    const voice = document.querySelector('#speak-options select[name="voice"]');
    voice.value = ${JSON.stringify(voice)};
    voice.dispatchEvent(new Event('change'));
    $('speak-start').click();
    await until(() => !$('speech-result').hidden && !$('speak-status').textContent, 'the speech');
    $('speech-transcribe').click();
    await until(() => !$('transcribe-start').disabled, 'the transcription form');
    $('transcribe-start').click();
    const card = document.querySelector('#transcribe-transcript .transcript');
    await until(() => !card.hidden && !$('transcribe-status').textContent, 'the text');
    window.scrollTo(0, 0);
    return card.querySelector('.transcript-text').textContent + ' | ' + card.querySelector('.meta').textContent;
  })()`));
  await wait(500);
  const height = await evaluate('Math.ceil(document.documentElement.scrollHeight)');
  await send('Emulation.setDeviceMetricsOverride', { width: WIDTH, height, deviceScaleFactor: 2, mobile: false });
  await wait(300);
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
  console.log('saved', out);
} finally {
  ws?.close();
  for (const child of running.reverse()) child.stop();
  await wait(500);
  fs.rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}
