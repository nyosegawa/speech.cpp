// Takes the README's screenshot of the page of speech serve: starts the server with a synthesis and a recognition
// model, opens the page in a headless Chrome in English, speaks the text with the voice, hands the speech to Transcribe,
// transcribes it, and saves the whole page at twice its CSS pixels. Chrome is the one at $CHROME, or macOS's.
// usage: node skills/release/scripts/page-screenshot.mjs <speech> <out.png> <synthesis model> <recognition model> <text> <voice>
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const [speech, out, synthesis, recognition, text, voice] = process.argv.slice(2);
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const server = spawn(speech, ['serve', synthesis, recognition, '--port', '8080'], { stdio: ['ignore', 'pipe', 'pipe'] });
const address = await new Promise((resolve, reject) => {
  let log = '';
  const read = (chunk) => {
    log += chunk;
    const m = /(http:\/\/127\.0\.0\.1:\d+\/#token=[0-9a-f]+)/.exec(log);
    if (m) resolve(m[1]);
  };
  server.stdout.on('data', read);
  server.stderr.on('data', read);
  server.on('exit', (code) => reject(new Error(`speech serve exited with ${code}:\n${log}`)));
});
console.log('server', address.replace(/token=.*/, 'token=…'));

const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'shot-chrome-'));
const chrome = spawn(process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--no-first-run',
  // The page names languages in navigator.languages, which --accept-lang sets; --lang alone leaves macOS's.
  '--hide-scrollbars', '--lang=en-US', '--accept-lang=en-US', '--window-size=1280,900', '--autoplay-policy=no-user-gesture-required',
], { stdio: 'ignore' });
let port;
for (let i = 0; i < 100 && !port; i++) {
  await wait(100);
  try {
    port = fs.readFileSync(path.join(profile, 'DevToolsActivePort'), 'utf8').split('\n')[0];
  } catch {
    // Chrome writes the file once it listens.
  }
}
if (!port) throw new Error(`Chrome did not start: no DevToolsActivePort in ${profile}`);
const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const page = targets.find((t) => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let id = 0;
const pending = new Map();
ws.addEventListener('message', (event) => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    pending.get(message.id)(message);
    pending.delete(message.id);
  }
});
const send = (method, params = {}) => new Promise((resolve) => {
  const n = ++id;
  pending.set(n, resolve);
  ws.send(JSON.stringify({ id: n, method, params }));
});
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails));
  return r.result?.result?.value;
};

try {
  await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 900, deviceScaleFactor: 2, mobile: false });
  await send('Page.navigate', { url: address.replace('127.0.0.1', 'localhost') });
  await wait(2500);
  console.log(await evaluate(`(async () => {
    const $ = (id) => document.getElementById(id);
    const until = async (check, ms = 60000) => { const t = Date.now(); while (!check()) { if (Date.now() - t > ms) throw new Error('timed out'); await new Promise((r) => setTimeout(r, 100)); } };
    $('speak-text').value = ${JSON.stringify(text)};
    $('speak-text').dispatchEvent(new Event('input'));
    const voice = document.querySelector('#speak-options select[name="voice"]');
    voice.value = ${JSON.stringify(voice)};
    voice.dispatchEvent(new Event('change'));
    $('speak-start').click();
    await until(() => !$('speech-result').hidden && !$('speak-status').textContent);
    $('speech-transcribe').click();
    await until(() => !$('transcribe-start').disabled);
    $('transcribe-start').click();
    await until(() => !$('transcript').hidden && !$('transcribe-status').textContent);
    window.scrollTo(0, 0);
    return $('transcript-text').textContent + ' | ' + $('transcript-meta').textContent;
  })()`));
  await wait(500);
  const height = await evaluate('Math.ceil(document.documentElement.scrollHeight)');
  await send('Emulation.setDeviceMetricsOverride', { width: 1280, height, deviceScaleFactor: 2, mobile: false });
  await wait(300);
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
  console.log('saved', out, height);
} finally {
  ws.close();
  chrome.kill();
  server.kill();
  await wait(500);
  fs.rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}
