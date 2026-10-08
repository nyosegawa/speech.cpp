// The page of speech serve: takes the token from its address, asks the server for the catalog and the models it holds,
// and shows a tab for each panel. A task has one model picker, which moves to the panel in view, so that the
// transcribe and live panels show the same recognition model and the same progress of a load. The transcribe panel
// also shows the detection model, which finds the regions it transcribes.

import * as api from './api.js';
import { LivePanel } from './live.js';
import { ModelPicker } from './picker.js';
import { SpeakPanel } from './speak.js';
import { Tabs } from './tabs.js';
import { TranscribePanel } from './transcribe.js';

const $ = (id) => document.getElementById(id);
const TASKS = ['synthesis', 'recognition', 'detection'];

/** How often the page asks again while a model of a task is being replaced by a load that another page started. */
const WATCH_MS = 500;

const pickers = Object.fromEntries(TASKS.map((task) => [task, new ModelPicker(task, (held) => showHeld(task, held), refresh)]));
const tabs = new Tabs($('tabs'), (panel) => {
  for (const slot of panel.querySelectorAll('.picker-slot')) slot.append(pickers[slot.dataset.task].element);
});
const transcribe = new TranscribePanel();
const speak = new SpeakPanel((wav, label) => {
  transcribe.use(wav, label);
  tabs.select('transcribe');
});
const panels = {
  synthesis: [speak],
  recognition: [transcribe, new LivePanel((busy) => tabs.busy('live', busy))],
  detection: [{ show: (held) => transcribe.showDetection(held) }],
};
/** What each panel shows, so that a refresh that changes nothing leaves its fields alone. */
const shown = {};
let watching = 0;

function fail(message) {
  $('page-error').textContent = message;
  $('page-error').hidden = !message;
}

function showHeld(task, held) {
  const key = JSON.stringify(held);
  if (shown[task] === key) return;
  shown[task] = key;
  for (const panel of panels[task]) panel.show(held);
}

/** Asks the server what is fetched and held, and keeps asking while another page replaces a model. */
async function refresh() {
  clearTimeout(watching);
  let state;
  try {
    state = await api.models();
  } catch (e) {
    fail(e.message);
    return;
  }
  fail('');
  $('about').textContent = `${state.catalog.version}, on ${location.host}`;
  for (const task of TASKS) {
    pickers[task].show(state);
    showHeld(task, state[task].held);
  }
  if (TASKS.some((task) => state[task].replacing !== null && !pickers[task].busy)) watching = setTimeout(refresh, WATCH_MS);
}

function start() {
  if (api.takeToken()) {
    refresh();
  } else {
    fail('Open this page from the address that speech serve printed when it started, which ends in #token=…; the page needs that token to fetch and load models.');
  }
}

// An address with a new token, typed into this tab, changes only its fragment, which does not load the page again.
addEventListener('hashchange', start);
start();
