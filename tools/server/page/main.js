// The page of speech serve: takes the token from its address, asks the server for the catalog and the models it holds,
// and shows a panel for each task with its model picker.

import * as api from './api.js';
import { ModelPicker } from './picker.js';
import { SpeakPanel } from './speak.js';
import { TranscribePanel } from './transcribe.js';

const $ = (id) => document.getElementById(id);
const TASKS = ['synthesis', 'recognition'];

/** How often the page asks again while a model of a task is being replaced by a load that another page started. */
const WATCH_MS = 500;

const transcribe = new TranscribePanel();
const panels = { synthesis: new SpeakPanel((wav, label) => transcribe.use(wav, label)), recognition: transcribe };
const pickers = {};
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
  panels[task].show(held);
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

for (const task of TASKS) {
  pickers[task] = new ModelPicker(document.querySelector(`.picker-slot[data-task="${task}"]`), task, (held) => showHeld(task, held), refresh);
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
