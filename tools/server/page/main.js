// The page of speech serve: takes the token from its address, asks the server for the catalog and the models it holds,
// and shows the lists of models and the panels that speak and transcribe with the models held.

import * as api from './api.js';
import { ModelLists } from './models.js';
import { SpeakPanel } from './speak.js';
import { TranscribePanel } from './transcribe.js';

const $ = (id) => document.getElementById(id);

function fail(message) {
  $('page-error').textContent = message;
  $('page-error').hidden = false;
}

const panels = { synthesis: new SpeakPanel(), recognition: new TranscribePanel() };
const lists = new ModelLists({ synthesis: $('synthesis-models'), recognition: $('recognition-models') }, (task, held) =>
  panels[task].show(held),
);

if (!api.takeToken()) {
  fail('Open this page from the address that speech serve printed when it started, which ends in #token=…; the page needs that token to fetch and load models.');
} else {
  try {
    const state = await api.models();
    $('about').textContent = `${state.catalog.version}, on ${location.host}`;
    $('model-dir').textContent = state.catalog.directory;
    $('model-dir').className = 'path';
    lists.show(state);
    panels.synthesis.show(state.synthesis.held);
    panels.recognition.show(state.recognition.held);
  } catch (e) {
    fail(e.message);
  }
}
