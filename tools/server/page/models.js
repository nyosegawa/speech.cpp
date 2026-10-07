// The lists of models, one for each task: each model of the catalog with its size, whether it is fetched, and the
// languages for which it is the one to start with, and a button that fetches it with its progress, which can be
// cancelled, and has the server load it in place of the task's model.

import * as api from './api.js';
import { languageName } from './options.js';

const TASKS = ['synthesis', 'recognition'];

function gigabytes(bytes) {
  return `${(bytes / 1e9).toFixed(2)} GB`;
}

function megabytes(bytes) {
  return `${Math.round(bytes / 1e6)} MB`;
}

/** How often the lists ask again while a model of a task is being replaced by a load that this page did not start. */
const WATCH_MS = 500;

export class ModelLists {
  #lists;
  #state = null;
  #busy = new Set();
  #watching = 0;
  /** The last load that failed, {name, message}, shown at its model until another load starts. */
  #failure = null;

  /** `loaded(task, held)` is called whenever the model of a task changes. */
  constructor(lists, loaded) {
    this.#lists = lists;
    this.loaded = loaded;
  }

  /**
   * Shows the state that GET /speech/models gave, and while a task's model is being replaced by a load that another
   * page started, or that this one stopped and the server has yet to give up, asks again until it has ended.
   */
  show(state) {
    this.#state = state;
    clearTimeout(this.#watching);
    if (TASKS.some((task) => state[task].replacing !== null && !this.#busy.has(task))) {
      this.#watching = setTimeout(() => this.refresh(), WATCH_MS);
    }
    for (const task of TASKS) {
      const list = this.#lists[task];
      list.replaceChildren();
      const held = state[task].held;
      // A model given on the command line has no entry in the catalog, so it is listed by its file.
      if (held && held.name === null) list.append(this.#item({ file: held.path }, task));
      for (const model of state.catalog.models) {
        if (model.task !== task) continue;
        for (const file of model.files) list.append(this.#item({ model, file }, task));
      }
    }
  }

  #item(entry, task) {
    const item = document.querySelector('#model-item').content.firstElementChild.cloneNode(true);
    const $ = (selector) => item.querySelector(selector);
    const held = this.#state[task].held;
    const replacing = this.#state[task].replacing;
    const use = $('.model-use');

    if (!entry.model) {
      $('.model-name').textContent = entry.file.split(/[\\/]/).pop();
      $('.model-meta').textContent = 'Given on the command line';
      $('.model-languages').textContent = held.model.languages.join(' ');
      item.dataset.state = 'held';
      $('.badge').hidden = false;
      use.remove();
      return item;
    }

    const { model, file } = entry;
    const name = file.type === model.type ? model.name : `${model.name}:${file.type}`;
    $('.model-name').textContent = name;
    const fetched = file.fetched ? 'fetched' : file.partial ? `${megabytes(file.partial)} of it fetched` : 'not fetched yet';
    $('.model-meta').textContent = `${gigabytes(file.size)}, ${fetched}`;
    const languagesLine = $('.model-languages');
    if (model.start.length) {
      const strong = document.createElement('strong');
      strong.textContent = 'The one to start with for ';
      languagesLine.append(strong, model.start.join(' '));
    } else {
      languagesLine.textContent = model.languages.length === 1 ? languageName(model.languages[0]) : model.languages.join(' ');
    }
    languagesLine.title = model.languages.map(languageName).join('\n');

    const inUse = held?.name === name;
    item.dataset.state = inUse ? 'held' : replacing === name ? 'loading' : file.fetched ? 'fetched' : 'remote';
    $('.badge').hidden = !inUse;
    if (inUse) use.remove();
    use.textContent = file.fetched ? 'Use' : file.partial ? 'Resume and use' : 'Fetch and use';
    use.setAttribute('aria-label', `${use.textContent} ${name}`);
    use.title = file.fetched ? `Load ${name}` : `Fetch ${gigabytes(file.size)} from Hugging Face, then load ${name}`;
    use.disabled = this.#busy.has(task) || replacing !== null;
    if (replacing === name) this.#loading(item);
    if (this.#failure?.name === name) {
      $('.error').textContent = this.#failure.message;
      $('.error').hidden = false;
    }
    use.addEventListener('click', () => this.#use(name, task, item));
    return item;
  }

  #loading(item) {
    item.querySelector('.model-use')?.remove();
    item.querySelector('.model-progress-row').hidden = false;
    item.querySelector('.model-progress').removeAttribute('value');
    item.querySelector('.model-progress-text').textContent = 'Loading';
    item.querySelector('.model-cancel').hidden = true;
  }

  async #use(name, task, item) {
    const $ = (selector) => item.querySelector(selector);
    const progress = $('.model-progress');
    const text = $('.model-progress-text');
    const cancel = $('.model-cancel');
    const controller = new AbortController();
    this.#failure = null;
    this.#busy.add(task);
    for (const button of this.#lists[task].querySelectorAll('.model-use')) button.disabled = true;
    $('.model-use').remove();
    $('.error').hidden = true;
    $('.model-progress-row').hidden = false;
    progress.removeAttribute('value');
    text.textContent = 'Starting';
    cancel.onclick = () => controller.abort();
    cancel.focus();
    try {
      for await (const event of api.load(name, controller.signal)) {
        if (event.type === 'fetch') {
          progress.value = event.done / event.total;
          text.textContent = `${megabytes(event.done)} of ${megabytes(event.total)}`;
        } else if (event.type === 'note') {
          text.textContent = event.message;
        } else if (event.type === 'load') {
          this.#loading(item);
        } else if (event.type === 'loaded') {
          this.loaded(task, event.held);
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') this.#failure = { name, message: e.message };
    } finally {
      this.#busy.delete(task);
    }
    await this.refresh();
  }

  /** Asks the server again what is fetched and held. */
  async refresh() {
    this.show(await api.models());
  }
}
