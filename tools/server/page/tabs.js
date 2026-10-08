// The tabs at the top of the page, one for each panel, of which one shows at a time. The arrow keys move between them,
// as on any tab list, and the page opens on the tab last chosen in this browser.

const KEY = 'speech.cpp tab';

export class Tabs {
  #tabs;
  #selected;

  /** Takes the tabs of `list`, each with aria-controls naming its panel; `selected(panel)` follows each choice. */
  constructor(list, selected) {
    this.#tabs = [...list.querySelectorAll('[role="tab"]')];
    this.#selected = selected;
    for (const tab of this.#tabs) tab.addEventListener('click', () => this.select(tab.getAttribute('aria-controls')));
    list.addEventListener('keydown', (event) => {
      const at = this.#tabs.indexOf(document.activeElement);
      const to = { ArrowLeft: at - 1, ArrowRight: at + 1, Home: 0, End: this.#tabs.length - 1 }[event.key];
      if (at < 0 || to === undefined) return;
      event.preventDefault();
      const tab = this.#tabs[(to + this.#tabs.length) % this.#tabs.length];
      this.select(tab.getAttribute('aria-controls'));
      tab.focus();
    });
    let kept = null;
    try {
      kept = localStorage.getItem(KEY);
    } catch {
      // Storage the browser refuses opens the page on its first tab.
    }
    this.select(this.#tabs.some((tab) => tab.getAttribute('aria-controls') === kept) ? kept : this.#tabs[0].getAttribute('aria-controls'));
  }

  /** Shows the panel whose id is `name` and hides the others. */
  select(name) {
    for (const tab of this.#tabs) {
      const chosen = tab.getAttribute('aria-controls') === name;
      tab.setAttribute('aria-selected', String(chosen));
      tab.tabIndex = chosen ? 0 : -1;
      document.getElementById(tab.getAttribute('aria-controls')).hidden = !chosen;
    }
    try {
      localStorage.setItem(KEY, name);
    } catch {
      // Without storage the page opens on its first tab next time.
    }
    this.#selected(document.getElementById(name));
  }

  /** Marks the tab of the panel `name` as busy, so that work going on there shows from the other tabs. */
  busy(name, busy) {
    this.#tabs.find((tab) => tab.getAttribute('aria-controls') === name).classList.toggle('busy', busy);
  }
}
