// The fields of a model's request options, built from its information as speech info --json gives it: each option's
// type, whether it is required, its default, its range and its choices. The common options are shown and the rest are
// folded away; an option the model checks but does not steer by has no field, and a field left at its default sends
// nothing, so that the model's own default applies.

const COMMON = new Set(['voice', 'language', 'speed', 'seed', 'instructions', 'prompt']);
const LONG = new Set(['instructions', 'prompt']);
const languages = new Intl.DisplayNames(['en'], { type: 'language' });
let fieldCount = 0;

function label(name) {
  return name[0].toUpperCase() + name.slice(1).replaceAll('_', ' ');
}

/** The English name of the language of a BCP 47 tag, or the tag where the browser has none. */
export function languageName(tag) {
  try {
    return languages.of(tag) ?? tag;
  } catch {
    return tag;
  }
}

function range(option) {
  const parts = [];
  if (option.minimum !== undefined) parts.push(`from ${option.minimum}`);
  if (option.exclusive_minimum !== undefined) parts.push(`above ${option.exclusive_minimum}`);
  if (option.maximum !== undefined && option.maximum < 1e15) parts.push(`to ${option.maximum}`);
  if (option.exclusive_maximum !== undefined) parts.push(`below ${option.exclusive_maximum}`);
  return parts.join(' ');
}

export class OptionForm {
  #fields = new Map();

  /**
   * Builds the fields of `info`'s options into `visible`, and the uncommon ones into `more`, the folded part, which is
   * hidden when it holds none; `only`, when given, names the options to build, those the endpoint's form takes, which
   * all go into `visible`.
   */
  constructor(info, visible, more = null, only = null) {
    visible.replaceChildren();
    more?.replaceChildren();
    for (const option of info.options) {
      if (!option.steers || (only && !only.has(option.name))) continue;
      const field = this.#build(option, info);
      this.#fields.set(option.name, field);
      (more && !COMMON.has(option.name) ? more : visible).append(field.element);
    }
    if (more) more.closest('details').hidden = more.children.length === 0;
  }

  #build(option, info) {
    const id = `option-${option.name}-${++fieldCount}`;
    const element = document.createElement('div');
    element.className = 'field';
    const caption = document.createElement('label');
    caption.htmlFor = id;
    caption.textContent = label(option.name);
    const hint = document.createElement('p');
    hint.className = 'hint';
    hint.id = `${id}-hint`;
    const error = document.createElement('p');
    error.className = 'error';
    error.id = `${id}-error`;
    error.hidden = true;

    let input;
    if (option.type === 'bool') {
      input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = option.default ?? false;
      element.classList.add('check');
    } else if (option.choices) {
      input = this.#select(option, info, hint);
    } else if (option.type === 'int' || option.type === 'float') {
      input = document.createElement('input');
      input.type = 'number';
      input.step = option.type === 'int' ? '1' : 'any';
      if (option.minimum !== undefined) input.min = option.minimum;
      if (option.maximum !== undefined && option.maximum < 1e15) input.max = option.maximum;
      input.placeholder = option.default !== undefined ? String(option.default) : '';
      hint.textContent = option.name === 'seed' ? 'Left empty, each request draws one.' : range(option);
    } else {
      input = document.createElement(LONG.has(option.name) ? 'textarea' : 'input');
      if (LONG.has(option.name)) {
        input.rows = 2;
        element.classList.add('wide');
      }
      input.placeholder = option.default ?? '';
    }
    input.id = id;
    input.name = option.name;
    input.setAttribute('aria-describedby', `${hint.id} ${error.id}`);
    if (option.type === 'bool') element.append(input, caption, error);
    else element.append(caption, input, hint, error);
    return { option, input, error, element };
  }

  #select(option, info, hint) {
    const select = document.createElement('select');
    const voices = new Map((info.voices ?? []).map((v) => [v.name, v]));
    if (!option.required) {
      const fallback = new Option(option.default !== undefined ? `Default (${option.default})` : 'Default', '');
      select.append(fallback);
    } else if (option.choices.length === 0) {
      select.append(new Option(option.name === 'voice' ? 'No voice yet; make one below' : 'None', ''));
      select.disabled = true;
    }
    for (const choice of option.choices) {
      const voice = voices.get(choice);
      let text = option.name === 'language' ? `${languageName(choice)} (${choice})` : choice;
      if (voice?.language) text += ` (${[languageName(voice.language), voice.gender].filter(Boolean).join(', ')})`;
      select.append(new Option(text, choice));
    }
    if (option.name === 'voice') {
      const describe = () => (hint.textContent = voices.get(select.value)?.description ?? '');
      select.addEventListener('change', describe);
      describe();
    }
    return select;
  }

  /** The values set, by option name and of the option's type, without the fields left at their defaults. */
  values() {
    const out = {};
    for (const [name, { option, input }] of this.#fields) {
      if (option.type === 'bool') {
        if (input.checked !== (option.default ?? false)) out[name] = input.checked;
      } else if (input.value.trim() !== '') {
        out[name] = option.type === 'int' || option.type === 'float' ? Number(input.value) : input.value;
      }
    }
    return out;
  }

  /** Sets the fields to `values`, as values() gave them, where the option and the choice are still there. */
  restore(values) {
    for (const [name, value] of Object.entries(values)) {
      const field = this.#fields.get(name);
      if (!field) continue;
      if (field.option.type === 'bool') field.input.checked = value;
      else if (!field.option.choices || field.option.choices.includes(value)) field.input.value = value;
      field.input.dispatchEvent(new Event('change'));
    }
  }

  select(name, value) {
    this.restore({ [name]: value });
  }

  /** Shows a failure at the field of the option it names, and returns whether there is one. */
  showError(name, message) {
    const field = this.#fields.get(name);
    if (!field) return false;
    field.error.textContent = message;
    field.error.hidden = false;
    field.input.setAttribute('aria-invalid', 'true');
    field.element.closest('details')?.setAttribute('open', '');
    field.input.focus();
    return true;
  }

  clearErrors() {
    for (const { input, error } of this.#fields.values()) {
      error.hidden = true;
      input.removeAttribute('aria-invalid');
    }
  }
}
