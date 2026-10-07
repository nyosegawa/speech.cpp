// Languages as people read them: the names of BCP 47 tags in the browser's own language, a long list as a count whose
// names open on request, and the language a text is written in where its script tells it.

const names = new Intl.DisplayNames([...navigator.languages, 'en'], { type: 'language' });

/** The name of the language of a tag in the browser's language, or the tag where the browser has none. */
export function languageName(tag) {
  try {
    return names.of(tag) ?? tag;
  } catch {
    return tag;
  }
}

/** The names of tags, in the order of the browser's language. */
export function languageNames(tags) {
  return tags.map(languageName).sort((a, b) => a.localeCompare(b));
}

/** How many names a list shows whole before it shows a count. */
const SHOWN = 3;

/**
 * An element for a list of languages: the names themselves when they are few, or "30 languages" that opens to them.
 * `prefix` goes before either.
 */
export function languageList(tags, prefix = '') {
  const listed = languageNames(tags);
  if (listed.length <= SHOWN) {
    const p = document.createElement('p');
    p.textContent = prefix + listed.join(', ');
    return p;
  }
  const details = document.createElement('details');
  const summary = document.createElement('summary');
  summary.textContent = `${prefix}${listed.length} languages`;
  const p = document.createElement('p');
  p.textContent = listed.join(', ');
  details.append(summary, p);
  return details;
}

/** A short phrase for the languages a model is the one to start with: "Japanese", "29 languages". */
export function startPhrase(tags) {
  return tags.length <= SHOWN ? languageNames(tags).join(', ') : `${tags.length} languages`;
}

/**
 * The tag of the language a text is written in where its script tells it: kana is Japanese, hangul Korean, and Han
 * characters without kana Chinese; a script several languages share, such as Latin, tells none.
 */
export function scriptLanguage(text) {
  if (/[\p{Script=Hiragana}\p{Script=Katakana}]/u.test(text)) return 'ja';
  if (/\p{Script=Hangul}/u.test(text)) return 'ko';
  if (/\p{Script=Han}/u.test(text)) return 'zh';
  return null;
}
