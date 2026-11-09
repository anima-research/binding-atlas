// The page editor (loaded only in the preview that page/edit_server.py serves). Text marked [data-copy] is outlined;
// clicking it opens its copy-file entry in a side panel, where typing previews in place and Save writes the entry,
// rebuilds and reloads at the same scroll position. "Edit" in the corner turns the outlines and clicks off, so links
// and the story's controls work as on the live page. The preview reloads by itself when the copy file changes.
(() => {
'use strict';
const KEY_SCROLL = 'copy-editor-scroll', KEY_ON = 'copy-editor-on';
let state = null, current = null, original = '', dirty = false, version = null, pending = {}, pendingNotes = {};
const on = () => sessionStorage.getItem(KEY_ON) !== '0';

const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const panel = el('aside', 'ce-panel'); panel.hidden = true;
const head = el('div', 'ce-head'); const keyName = el('code', 'ce-key'); const close = el('button', 'ce-x', '×');
close.type = 'button'; close.title = 'Close (Esc)'; head.append(keyName, close);
const area = el('textarea', 'ce-text'); area.spellcheck = true;
const msg = el('div', 'ce-msg');
const row = el('div', 'ce-row'); const save = el('button', 'ce-save', 'Save'); const revert = el('button', 'ce-revert', 'Undo changes');
const citeBtn = el('button', 'ce-citebtn', 'Add citation'); const noteBtn = el('button', 'ce-notebtn', 'Add note');
save.type = revert.type = citeBtn.type = noteBtn.type = 'button'; row.append(save, revert, citeBtn, noteBtn);
// the citation form: a new reference (short name, authors, title, year, link) or one the page already has
const cform = el('div', 'ce-cform'); cform.hidden = true;
const cf = (label, ph) => { const w = el('label', 'ce-cf'); const i = el('input'); i.placeholder = ph; w.append(el('span', null, label), i); cform.append(w); return i; };
const cExisting = el('select', 'ce-cexist'); const cExistW = el('label', 'ce-cf'); cExistW.append(el('span', null, 'Use one the page has'), cExisting); cform.append(cExistW);
cform.append(el('div', 'ce-cor', 'or add a new one:'));
const cName = cf('Short name (the marker)', 'e.g. UncheatableEval'), cAuth = cf('Authors', 'e.g. Tan, Li, Shen'),
      cTitle = cf('Title', 'the paper\'s title'), cYear = cf('Year', 'e.g. 2026'), cLink = cf('Link', 'https://arxiv.org/abs/…');
const crow = el('div', 'ce-row'); const cAdd = el('button', 'ce-save', 'Insert at cursor'); const cCancel = el('button', null, 'Cancel');
cAdd.type = cCancel.type = 'button'; crow.append(cAdd, cCancel); cform.append(crow);
const cmsg = el('div', 'ce-msg'); cform.append(cmsg);
const hint = el('div', 'ce-hint', 'Ctrl+Enter saves · Esc closes. Text is HTML: keep tags balanced and placeholders such as __ATLAS_LINK__ as they are.');
const otherH = el('div', 'ce-other-h', 'Other text (not shown on the page)'); const otherList = el('div', 'ce-other');
const pick = el('div', 'ce-pick', 'Pick an item below to edit it.'); pick.hidden = true;
panel.append(head, pick, area, msg, row, cform, hint, otherH, otherList);
const bar = el('div', 'ce-bar'); const toggle = el('button', 'ce-toggle'); const otherBtn = el('button', 'ce-otherbtn', 'Other text');
toggle.type = otherBtn.type = 'button'; const file = el('span', 'ce-file'); bar.append(toggle, otherBtn, file);
document.body.append(panel, bar);

const setOn = v => { sessionStorage.setItem(KEY_ON, v ? '1' : '0'); document.documentElement.classList.toggle('ce-on', v); toggle.textContent = v ? 'Edit: on' : 'Edit: off'; };
toggle.onclick = () => setOn(!on());
setOn(on());

const preview = t => t.replace(/\{\{note:[A-Za-z0-9_-]+\|([^|}]+)\}\}/g, '<u style="text-decoration-style:dotted">$1</u>').replace(/\{\{cite:([A-Za-z0-9_-]+)\}\}/g, '<sup class="ce-cite">[$1]</sup>').replace(/__[A-Z_]+_LINK__/g, '#');
const say = (t, bad) => { msg.textContent = t || ''; msg.classList.toggle('bad', !!bad); };

function open(key) {
  if (dirty && current && current !== key && !confirm('Discard the unsaved change to ' + current + '?')) return;
  if (dirty) undo();
  current = key; original = state.entries[key]; dirty = false; pending = {}; pendingNotes = {}; showEditor(true); cform.hidden = true;
  keyName.textContent = key + (state.files && state.files[key] ? '  ·  ' + state.files[key] : ''); area.value = original; say(''); panel.hidden = false;
  document.querySelectorAll('[data-copy].ce-sel').forEach(n => n.classList.remove('ce-sel'));
  document.querySelectorAll(`[data-copy="${CSS.escape(key)}"]`).forEach(n => n.classList.add('ce-sel'));
  area.focus();
}
function showEditor(on) { [area, row].forEach(n => { n.hidden = !on; }); pick.hidden = on; }
function undo() {
  if (!current) return;
  area.value = original; dirty = false; pending = {}; pendingNotes = {};
  document.querySelectorAll(`[data-copy="${CSS.escape(current)}"]`).forEach(n => { n.innerHTML = preview(original); });
}
function shut() {
  if (dirty && !confirm('Discard the unsaved change?')) return;
  undo(); panel.hidden = true; current = null;
  document.querySelectorAll('[data-copy].ce-sel').forEach(n => n.classList.remove('ce-sel'));
}
area.addEventListener('input', () => {
  dirty = area.value !== original;
  document.querySelectorAll(`[data-copy="${CSS.escape(current)}"]`).forEach(n => { n.innerHTML = preview(area.value); });
});
async function doSave() {
  if (!current) return;
  if (!dirty) { say('No changes.'); return; }
  say('Saving…');
  const cites = {};
  for (const [n, t] of Object.entries(pending)) if (area.value.includes('{{cite:' + n + '}}')) cites[n] = t;
  const notes = {};
  for (const [n, t] of Object.entries(pendingNotes)) if (area.value.includes('{{note:' + n + '|')) notes[n] = t;
  const r = await fetch('/__save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key: current, text: area.value, cites, notes }) });
  const j = await r.json().catch(() => ({ error: 'no answer from the server' }));
  if (j.error) { say(j.error, true); return; }
  version = j.version;                     // our own save: the poll must not reload for it
  if (Object.keys(cites).length || Object.keys(notes).length) {   // a new citation or note: reload once
    dirty = false; sessionStorage.setItem(KEY_SCROLL, String(window.scrollY)); location.reload(); return;
  }
  // Swap in only the edited text, as the rebuilt page renders it (numbers, citations, links filled in); the rest of
  // the page, and any animation, keeps running.
  const key = current, text = area.value;
  try {
    const doc = new DOMParser().parseFromString(await (await fetch('/', { cache: 'no-store' })).text(), 'text/html');
    const fresh = [...doc.querySelectorAll(`[data-copy="${CSS.escape(key)}"]`)];
    const live = [...document.querySelectorAll(`[data-copy="${CSS.escape(key)}"]`)];
    if (fresh.length && fresh.length === live.length) live.forEach((n, i) => { n.innerHTML = fresh[i].innerHTML; });
  } catch (_) { /* the live preview already shows the text */ }
  state.entries[key] = text; original = text; dirty = false;
  const item = [...otherList.querySelectorAll('.ce-other-item')].find(b => b.firstChild.textContent === key);
  if (item) item.lastChild.textContent = text;
  say('Saved.');
}
save.onclick = doSave; revert.onclick = () => { undo(); say('Back to the saved text.'); }; close.onclick = shut;
document.addEventListener('keydown', e => {
  if (panel.hidden) return;
  if (e.key === 'Escape') { e.preventDefault(); shut(); }
  else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); doSave(); }
});
// The contents rail: a label is a heading's text, or a section's data-toc label, whose key the edit build names in
// data-copy-toc; clicking it opens that key.
function railKey(a) {
  const id = decodeURIComponent((a.getAttribute('href') || '').slice(1)), t = id && document.getElementById(id);
  if (!t) return null;
  if (t.dataset.copyToc) return t.dataset.copyToc;
  const inner = t.matches('[data-copy]') ? t : t.querySelector(':scope > [data-copy]');
  return inner ? inner.dataset.copy : null;
}
document.addEventListener('click', e => {
  if (!on() || panel.contains(e.target) || bar.contains(e.target)) return;
  const a = e.target.closest && e.target.closest('#toc a');
  if (a) {
    const k = railKey(a);
    e.preventDefault(); e.stopPropagation();
    if (k && state.entries[k] != null) open(k);
    else { panel.hidden = false; keyName.textContent = a.textContent; showEditor(false); say('This label is not in a copy file yet; ask for it to be added.', true); }
    return;
  }
  const t = e.target.closest && e.target.closest('[data-copy]');
  if (!t) return;
  e.preventDefault(); e.stopPropagation(); open(t.dataset.copy);
}, true);
otherBtn.onclick = () => { panel.hidden = false; if (!current) { keyName.textContent = 'Other text'; showEditor(false); } otherList.scrollIntoView({ block: 'nearest' }); };

const esc = t => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
function insertMarker(name) {
  const a = area.selectionStart ?? area.value.length, b = area.selectionEnd ?? a, m = '{{cite:' + name + '}}';
  area.value = area.value.slice(0, a) + m + area.value.slice(b);
  area.focus(); area.selectionStart = area.selectionEnd = a + m.length;
  area.dispatchEvent(new Event('input'));
}
citeBtn.onclick = () => {
  const names = Object.keys(state.entries).filter(k => k.startsWith('cite.')).map(k => k.slice(5));
  cExisting.replaceChildren(el('option', null, '—'), ...names.map(n => { const o = el('option', null, n); o.value = n; return o; }));
  [cName, cAuth, cTitle, cYear, cLink].forEach(i => { i.value = ''; }); cmsg.textContent = ''; cform.hidden = false; cName.focus();
};
cCancel.onclick = () => { cform.hidden = true; };
// Add note: select the phrase in the text box first; the note's text is asked for, and the phrase becomes
// {{note:NAME|phrase}}; the note's entry is written with the text on save.
noteBtn.onclick = () => {
  const a = area.selectionStart, b = area.selectionEnd, phrase = area.value.slice(a, b);
  if (!phrase.trim()) { say('Select the words the note should hang on first.', true); return; }
  if (/[|}]/.test(phrase)) { say('The phrase can\'t contain | or }.', true); return; }
  const name = (prompt('Short name for the note (letters, digits, - or _):', '') || '').trim();
  if (!name) return;
  if (!/^[A-Za-z0-9_-]+$/.test(name) || state.entries['note.' + name] || pendingNotes[name]) { say('That name is not allowed or already taken.', true); return; }
  const text = (prompt('The note\'s text:', '') || '').trim();
  if (!text) return;
  pendingNotes[name] = text;
  area.value = area.value.slice(0, a) + '{{note:' + name + '|' + phrase + '}}' + area.value.slice(b);
  area.dispatchEvent(new Event('input'));
  say('Note "' + name + '" added; it is written to the file when you save.');
};
cExisting.onchange = () => { if (cExisting.value && cExisting.value !== '—') { insertMarker(cExisting.value); cform.hidden = true; } };
cAdd.onclick = () => {
  const n = cName.value.trim(), au = cAuth.value.trim(), ti = cTitle.value.trim(), yr = cYear.value.trim(), ln = cLink.value.trim();
  const bad = t => { cmsg.textContent = t; cmsg.classList.add('bad'); };
  if (!/^[A-Za-z0-9_-]+$/.test(n)) return bad('The short name can use letters, digits, - and _ only (no spaces).');
  if (state.entries['cite.' + n] || pending[n]) return bad('That name is taken; pick it from the list above instead.');
  if (!ti) return bad('A title is needed.');
  if (ln && !/^https?:\/\//.test(ln)) return bad('The link should start with https://');
  const title = ln ? `<a href="${esc(ln)}"><i>${esc(ti)}</i></a>` : `<i>${esc(ti)}</i>`;
  pending[n] = [au && esc(au), title, yr && esc(yr)].filter(Boolean).join(', ') + '.';
  insertMarker(n); cform.hidden = true;
  say('Citation "' + n + '" added; it is written to the file when you save.');
};

async function load() {
  const j = await (await fetch('/__copy')).json();
  state = j; version = j.version; file.textContent = j.file;
  otherList.replaceChildren(...j.other.map(k => {
    const b = el('button', 'ce-other-item'); b.type = 'button';
    b.append(el('code', null, k), el('span', null, j.entries[k]));
    b.onclick = () => open(k); return b;
  }));
  if (j.error) say('The last build failed: ' + j.error, true);
  else if (j.warning) say(j.warning, true);
}
async function poll() {
  try {
    const j = await (await fetch('/__version')).json();
    if (version !== null && j.version !== version && !dirty) { sessionStorage.setItem(KEY_SCROLL, String(window.scrollY)); location.reload(); return; }
    if (j.error) say('The last build failed: ' + j.error, true);
  } catch (_) { say('The edit server is not answering.', true); }
  setTimeout(poll, 1500);
}
const y = sessionStorage.getItem(KEY_SCROLL);
if (y !== null) { sessionStorage.removeItem(KEY_SCROLL); window.addEventListener('load', () => window.scrollTo(0, +y)); }
load().then(() => setTimeout(poll, 1500));
})();
