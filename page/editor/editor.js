// The page editor (loaded only in the preview that page/edit_server.py serves). Text marked [data-copy] is outlined;
// clicking it opens its copy-file entry in a side panel, where typing previews in place and Save writes the entry,
// rebuilds and reloads at the same scroll position. "Edit" in the corner turns the outlines and clicks off, so links
// and the story's controls work as on the live page. The preview reloads by itself when the copy file changes.
(() => {
'use strict';
const KEY_SCROLL = 'copy-editor-scroll', KEY_ON = 'copy-editor-on';
let state = null, current = null, original = '', dirty = false, version = null;
const on = () => sessionStorage.getItem(KEY_ON) !== '0';

const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const panel = el('aside', 'ce-panel'); panel.hidden = true;
const head = el('div', 'ce-head'); const keyName = el('code', 'ce-key'); const close = el('button', 'ce-x', '×');
close.type = 'button'; close.title = 'Close (Esc)'; head.append(keyName, close);
const area = el('textarea', 'ce-text'); area.spellcheck = true;
const msg = el('div', 'ce-msg');
const row = el('div', 'ce-row'); const save = el('button', 'ce-save', 'Save'); const revert = el('button', 'ce-revert', 'Undo changes');
save.type = revert.type = 'button'; row.append(save, revert);
const hint = el('div', 'ce-hint', 'Ctrl+Enter saves · Esc closes. Text is HTML: keep tags balanced and placeholders such as __ATLAS_LINK__ as they are.');
const otherH = el('div', 'ce-other-h', 'Other text (not shown on the page)'); const otherList = el('div', 'ce-other');
panel.append(head, area, msg, row, hint, otherH, otherList);
const bar = el('div', 'ce-bar'); const toggle = el('button', 'ce-toggle'); const otherBtn = el('button', 'ce-otherbtn', 'Other text');
toggle.type = otherBtn.type = 'button'; const file = el('span', 'ce-file'); bar.append(toggle, otherBtn, file);
document.body.append(panel, bar);

const setOn = v => { sessionStorage.setItem(KEY_ON, v ? '1' : '0'); document.documentElement.classList.toggle('ce-on', v); toggle.textContent = v ? 'Edit: on' : 'Edit: off'; };
toggle.onclick = () => setOn(!on());
setOn(on());

const preview = t => t.replace(/\{\{cite:([A-Za-z0-9_-]+)\}\}/g, '<sup class="ce-cite">[$1]</sup>').replace(/__[A-Z_]+_LINK__/g, '#');
const say = (t, bad) => { msg.textContent = t || ''; msg.classList.toggle('bad', !!bad); };

function open(key) {
  if (dirty && current && current !== key && !confirm('Discard the unsaved change to ' + current + '?')) return;
  if (dirty) undo();
  current = key; original = state.entries[key]; dirty = false;
  keyName.textContent = key; area.value = original; say(''); panel.hidden = false;
  document.querySelectorAll('[data-copy].ce-sel').forEach(n => n.classList.remove('ce-sel'));
  document.querySelectorAll(`[data-copy="${CSS.escape(key)}"]`).forEach(n => n.classList.add('ce-sel'));
  area.focus();
}
function undo() {
  if (!current) return;
  area.value = original; dirty = false;
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
  const r = await fetch('/__save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key: current, text: area.value }) });
  const j = await r.json().catch(() => ({ error: 'no answer from the server' }));
  if (j.error) { say(j.error, true); return; }
  dirty = false; sessionStorage.setItem(KEY_SCROLL, String(window.scrollY)); location.reload();
}
save.onclick = doSave; revert.onclick = () => { undo(); say('Back to the saved text.'); }; close.onclick = shut;
document.addEventListener('keydown', e => {
  if (panel.hidden) return;
  if (e.key === 'Escape') { e.preventDefault(); shut(); }
  else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); doSave(); }
});
document.addEventListener('click', e => {
  if (!on() || panel.contains(e.target) || bar.contains(e.target)) return;
  const t = e.target.closest && e.target.closest('[data-copy]');
  if (!t) return;
  e.preventDefault(); e.stopPropagation(); open(t.dataset.copy);
}, true);
otherBtn.onclick = () => { panel.hidden = false; if (!current) { keyName.textContent = 'Other text'; area.value = ''; } otherList.scrollIntoView({ block: 'nearest' }); };

async function load() {
  const j = await (await fetch('/__copy')).json();
  state = j; version = j.version; file.textContent = j.file;
  otherList.replaceChildren(...j.other.map(k => {
    const b = el('button', 'ce-other-item'); b.type = 'button';
    b.append(el('code', null, k), el('span', null, j.entries[k]));
    b.onclick = () => open(k); return b;
  }));
  if (j.error) say('The last build failed: ' + j.error, true);
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
