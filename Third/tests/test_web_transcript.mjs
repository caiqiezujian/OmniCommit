// Run with node Third/tests/test_web_transcript.mjs.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const app = readFileSync(new URL('../web/meeting-v2/app.js', import.meta.url), 'utf8');
const fields = new Map();
function element() {
  return {children: [], files: [], append(row) { this.children.push(row); }, addEventListener() {}};
}
function field(id) {
  if (!fields.has(id)) fields.set(id, element());
  return fields.get(id);
}
let downloaded;
const ctx = vm.createContext({
  document: {getElementById: field, querySelectorAll: () => [], createElement: element},
  MeetingTurnLog: class {}, fetch: () => new Promise(() => {}),
  save: text => { downloaded = text; }
});
vm.runInContext(app, ctx);
vm.runInContext('download = (name, type, text) => save(text)', ctx);
const words = field('words');
for (const [action, text] of [
  ['DIRECT', 'Please'], ['WAIT', ''], ['DIRECT', ' continue'], ['DIRECT', '。然后呢？\nYes.'],
]) {
  ctx.result = {type:'result', action, display_text:text, end_seconds:5, generation_seconds:.1};
  vm.runInContext('showResult(result)', ctx);
}
assert.deepEqual(words.children.map(el => el.textContent), ['Please', ' continue', '。然后呢？\nYes.']);
field('saveText').onclick();
assert.equal(downloaded, words.children.map(el => el.textContent).join('\n'));
console.log('Third transcript row and export checks passed');
