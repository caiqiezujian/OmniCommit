// Run with node Fourth/tests/test_web_transcript.mjs.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const app = readFileSync(new URL('../web/meeting-v2/app.js', import.meta.url), 'utf8');
const turnAudio = readFileSync(new URL('../web/meeting-v2/turn-audio.js', import.meta.url), 'utf8');
const log = readFileSync(new URL('../web/meeting-v2/turn-log.js', import.meta.url), 'utf8');
const fields = new Map();
function element() {
  return {
    children: [], files: [],
    get textContent() { return this.value ?? this.children.map(el => el.textContent).join(""); },
    set textContent(value) { this.value = value; },
    append(row) { this.children.push(row); },
    setAttribute() {},
    addEventListener() {},
  };
}
function field(id) {
  if (!fields.has(id)) fields.set(id, element());
  return fields.get(id);
}
let downloaded;
const ctx = vm.createContext({
  document: {getElementById: field, querySelectorAll: () => [], createElement: element, createTextNode: text => ({textContent: text})},
  MeetingTurnLog: class {}, fetch: () => new Promise(() => {}),
  save: text => { downloaded = text; }
});
vm.runInContext(`(() => {${log}; Object.assign(globalThis, {gapMessage, indexLogRow, logHeading, logHistory});})();`, ctx);
vm.runInContext(turnAudio, ctx);
vm.runInContext(app, ctx);
vm.runInContext('download = (name, type, text) => save(text)', ctx);
const words = field('words');
for (const [action, text] of [
  ['DIRECT', 'Please'], ['WAIT', ''], ['DIRECT', ' continue'], ['DIRECT', '。然后呢？\nYes.'],
]) {
  ctx.result = {type:'result', action, display_text:text, end_seconds:5, generation_seconds:.1};
  vm.runInContext('showResult(result)', ctx);
}
assert.deepEqual(words.children.map(el => el.textContent), ['Please continue。然后呢？\nYes.']);
field('saveText').onclick();
assert.equal(downloaded, words.children.map(el => el.textContent).join('\n'));
const gap = {type:'result', turn:8, event:'audio', action:null, text:'', display_text:'',
  start_seconds:1.5, end_seconds:2, generation_seconds:.1,
  truncation:{status:'gap', start_seconds:1, end_seconds:2, source_start_seconds:13, source_end_seconds:14}};
const resumed = {...gap, action:'DIRECT', text:'Next.', display_text:'Next.', truncation:{status:'recovered'}};
for (const result of [gap, resumed]) {
  ctx.result = result;
  vm.runInContext('showResult(result)', ctx);
}
const notice = field('gapNotice');
assert.equal(notice.hidden, false);
assert.match(notice.textContent, /原文件 13.000–14.000 秒/);
assert.equal(words.children.at(-1).textContent, 'Next.');
assert.equal(words.children.length, 2);
field('saveText').onclick();
assert.match(downloaded, /\[原文件 13.000–14.000 秒，转录待核对\]\nNext\.$/);
const history = ctx.logHistory();
for (const result of [gap, resumed]) ctx.indexLogRow(history, result);
assert.equal(history.model.length, 2);
assert.equal(history.gaps.length, 1);
assert.equal(history.DIRECT.length, 1);
assert.equal(history.WAIT.length, 0);
assert.match(ctx.logHeading(gap), /原文件 13.000–14.000 秒/);
assert.match(ctx.logHeading(resumed), /重试已完成/);
assert.match(ctx.gapMessage({event:'end_of_audio', truncation:{start_seconds:8, end_seconds:8}}), /结束轮/);
const switches = [{text:' Okay', speaker_change:false}, {text:'好。', speaker_change:true},
                  {text:' Weiter.', speaker_change:true}];
ctx.result = {...resumed, transcript_parts:switches, text:'<|DIRECT|> Okay<|SPK_CHANGE|>好。<|SPK_CHANGE|> Weiter.'};
vm.runInContext('showResult(result)', ctx);
assert.deepEqual(words.children.slice(-3).map(el => el.textContent), ['Next. Okay', '好。', ' Weiter.']);
assert.equal(words.children.at(-1).className, 'segment speaker-change');
field('saveText').onclick();
assert.ok(downloaded.endsWith('Next. Okay\n好。\n Weiter.'));
assert.ok(!downloaded.includes('<|'));
console.log('Fourth transcript row and export checks passed');
