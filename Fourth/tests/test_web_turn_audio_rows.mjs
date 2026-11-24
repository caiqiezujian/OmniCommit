import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const bundle = new URL('../web/meeting-v2/', import.meta.url);
const script = readFileSync(new URL('turn-log.js', bundle), 'utf8');

class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this.classList = {add: cls => { this.className += ' ' + cls; }};
  }
  append(...children) { this.children.push(...children); }
  setAttribute(key, value) { this.attributes[key] = value; }
  get textContent() { return this.children.map(el => el.textContent).join(''); }
  set textContent(value) { this.children = [{textContent: value}]; }
  get childElementCount() { return this.children.filter(el => el.tag).length; }
}
const context = vm.createContext({
  document: {createElement: tag => new Element(tag), createTextNode: text => ({textContent: text})},
});
vm.runInContext(script, context);

test('a WAIT can be reviewed and downloaded, with no previous turn at the start', () => {
  const calls = [];
  const review = {
    available: () => true, availablePair: () => false,
    play: (...args) => calls.push(args), download: (...args) => calls.push(args),
  };
  const row = {type: 'result', event: 'audio', turn: 1, action: 'WAIT', start_seconds: 0, end_seconds: 1};
  const [details, audio] = context.logEntry(row, review, 'live').children;
  assert.match(details.textContent, /本轮新增音频 · 会话 0.000–1.000 s/);
  assert.match(details.textContent, /本轮提交文字 · WAIT/);
  const [play, pair, download] = audio.children;
  assert.equal(pair.disabled, true);
  play.onclick();
  download.onclick();
  assert.deepEqual(calls, [['live', 1, false], ['live', 1]]);
});

test('the pair button uses the true previous audio turn and displays attributed text', () => {
  const calls = [];
  const review = {
    available: () => true, availablePair: (id, turn) => id === 'file' && turn === 5,
    play: (...args) => calls.push(args),
  };
  const row = {type: 'result', event: 'audio', turn: 5, action: 'DIRECT',
    source_start_seconds: 42.25, source_end_seconds: 43.25,
    text: '<|DIRECT|>预算。<|SPK_CHANGE|>可以。',
    transcript_parts: [{text: '预算。', speaker_change: false}, {text: '可以。', speaker_change: true}]};
  const [details, audio] = context.logEntry(row, review, 'file').children;
  assert.match(details.textContent, /原文件 42.250–43.250 s/);
  assert.match(details.textContent, /本轮提交文字 · DIRECT预算。换人可以。/);
  assert.equal(audio.children[1].disabled, false);
  audio.children[1].onclick();
  assert.deepEqual(calls, [['file', 5, true]]);
  details.open = true;
  details.ontoggle();
  assert.match(details.children[1].textContent, /<\|SPK_CHANGE\|>/);
});

test('old text stays readable when audio is unavailable and an end event has no clip', () => {
  const review = {available: () => false};
  const row = {type: 'result', event: 'audio', turn: 10, action: 'DIRECT', display_text: 'Hello.'};
  const [details, audio] = context.logEntry(row, review, 'archived').children;
  assert.ok(details.textContent.includes('Hello.'));
  assert.equal(audio.textContent, '此轮音频未保留');
  assert.equal(context.logEntry({...row, event: 'end_of_audio'}, review, 'archived').childElementCount, 1);
  assert.equal(context.logEntry({type: 'file_wait'}, review, 'archived').childElementCount, 1);
});
