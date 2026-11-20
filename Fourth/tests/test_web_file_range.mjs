import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const web = new URL('../web/meeting-v2/', import.meta.url);
const html = readFileSync(new URL('index.html', web), 'utf8');
const script = readFileSync(new URL('app.js', web), 'utf8');

async function transcribe({start = '', end = '', follow = false, duration = 4,
                           source = 'file', mediaDevices, secure = true} = {}) {
  const fields = new Map();
  for (const [, id] of html.matchAll(/\bid="([^"]+)"/g)) {
    fields.set(id, Object.assign(new EventTarget(), {
      value: '', files: [], classList: {toggle() {}}, replaceChildren() {}, focus() {},
    }));
  }
  const field = id => fields.get(id);
  field('file').files = [{name: 'source.wav'}];
  field('fileStart').value = start;
  field('fileEnd').value = end;
  field('playback').checked = follow;
  const player = field('player');
  const playbackStarts = [];
  let position = 0;
  Object.assign(player, {
    readyState: 4, duration, pause() {},
    async play() {
      playbackStarts.push(position);
      position = duration;
      player.dispatchEvent(new Event('timeupdate'));
    },
  });
  Object.defineProperty(player, 'currentTime', {
    get: () => position,
    set(value) {
      position = value;
      queueMicrotask(() => player.dispatchEvent(new Event('seeked')));
    },
  });
  const sent = [];
  let uploads = 0;
  let heard, limit;
  class Socket {
    constructor() { queueMicrotask(() => this.onopen()); }
    send(packet) {
      const msg = JSON.parse(packet);
      sent.push(msg);
      let row;
      if (msg.type === 'start') {
        heard = msg.file_start_seconds ?? 0;
        limit = msg.file_end_seconds ?? duration;
        row = heard < 0 || limit > duration || heard >= limit
          ? {type: 'error', error: 'Invalid file bounds'}
          : {type: 'ready', protocol: 'fourth', chunk_seconds: 1, file_start_seconds: heard, file_end_seconds: limit};
      } else if (msg.type === 'file_next' && sent.length < 10) {
        heard = Math.min(heard + 1, msg.release_seconds, limit);
        row = {type: 'result', action: 'WAIT', end_seconds: heard, source_end_seconds: heard,
          generation_seconds: 0, file_complete: heard === limit};
      } else if (msg.type === 'end') {
        row = {type: 'file_end_empty'};
      } else {
        throw Error('Unexpected request ' + packet);
      }
      queueMicrotask(() => this.onmessage({data: JSON.stringify(row)}));
    }
    close() {}
  }
  const context = vm.createContext({
    document: {getElementById: field, querySelectorAll: () => [], querySelector: () => ({value: source})},
    window: {isSecureContext: secure}, navigator: {mediaDevices},
    location: {protocol: 'http:', host: 'localhost'},
    MeetingTurnLog: class { start() {} append() {} },
    WebSocket: Socket, setTimeout, clearTimeout,
    fetch: async path => {
      if (path === '/health') return {ok: true, json: async () => ({ready: true, protocol: 'fourth', chunk_seconds: 1, file_input: true})};
      uploads++;
      return {ok: true, json: async () => ({media_id: 'upload', duration_seconds: duration, sample_rate: 16000, samples: duration * 16000})};
    },
  });
  vm.runInContext(script, context);
  await new Promise(setImmediate);
  await field('start').onclick();
  return {sent, playbackStarts, uploads, status: field('status').textContent};
}

test('file range and whole-file default use the source clock in continuous mode', async () => {
  for (const [fields, releases, bounds] of [
    [{}, [1, 2, 3, 4], {}],
    [{start: '1.375'}, [2.375, 3.375, 4], {file_start_seconds: 1.375}],
    [{end: '00:02.25'}, [1, 2, 2.25], {file_end_seconds: 2.25}],
    [{start: '0:00:01.375', end: '0:03.5'}, [2.375, 3.375, 3.5], {file_start_seconds: 1.375, file_end_seconds: 3.5}],
  ]) {
    const {sent, status} = await transcribe(fields);
    assert.equal(sent.at(-1)?.type, 'end', status);
    const {file_start_seconds, file_end_seconds} = sent[0];
    assert.equal(file_start_seconds, bounds.file_start_seconds);
    assert.equal(file_end_seconds, bounds.file_end_seconds);
    assert.deepEqual(sent.filter(row => row.type === 'file_next').map(row => row.release_seconds), releases);
  }
});

test('following the player seeks to the start and caps releases at the selected end', async () => {
  const {sent, playbackStarts, status} = await transcribe({start: '1.25', end: '2.75', follow: true});
  assert.deepEqual(playbackStarts, [1.25], status);
  assert.equal(sent.at(-1)?.type, 'end', status);
  const releases = sent.filter(row => row.type === 'file_next').map(row => row.release_seconds);
  assert.ok(releases.every(value => value >= 1.25 && value <= 2.75));
  assert.equal(releases.at(-1), 2.75);
});

test('invalid or reversed ranges never begin decoding', async () => {
  for (const bounds of [{start: '2', end: '1'}, {start: 'n/a'}, {end: '8'}]) {
    const {sent, status, uploads} = await transcribe(bounds);
    if (bounds.start !== undefined) assert.equal(uploads, 0, status);
    assert.ok(sent.every(row => row.type === 'start'), status);
    assert.ok(status && status !== '转录结束');
  }
});

test('system capture requests shared audio and releases a video-only selection', async () => {
  let stopped = false, requested;
  const video = {stop() { stopped = true; }};
  const mediaDevices = {async getDisplayMedia(options) {
    requested = {...options};
    return {getAudioTracks: () => [], getTracks: () => [video]};
  }};
  const run = await transcribe({source: 'system', mediaDevices});
  assert.deepEqual(requested, {video: true, audio: true, systemAudio: 'include'});
  assert.match(run.status, /共享音频/);
  assert.equal(stopped, true);
  assert.equal(run.sent.length, 0);
});

test('system capture explains an insecure origin and a missing browser API', async () => {
  assert.match((await transcribe({source: 'system', secure: false})).status, /HTTPS/);
  assert.match((await transcribe({source: 'system', mediaDevices: {}})).status, /桌面浏览器/);
});
