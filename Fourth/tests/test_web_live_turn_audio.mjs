import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const web = new URL('../web/meeting-v2/', import.meta.url);
const script = name => readFileSync(new URL(name, web), 'utf8');

// Drive the actual start button, capture queue and WebSocket request/response handlers.
// Only the browser devices, network peer and journal UI are simulated.
for (const source of ['microphone', 'system']) {
  test(source + ' turns replay the same wire PCM delivered with WAIT and DIRECT, excluding EOS', async () => {
    function element() {
      return Object.assign(new EventTarget(), {
        classList: {toggle() {}}, pause() {}, replaceChildren() {}, append() {}, setAttribute() {},
      });
    }
    const fields = new Map(Array.from(script('index.html').matchAll(/\bid="([^"]+)"/g),
      match => [match[1], element()]));
    const get = id => fields.get(id);
    const track = {stop() {}};
    const stream = {getAudioTracks: () => [track], getTracks: () => [track]};
    let opened, worklet;
    const pcm = [new Int16Array(16000).fill(-1234).buffer, new Int16Array(2853).fill(2345).buffer];
    const sent = [], journal = [];

    class Socket {
      constructor() { queueMicrotask(() => this.onopen()); }
      send(packet) {
        let reply;
        if (packet instanceof ArrayBuffer) {
          sent.push(Buffer.from(packet));
          const second = sent.length === 2;
          reply = {type: 'result', event: 'audio', turn: sent.length, action: second ? 'DIRECT' : 'WAIT',
            start_seconds: second ? 1 : 0, end_seconds: second ? 1 + 2853 / 16000 : 1,
            generation_seconds: 0, display_text: second ? '可以。' : ''};
        } else {
          const message = JSON.parse(packet);
          if (message.type === 'start') {
            assert.equal(message.source_kind, source);
            assert.notEqual(message.include_turn_audio, true);
            reply = {type: 'ready', protocol: 'fourth', chunk_seconds: 1};
          } else {
            assert.equal(message.type, 'end');
            reply = {type: 'result', event: 'end_of_audio', action: 'WAIT', turn: 3,
              start_seconds: 1 + 2853 / 16000, end_seconds: 1 + 2853 / 16000, generation_seconds: 0};
          }
        }
        queueMicrotask(() => this.onmessage({data: JSON.stringify(reply)}));
      }
      close() {}
    }
    class CaptureContext {
      sampleRate = 16000;
      audioWorklet = {addModule() {}};
      suspend() {}
      createMediaStreamSource() { return {connect() {}}; }
      resume() {
        for (const buffer of pcm) worklet.port.onmessage({data: buffer});
        worklet.port.onmessage({data: 'stopped'});
      }
      close() {}
    }
    class Worklet {
      constructor() { this.port = {}; worklet = this; }
      connect() {}
    }
    const session = 'live-' + source;
    class Journal {
      constructor(_get, _download, review) { this.review = review; }
      start() { this.currentId = session; }
      append(row) {
        if (row.type === 'result') journal.push([row.action, this.review.available(session, row.turn)]);
      }
    }
    const context = vm.createContext({
      ArrayBuffer, Blob, WebSocket: Socket, AudioContext: CaptureContext, AudioWorkletNode: Worklet,
      MeetingTurnLog: Journal, window: {isSecureContext: true}, location: {protocol: 'https:', host: 'localhost'},
      document: {getElementById: get, querySelectorAll: () => [], querySelector: () => ({value: source}),
        createElement: element, createTextNode: text => text},
      navigator: {mediaDevices: {getUserMedia() { opened = 'microphone'; return stream; },
        getDisplayMedia() { opened = 'system'; return stream; }}},
      fetch: async () => ({json: async () => ({ready: true, protocol: 'fourth', chunk_seconds: 1})}),
    });
    vm.runInContext(script('turn-audio.js'), context);
    vm.runInContext(script('app.js'), context);
    await new Promise(setImmediate);
    await get('start').onclick();

    assert.equal(get('status').textContent, '转录结束');
    assert.equal(opened, source);
    assert.deepEqual(journal, [['WAIT', true], ['DIRECT', true], ['WAIT', false]]);
    assert.deepEqual(sent, pcm.map(buffer => Buffer.from(buffer)));
    const audio = vm.runInContext('turnAudio', context);
    for (const [index, bytes] of sent.entries()) {
      const wav = await audio.wav(session, index + 1);
      assert.deepEqual(Buffer.from(await wav.arrayBuffer()).subarray(44), bytes);
    }
    const pair = await audio.wav(session, 2, {previous: true});
    assert.deepEqual(Buffer.from(await pair.arrayBuffer()).subarray(44), Buffer.concat(sent));
  });
}
