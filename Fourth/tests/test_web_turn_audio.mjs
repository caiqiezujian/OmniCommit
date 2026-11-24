import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const script = new URL('../web/meeting-v2/turn-audio.js', import.meta.url);
const MeetingTurnAudio = vm.runInNewContext(readFileSync(script, 'utf8') + '\nMeetingTurnAudio',
  {ArrayBuffer, Blob});

function row(turn, start, end, action = 'DIRECT') {
  return {type: 'result', event: 'audio', action, turn, start_seconds: start / 16000, end_seconds: end / 16000};
}

async function payload(wav) {
  assert.equal(wav.type, 'audio/wav');
  const bytes = Buffer.from(await wav.arrayBuffer());
  assert.equal(bytes.toString('ascii', 0, 4), 'RIFF');
  assert.equal(bytes.readUInt32LE(4), bytes.length - 8);
  assert.equal(bytes.toString('ascii', 8, 16), 'WAVEfmt ');
  assert.equal(bytes.readUInt32LE(16), 16);
  assert.equal(bytes.readUInt16LE(20), 1);
  assert.equal(bytes.readUInt16LE(22), 1);
  assert.equal(bytes.readUInt32LE(24), 16000);
  assert.equal(bytes.readUInt32LE(28), 32000);
  assert.equal(bytes.readUInt16LE(32), 2);
  assert.equal(bytes.readUInt16LE(34), 16);
  assert.equal(bytes.toString('ascii', 36, 40), 'data');
  assert.equal(bytes.readUInt32LE(40), bytes.length - 44);
  return Array.from(bytes.subarray(44));
}

test('WAIT, partial DIRECT and previous+current WAV use the exact immutable wire samples', async () => {
  const audio = new MeetingTurnAudio();
  audio.start('test');
  const packet = Uint8Array.from([0xfa, 0xfb, 0x01, 0x80, 0xff, 0x7f, 0xfa, 0xfb]);
  const first = row(1, 0, 2, 'WAIT');
  first.source_start_seconds = 3600; // Original-file time is independent of the 16 kHz stream clock.
  first.source_end_seconds = 3600.00012;
  assert.equal(audio.put('test', first, packet.subarray(2, 6)), true);
  packet.fill(0);
  assert.equal(audio.put('test', row(2, 2, 3), Uint8Array.from([42, 0]).buffer), true);
  assert.deepEqual(await payload(await audio.wav('test', 1)), [1, 128, 255, 127]);
  assert.deepEqual(await payload(await audio.wav('test', 2)), [42, 0]);
  assert.equal(audio.hasPair('test', 1), false);
  assert.deepEqual(await payload(await audio.wav('test', 2, {previous: true})), [1, 128, 255, 127, 42, 0]);
});

test('gaps, missing previous turns and zero-length EOS do not create fake or stretched audio', async () => {
  const audio = new MeetingTurnAudio();
  audio.start('test');
  const pcm = new ArrayBuffer(2);
  assert.equal(audio.put('test', row(1, 0, 1), pcm), true);
  assert.equal(audio.put('test', {...row(2, 1, 1), event: 'end_of_audio'}, new ArrayBuffer(0)), false);
  assert.equal(audio.put('test', row(2, 1, 2), new ArrayBuffer(16000)), false); // Model padding
  assert.equal(audio.put('test', row(2, 2, 3, null), pcm), true);
  assert.equal(audio.hasPair('test', 2), false);
  await assert.rejects(audio.wav('test', 2, {previous: true}));
  assert.equal(audio.put('test', row(4, 3, 4), pcm), true);
  await assert.rejects(audio.wav('test', 4, {previous: true}));
});

test('identical turn numbers from different meetings and late results never share clips', async () => {
  const audio = new MeetingTurnAudio();
  const first = row(1, 0, 1);
  audio.start('old');
  assert.equal(audio.put('old', first, new ArrayBuffer(2)), true);
  audio.start('new');
  assert.equal(audio.has('old', 1), false);
  assert.equal(audio.put('old', first, new ArrayBuffer(2)), false);
  assert.equal(audio.put('new', first, Uint8Array.from([3, 4])), true);
  assert.equal(audio.put('new', first, new ArrayBuffer(2)), false);
  await assert.rejects(audio.wav('old', 1));
  assert.deepEqual(await payload(await audio.wav('new', 1)), [3, 4]);
});
