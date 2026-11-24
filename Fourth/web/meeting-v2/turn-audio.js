'use strict';

// The Web transport uses mono 16 kHz signed little-endian PCM16.
const TURN_AUDIO_RATE = 16000;

function turnWav(pcmBlobs) {
  const size = pcmBlobs.reduce((total, pcm) => total + pcm.size, 0);
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  function fourCC(offset, text) {
    for (let i = 0; i < 4; i++) view.setUint8(offset + i, text.charCodeAt(i));
  }
  fourCC(0, 'RIFF');
  view.setUint32(4, 36 + size, true);
  fourCC(8, 'WAVE');
  fourCC(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // Linear PCM
  view.setUint16(22, 1, true);
  view.setUint32(24, TURN_AUDIO_RATE, true);
  view.setUint32(28, TURN_AUDIO_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  fourCC(36, 'data');
  view.setUint32(40, size, true);
  return new Blob([header, ...pcmBlobs], {type: 'audio/wav'});
}

class MeetingTurnAudio {
  constructor() {
    this.sessionId = null;
    this.turns = new Map();
  }

  start(sessionId) {
    if (sessionId !== this.sessionId) {
      this.turns.clear();
      this.sessionId = sessionId;
    }
  }

  put(sessionId, row, pcm) {
    if (!sessionId || sessionId !== this.sessionId || row?.event !== 'audio'
        || !Number.isSafeInteger(row.turn) || row.turn < 1 || this.turns.has(row.turn)
        || !(pcm instanceof ArrayBuffer || ArrayBuffer.isView(pcm))) return false;
    const start = Math.round(row.start_seconds * TURN_AUDIO_RATE);
    const end = Math.round(row.end_seconds * TURN_AUDIO_RATE);
    if (!Number.isFinite(row.start_seconds) || !Number.isFinite(row.end_seconds)
        || !Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end <= start
        || pcm.byteLength !== (end - start) * 2) return false;
    // Blob snapshots exactly this buffer/view, including the short last packet.
    // Store only the input samples; wrap in WAV on demand without joining a meeting in memory.
    this.turns.set(row.turn, {start, end, pcm: new Blob([pcm])});
    return true;
  }

  has(sessionId, turn) {
    return Boolean(sessionId && sessionId === this.sessionId && this.turns.has(turn));
  }

  hasPair(sessionId, turn) {
    return this.has(sessionId, turn) && this.turns.get(turn - 1)?.end === this.turns.get(turn).start;
  }

  async wav(sessionId, turn, {previous = false} = {}) {
    if (!this.has(sessionId, turn)) throw new Error('本页没有保存这一轮的音频。');
    if (previous && !this.hasPair(sessionId, turn)) throw new Error('缺少与本轮连续的上一轮音频。');
    const current = this.turns.get(turn).pcm;
    return turnWav(previous ? [this.turns.get(turn - 1).pcm, current] : [current]);
  }
}
