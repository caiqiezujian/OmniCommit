'use strict';

// Keep the displayed range independent from the append-only event history.
const LOG_PAGE_SIZE = 50;

function logHistory(protocol = 'fourth') {
  return {protocol, all: [], model: [], WAIT: [], DIRECT: [], gaps: [], waitRun: 0, waitSeconds: 0, lastDirect: null, error: null};
}

function indexLogRow(history, row) {
  history.all.push(row);
  if (row.type === 'ready' && row.protocol) history.protocol = row.protocol;
  if (row.type === 'error') history.error = row;
  if (row.type !== 'result') return;
  history.model.push(row);
  if (row.truncation?.status === 'gap') {
    history.gaps.push(row);
    history.waitRun = 0;
    history.waitSeconds = 0;
  } else if (row.action === 'WAIT' || row.action === 'DIRECT') {
    history[row.action].push(row);
    if (row.action === 'DIRECT') {
      history.lastDirect = row;
      history.waitRun = 0;
      history.waitSeconds = 0;
    } else if (row.event === 'audio') {
      history.waitRun++;
      history.waitSeconds += row.end_seconds - row.start_seconds;
    }
  }
}

function gapMessage(row) {
  const gap = row.truncation;
  const file = Number.isFinite(gap.source_start_seconds) && Number.isFinite(gap.source_end_seconds);
  const start = file ? gap.source_start_seconds : gap.start_seconds;
  const end = file ? gap.source_end_seconds : gap.end_seconds;
  const where = file ? '原文件 ' : '';
  if (row.event === 'end_of_audio' && start === end) return where + end.toFixed(3) + ' 秒，结束轮解码未完成';
  return where + start.toFixed(3) + '–' + end.toFixed(3) + ' 秒，转录待核对';
}

function logHeading(row) {
  if (row.type !== 'result') return row.type;
  const fields = ['第 ' + row.turn + ' 轮'];
  const audioLabel = row.event === 'audio' ? '本轮新增音频 · ' : '';
  if (Number.isFinite(row.source_start_seconds) && Number.isFinite(row.source_end_seconds)) {
    fields.push(audioLabel + '原文件 ' + row.source_start_seconds.toFixed(3) + '–' + row.source_end_seconds.toFixed(3) + ' s');
  } else if (Number.isFinite(row.start_seconds) && Number.isFinite(row.end_seconds)) {
    fields.push(audioLabel + '会话 ' + row.start_seconds.toFixed(3) + '–' + row.end_seconds.toFixed(3) + ' s');
  }
  if (row.event === 'end_of_audio') fields.push('音频结束');
  const vad = row.silence_gate?.decision;
  if (vad === 'skip' || vad === 'end_quiet') fields.push('静音跳过解码');
  if (vad === 'tail') fields.push('停顿首轮，允许提交句尾');
  if (vad === 'uncertain') fields.push('人声不确定，已送模型');
  if (vad === 'bypass_error') fields.push('静音检测异常，本场继续用主模型');
  if (row.truncation?.status === 'gap') fields.push(gapMessage(row));
  if (row.truncation?.status === 'recovered') fields.push('生成达到长度上限，重试已完成');
  if (row.end_reason === 'gap_already_reported') fields.push('已关闭，缺口见上一条记录');
  if (row.evicted_this_turn) fields.push('本轮裁剪');
  if (Number.isFinite(row.generation_seconds)) fields.push('模型 ' + row.generation_seconds.toFixed(2) + ' s');
  return fields.join(' · ');
}

function turnAudioControls(review, sessionId, turn) {
  const group = document.createElement('div');
  group.className = 'turn-audio-controls';
  if (!review?.available(sessionId, turn)) {
    group.textContent = '此轮音频未保留';
    return group;
  }
  function addButton(label, action, disabled = false) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.setAttribute('aria-label', '第 ' + turn + ' 轮：' + label);
    button.disabled = disabled;
    button.onclick = action;
    group.append(button);
  }
  addButton('▶ 本轮', () => review.play(sessionId, turn, false));
  addButton('▶ 前一轮 + 本轮', () => review.play(sessionId, turn, true), !review.availablePair(sessionId, turn));
  addButton('下载本轮 WAV', () => review.download(sessionId, turn));
  return group;
}

function logEntry(row, review, sessionId) {
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.dataset.action = row.action || '';
  const details = document.createElement('details');
  const summary = document.createElement('summary');
  const heading = document.createElement('span');
  heading.className = 'log-meta';
  heading.textContent = logHeading(row);
  const output = document.createElement('span');
  output.className = 'log-output';
  if (row.type === 'result' && row.event === 'audio') {
    output.classList.add('log-commit');
    const label = document.createElement('span');
    label.className = 'log-output-label';
    label.textContent = '本轮提交文字 · ' + (row.action || '缺口');
    output.append(label);
    if (row.action === 'DIRECT') {
      for (const part of row.transcript_parts || [{text: row.display_text ?? row.text}]) {
        if (part.speaker_change) {
          const change = document.createElement('span');
          change.className = 'log-speaker-change';
          change.textContent = '换人';
          output.append(change);
        }
        output.append(document.createTextNode(part.text));
      }
    }
  } else {
    output.textContent = row.text ?? row.error ?? row.op ?? row.action ?? '';
  }
  summary.append(heading, output);
  details.append(summary);
  details.ontoggle = () => {
    if (details.open && details.childElementCount === 1) {
      const raw = document.createElement('pre');
      raw.textContent = JSON.stringify(row, null, 2);
      details.append(raw);
    }
  };
  entry.append(details);
  if (row.type === 'result' && row.event === 'audio') entry.append(turnAudioControls(review, sessionId, row.turn));
  return entry;
}

class MeetingTurnLog {
  constructor(get, download, review = null) {
    this.get = get;
    this.download = download;
    this.review = review;
    this.protocol = 'fourth';
    this.current = logHistory(this.protocol);
    this.view = this.current;
    this.currentId = null;
    this.selectedId = null;
    this.offset = 0;
    this.rendered = 0;
    this.filter = 'model';
    this.sessions = new Map();
    this.unsaved = new Map();
    this.loading = false;
    this.archive = new MeetingLogArchive({onError: () => this.warnStorage()});
    const follow = get('logFollow');
    follow.onchange = () => {
      if (follow.checked) this.showLatest();
    };
    get('logFilter').onchange = () => {
      this.filter = get('logFilter').value;
      this.offset = follow.checked ? Math.max(0, this.visible().length - LOG_PAGE_SIZE) : 0;
      this.draw(follow.checked);
    };
    get('logSession').onchange = () => this.select(get('logSession').value);
    get('logFirst').onclick = () => this.go(0);
    get('logPrev').onclick = () => this.go(this.offset - LOG_PAGE_SIZE);
    get('logNext').onclick = () => this.go(this.offset + LOG_PAGE_SIZE);
    get('logLatest').onclick = () => {
      follow.checked = false;
      this.showLatest();
    };
    const box = get('events');
    box.onclick = event => {
      if (event.target.closest('summary, button')) follow.checked = false;
    };
    box.onscroll = () => {
      if (follow.checked && box.scrollTop + box.clientHeight < box.scrollHeight - 4) follow.checked = false;
    };
    get('logPanel').ontoggle = () => { if (get('logPanel').open) this.draw(follow.checked); };
    get('logTxt').onclick = () => this.export(false);
    get('logJsonl').onclick = () => this.export(true);
    this.draw(false);
    this.restore();
  }

  warnStorage() {
    const warning = this.get('logWarning');
    warning.hidden = false;
    warning.textContent = '浏览器未能完整读写本地日志。请在关闭或刷新页面前下载当前会议的完整 JSONL。';
  }

  async restore() {
    try {
      for (const session of await this.archive.list()) this.sessions.set(session.id, session);
      this.updateSessions();
      if (!this.selectedId && this.sessions.size) await this.select(this.get('logSession').value);
    } catch { this.warnStorage(); }
  }

  updateSessions() {
    const menu = this.get('logSession');
    const names = {file: '音视频文件', microphone: '麦克风', system: '电脑声音'};
    menu.replaceChildren();
    const sessions = [...this.sessions.values()].sort((a, b) => b.startedAt.localeCompare(a.startedAt));
    for (const session of sessions) {
      const option = document.createElement('option');
      option.value = session.id;
      option.textContent = new Date(session.startedAt).toLocaleString('zh-CN', {hour12: false})
        + ' · ' + (session.sourceName || names[session.sourceKind] || session.sourceKind)
        + (session.id === this.currentId ? '（本次）' : '');
      menu.append(option);
    }
    menu.disabled = !sessions.length;
    if (this.selectedId) menu.value = this.selectedId;
  }

  start(sourceKind, sourceName = '') {
    const previous = this.currentId;
    if (previous) {
      this.unsaved.set(previous, this.current);
      this.archive.flush(previous).then(() => this.unsaved.delete(previous), () => this.warnStorage());
    }
    this.currentId = this.archive.start(sourceKind, sourceName);
    this.current = logHistory(this.protocol);
    this.sessions.set(this.currentId, {id: this.currentId, startedAt: new Date().toISOString(), sourceKind, sourceName});
    this.selectedId = this.currentId;
    this.updateSessions();
    this.select(this.currentId);
  }

  async select(id) {
    this.review?.stop();
    this.selectedId = id;
    this.offset = 0;
    this.get('logFollow').checked = false;
    this.view = id === this.currentId ? this.current : this.unsaved.get(id);
    this.loading = !this.view;
    if (!this.view) this.view = logHistory();
    this.draw(false);
    if (!this.loading) return;
    const history = this.view;
    try {
      const records = await this.archive.read(id);
      if (this.view !== history) return;
      for (const row of records) indexLogRow(history, row);
    } catch { if (this.view === history) this.warnStorage(); }
    if (this.view === history) {
      this.loading = false;
      this.draw(false);
    }
  }

  append(row) {
    // Transcript corrections can mutate segment objects after the frame is received.
    const original = JSON.parse(JSON.stringify(row));
    indexLogRow(this.current, original);
    if (this.currentId) this.archive.append(this.currentId, original);
    if (this.view !== this.current) return;
    const entries = this.visible();
    if (entries[entries.length - 1] !== original) {
      if (this.current.all.length === 1 || (this.get('logPanel').open && (row.type === 'result' || row.type === 'error'))) this.controls();
      return;
    }
    const follow = this.get('logFollow').checked;
    const next = follow ? Math.max(0, entries.length - LOG_PAGE_SIZE) : this.offset;
    if (!this.get('logPanel').open) { this.offset = next; return; }
    const box = this.get('events');
    if (next === this.offset + 1 && box.firstElementChild) {
      box.firstElementChild.remove();
      this.rendered--;
    }
    this.offset = next;
    if (this.rendered < Math.min(LOG_PAGE_SIZE, entries.length - next)) {
      box.append(logEntry(original, this.review, this.selectedId));
      this.rendered++;
      if (follow) box.scrollTop = box.scrollHeight;
    }
    this.controls();
  }

  visible() { return this.view[this.filter]; }

  showLatest() {
    this.offset = Math.max(0, this.visible().length - LOG_PAGE_SIZE);
    this.draw(true);
  }

  go(offset) {
    this.get('logFollow').checked = false;
    this.offset = Math.max(0, Math.min(offset, Math.max(0, this.visible().length - 1)));
    this.draw(false);
  }

  controls() {
    const n = this.visible().length;
    const last = Math.min(this.offset + LOG_PAGE_SIZE, n);
    const g = this.get;
    const commit = this.view.protocol === 'second' ? '正文' : 'DIRECT';
    g('logTitle').textContent = '逐轮日志 · WAIT / ' + commit;
    g('committedFilter').textContent = '仅' + (commit === 'DIRECT' ? ' DIRECT' : '正文');
    g('logCounts').textContent = 'WAIT ' + this.view.WAIT.length + ' · ' + commit + ' ' + this.view.DIRECT.length
      + (this.view.gaps.length ? ' · 缺口 ' + this.view.gaps.length : '');
    const {lastDirect, waitRun, waitSeconds, error} = this.view;
    const details = [];
    if (lastDirect) {
      const segments = lastDirect.segments || [];
      const lastSpan = segments[segments.length - 1];
      const when = Number.isFinite(lastDirect.source_end_seconds) ? '原文件 ' + lastDirect.source_end_seconds.toFixed(2) + ' 秒'
        : Number.isFinite(lastDirect.end_seconds) ? lastDirect.end_seconds.toFixed(1) + ' 秒' : '第 ' + lastDirect.turn + ' 轮';
      details.push('最后一次' + (commit === 'DIRECT' ? ' DIRECT ' : '提交正文') + '在 ' + when
        + (lastSpan ? '（' + lastSpan.language + ' · ' + lastSpan.speaker + '）' : ''));
    }
    if (waitRun) details.push('连续 WAIT ' + waitRun + ' 个音频块' + (Number.isFinite(waitSeconds) ? ' / ' + waitSeconds.toFixed(1) + ' 秒' : ''));
    g('logProgress').textContent = details.join('；');
    g('logError').hidden = !error;
    if (error) {
      const position = Number.isFinite(error.source_heard_seconds) ? '，最后处理到原文件 ' + error.source_heard_seconds.toFixed(2) + ' 秒'
        : Number.isFinite(error.heard_seconds) ? '，最后处理到 ' + error.heard_seconds.toFixed(1) + ' 秒' : '';
      g('logError').textContent = '服务已报错' + position + '：' + error.error;
    }
    g('logRange').textContent = this.loading ? '正在读取会议日志…'
      : n ? '第 ' + (this.offset + 1) + '–' + last + ' 条，共 ' + n + ' 条' : '暂无记录';
    g('logFirst').disabled = g('logPrev').disabled = this.offset === 0;
    g('logNext').disabled = last >= n;
    g('logLatest').disabled = !n;
    g('logTxt').disabled = this.loading || !this.view.model.length;
    g('logJsonl').disabled = this.loading || !this.view.all.length;
  }

  draw(bottom) {
    const box = this.get('events');
    const entries = this.visible().slice(this.offset, this.offset + LOG_PAGE_SIZE);
    if (this.get('logPanel').open) {
      box.replaceChildren(...entries.map(row => logEntry(row, this.review, this.selectedId)));
      box.scrollTop = bottom ? box.scrollHeight : 0;
      this.rendered = entries.length;
    } else {
      this.rendered = 0;
    }
    this.controls();
  }

  export(jsonl) {
    if (this.loading) return;
    const rows = jsonl ? this.view.all : this.view.model;
    if (!rows.length) return;
    const lines = rows.map(row => jsonl ? JSON.stringify(row) : logHeading(row) + '  ' + (row.text ?? row.action));
    const session = this.sessions.get(this.selectedId);
    const date = session.startedAt.slice(0, 19).replace(/:/g, '-');
    const source = (session.sourceName || session.sourceKind).replace(/[<>:"/\\|?*\x00-\x1f\x7f]/g, '_').slice(0, 64);
    const name = ['omnicommit', date + 'Z', source, session.id.slice(0, 8)].join('-')
      + (jsonl ? '-events.jsonl' : '-turns.txt');
    this.download(name, jsonl ? 'application/x-ndjson;charset=utf-8' : 'text/plain;charset=utf-8', lines.join('\n') + '\n');
  }
}
