const $ = id => document.getElementById(id);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const selected = () => document.querySelector("input[name=source]:checked").value;
const player = $("player");
let health, running, mediaURL;
let transcript = [];
let currentParagraph = null;
let gapCount = 0;
const turnLog = new MeetingTurnLog($, download);

function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function refreshSource() {
  const file = selected() === "file";
  $("fileOptions").hidden = !file;
  $("captureHelp").hidden = file;
  player.hidden = !file || !$("file").files[0] || !$("playback").checked;
  if (!file) player.pause();
}
for (const radio of document.querySelectorAll("input[name=source]")) radio.onchange = refreshSource;
$("playback").onchange = refreshSource;
$("file").onchange = () => {
  if (mediaURL) URL.revokeObjectURL(mediaURL);
  const file = $("file").files[0];
  $("fileStart").value = $("fileEnd").value = "";
  mediaURL = file ? URL.createObjectURL(file) : null;
  if (mediaURL) player.src = mediaURL;
  else player.removeAttribute("src");
  refreshSource();
};

function download(name, type, value) {
  const url = URL.createObjectURL(new Blob([value], {type}));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 3000);
}
$("saveText").onclick = () => download("fourth.txt", "text/plain;charset=utf-8", transcript.join(""));
$("saveLog").onclick = () => download("fourth.jsonl", "application/x-ndjson", turnLog.current.model.map(e => JSON.stringify(e) + "\n").join(""));

function showResult(result) {
  if (result.type !== "result") return;
  $("saveLog").disabled = false;
  if (result.truncation?.status === "gap") {
    const message = gapMessage(result);
    gapCount++;
    $("gapNotice").textContent = "本场有 " + gapCount + " 条转录缺口记录。最近一次：" + message + "。";
    $("gapNotice").hidden = false;
    if (transcript.length) transcript.push("\n");
    transcript.push("[" + message + "]");
    currentParagraph = null;
    $("saveText").disabled = false;
  }
  if (result.action === "DIRECT") {
    const box = $("words");
    const follow = box.scrollHeight - box.scrollTop - box.clientHeight < 32;
    for (const part of result.transcript_parts || [{text: result.display_text}]) {
      if (!currentParagraph || part.speaker_change) {
        if (transcript.length) transcript.push("\n");
        currentParagraph = document.createElement("div");
        currentParagraph.className = part.speaker_change ? "segment speaker-change" : "segment";
        if (part.speaker_change) currentParagraph.setAttribute("aria-label", "换人");
        box.append(currentParagraph);
      }
      currentParagraph.append(document.createTextNode(part.text));
      transcript.push(part.text);
    }
    if (follow) box.scrollTop = box.scrollHeight;
    $("saveText").disabled = false;
  }
  $("heard").textContent = result.end_seconds.toFixed(1) + " s";
  $("latency").textContent = result.generation_seconds.toFixed(2) + " s";
  if (Number.isFinite(result.window_start_seconds)) {
    const seconds = n => Number(n.toFixed(3));
    let metric = "模型窗口：" + seconds(result.window_start_seconds) + "–" + seconds(result.end_seconds) + " 秒";
    if (result.discarded_seconds > 0) metric += " · 本轮丢弃 " + seconds(result.discarded_seconds) + " 秒";
    if (Number.isInteger(result.num_cached_tokens) && result.num_cached_tokens >= 0) {
      metric += " · KV 命中 " + result.num_cached_tokens + " tokens";
    }
    $("windowStatus").textContent = metric;
    $("windowStatus").hidden = false;
  }
}

class Channel {
  constructor(onFailure, onEvent) {
    this.socket = new WebSocket((location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host + "/stream");
    this.open = new Promise((resolve, reject) => {
      this.socket.onopen = resolve;
      this.socket.onclose = () => {
        this.error = new Error("转录连接已关闭");
        reject(this.error);
        if (this.pending) this.pending.reject(this.error);
        onFailure(this.error);
      };
      this.socket.onerror = () => reject(new Error("无法连接转录服务"));
    });
    this.socket.onmessage = event => {
      const pending = this.pending;
      if (!pending) return;
      this.pending = null;
      try {
        const row = JSON.parse(event.data);
        onEvent(row);
        if (row.type === "error") throw Error(row.error);
        pending.resolve(row);
      } catch (error) { pending.reject(error); }
    };
  }

  request(packet) {
    if (this.error) return Promise.reject(this.error);
    return new Promise((resolve, reject) => {
      this.pending = {resolve, reject};
      this.socket.send(packet instanceof ArrayBuffer ? packet : JSON.stringify(packet));
    });
  }
}

class CaptureQueue {
  constructor() {
    this.chunks = [];
    this.samples = 0;
    this.ended = false;
  }
  push(chunk) {
    if (this.error || this.ended) return;
    this.chunks.push(chunk);
    this.samples += chunk.byteLength / 2;
    this.update();
    if (this.samples > 16000 * 60) {
      this.fail(new Error("音频积压超过 60 秒，已停止录制。已提交的正文可以导出。"));
    }
  }
  update() {
    $("queued").textContent = (this.samples / 16000).toFixed(1) + " s";
    this.wake?.();
  }
  finish() { this.ended = true; this.wake?.(); }
  fail(error) { this.error = error; this.wake?.(); }
  async next() {
    while (!this.chunks.length && !this.ended && !this.error) {
      await new Promise(resolve => { this.wake = resolve; });
    }
    if (this.error) throw this.error;
    const chunk = this.chunks.shift();
    if (chunk) this.samples -= chunk.byteLength / 2;
    this.update();
    return chunk;
  }
}

async function openCapture(state) {
  if (!window.isSecureContext) throw Error("采音需要 HTTPS 或 localhost。");
  const devices = navigator.mediaDevices;
  if (state.kind === "system") {
    if (typeof devices?.getDisplayMedia !== "function") {
      throw Error("当前浏览器未提供屏幕共享，请用支持共享音频的桌面浏览器打开会议页。");
    }
    state.stream = await devices.getDisplayMedia({video: true, audio: true, systemAudio: "include"});
    if (!state.stream.getAudioTracks().length) {
      throw Error("未收到电脑声音。请重新选择提供音频选项的标签页或屏幕，勾选“共享音频”。");
    }
  } else {
    if (typeof devices?.getUserMedia !== "function") throw Error("当前浏览器未提供麦克风采集。");
    state.stream = await devices.getUserMedia({audio: true});
    if (!state.stream.getAudioTracks().length) throw Error("未收到麦克风声音。");
  }
  state.context = new AudioContext({sampleRate: 16000});
  await state.context.suspend();
  if (state.context.sampleRate !== 16000) throw Error("浏览器未提供所需的 16 kHz 采样率。");
  await state.context.audioWorklet.addModule("/web/meeting-v2/capture.js?v=speaker1");
}

async function streamLive(state, channel, ready) {
  const ac = state.context;
  const node = state.node = new AudioWorkletNode(ac, "fourth-capture", {
    processorOptions: {samples: Math.round(16000 * ready.chunk_seconds)}
  });
  node.port.onmessage = e => {
    if (e.data === "stopped") state.queue.finish();
    else state.queue.push(e.data);
  };
  node.onprocessorerror = () => state.queue.fail(new Error("浏览器采音中断"));
  ac.onstatechange = () => {
    if (!state.stopping && ac.state === "interrupted") state.queue.fail(new Error("音频设备中断"));
  };
  state.stream.getAudioTracks()[0].onended = () => stop();
  ac.createMediaStreamSource(state.stream).connect(node);
  node.connect(ac.destination); // The worklet produces silence on the output.
  await ac.resume();
  $("stop").disabled = false;
  let count = 0;
  for (;;) {
    const chunk = await state.queue.next();
    if (!chunk) break;
    showResult(await channel.request(chunk));
    count++;
  }
  if (count) showResult(await channel.request({type: "end"}));
}

function fileRangeOptions() {
  const range = {};
  for (const [id, key] of [["fileStart", "file_start_seconds"], ["fileEnd", "file_end_seconds"]]) {
    const value = $(id).value.trim();
    if (!value) continue;
    const seconds = value.split(":").reduce((total, part) => total * 60 + Number(part), 0);
    if (!/^(?:\d+(?:\.\d+)?|\d+:[0-5]\d(?::[0-5]\d)?(?:\.\d+)?)$/.test(value) || !Number.isFinite(seconds)) {
      throw Error("转录时间请输入非负秒数、分:秒或时:分:秒。");
    }
    range[key] = seconds;
  }
  if (range.file_end_seconds <= (range.file_start_seconds ?? 0)) throw Error("转录终点必须晚于起点。");
  return range;
}

function waitForPlayer(event, trigger) {
  return new Promise((resolve, reject) => {
    const finish = error => {
      clearTimeout(timer);
      player.removeEventListener(event, done);
      player.removeEventListener("error", failed);
      if (error) reject(error);
      else resolve();
    };
    const done = () => finish();
    const failed = () => finish(Error("播放器无法定位到转录起点，请检查文件或取消勾选跟随播放器。"));
    const timer = setTimeout(failed, 10000);
    player.addEventListener(event, done);
    player.addEventListener("error", failed);
    try { trigger?.(); } catch (error) { finish(error); }
  });
}

async function streamFile(state, channel, ready, upload) {
  const follow = $("playback").checked;
  const start = ready.file_start_seconds, end = ready.file_end_seconds;
  const startFrame = Math.round(start * upload.sample_rate);
  let release = start, heard = start, turn = 1;
  state.fileHeard = start;
  if (follow) {
    if (player.error) throw Error("浏览器无法播放此文件；取消勾选跟随播放器可逐块识别。");
    player.pause();
    if (player.readyState < 1) await waitForPlayer("loadedmetadata");
    if (player.currentTime !== start || player.seeking) {
      await waitForPlayer("seeked", () => { player.currentTime = start; });
    }
    if (Math.abs(player.currentTime - start) > 0.1) throw Error("播放器未能定位到转录起点。");
    state.fileEnd = end;
    state.fileStarted = true;
    try { await player.play(); } catch { status("请点击播放器开始；转录将跟随播放。"); }
  }
  $("stop").disabled = false;
  while (true) {
    if (state.fileError) throw state.fileError;
    if (channel.error) throw channel.error;
    const nextFrame = Math.ceil(turn * Math.round(1000 * ready.chunk_seconds) * upload.sample_rate / 1000);
    if (follow && player.ended && player.currentTime + 0.1 < end) throw Error("播放已提前结束，请取消跟随播放器后重试。");
    const played = follow ? (player.ended ? end : player.currentTime) : (startFrame + nextFrame) / upload.sample_rate;
    const target = state.stopping ? state.finalRelease : played;
    release = Math.max(release, Math.min(target, end));
    const final = state.stopping || release >= end;
    if (follow && final && !player.paused) player.pause();
    $("queued").textContent = Math.max(0, release - heard).toFixed(1) + " s";
    if (!final && release - heard + 1 / upload.sample_rate < ready.chunk_seconds) {
      await sleep(60);
      continue;
    }
    const row = await channel.request({type: "file_next", release_seconds: release, final});
    if (row.type === "file_done") break;
    if (row.type === "file_wait") { await sleep(60); continue; }
    showResult(row);
    heard = state.fileHeard = row.source_end_seconds;
    turn++;
    if (row.file_complete) break;
  }
  showResult(await channel.request({type: "end"}));
}

function stop() {
  if (!running || running.stopping) return;
  running.stopping = true;
  $("stop").disabled = true;
  if (running.kind === "file") {
    running.finalRelease = $("playback").checked ? player.currentTime : running.fileHeard;
    player.pause();
  } else {
    running.node?.port.postMessage("stop");
  }
  status("正在提交最后的音频…");
}
$("stop").onclick = stop;
player.addEventListener("timeupdate", () => {
  if (running?.fileStarted && player.currentTime >= running.fileEnd && !player.paused) player.pause();
});
player.addEventListener("seeking", () => {
  if (running?.kind === "file" && running.fileStarted && !running.stopping) {
    running.fileError = new Error("播放位置已改变，请重新开始转录。");
  }
});
player.addEventListener("error", () => {
  if (running?.kind === "file" && $("playback").checked) {
    running.fileError = new Error("浏览器无法播放此文件；取消勾选跟随播放器可逐块识别。");
  }
});

$("start").onclick = async () => {
  if (!health || running) return;
  let windowOptions = {};
  if (health.context_policy === "window") {
    const windowSeconds = $("windowSeconds").valueAsNumber;
    const discardSeconds = $("discardSeconds").valueAsNumber;
    if (!Number.isFinite(windowSeconds) || windowSeconds < health.chunk_seconds) {
      status("窗口至少需要容纳一个音频块。", true);
      $("windowSeconds").focus();
      return;
    }
    if (Number.isFinite(health.max_window_seconds) && windowSeconds > health.max_window_seconds) {
      status("本服务的窗口上限是 " + health.max_window_seconds + " 秒。", true);
      $("windowSeconds").focus();
      return;
    }
    if (!Number.isFinite(discardSeconds) || discardSeconds <= 0 || discardSeconds > windowSeconds) {
      status("每次丢弃长度须大于零，且不能超过窗口长度。", true);
      $("discardSeconds").focus();
      return;
    }
    windowOptions = {window_seconds: windowSeconds, discard_seconds: discardSeconds};
  }
  const state = running = {kind: selected(), queue: new CaptureQueue(), fileHeard: 0, stopping: false};
  $("start").disabled = $("settings").disabled = true;
  try {
    let upload, range = {};
    status("准备音频…");
    if (state.kind === "file") {
      if (!health.file_input) throw Error("服务器需要 FFmpeg 才能读取音视频文件。");
      const file = $("file").files[0];
      if (!file) throw Error("请选择音视频文件。");
      range = fileRangeOptions();
      player.pause();
      const response = await fetch("/api/media", {method: "POST", body: file});
      upload = await response.json();
      if (!response.ok) throw Error(upload.error || "文件上传失败");
    } else await openCapture(state);
    let logStarted = false;
    const channel = state.channel = new Channel(error => state.queue.fail(error), row => {
      if (row.type === "ready" && row.protocol === "fourth") {
        turnLog.start(state.kind, state.kind === "file" ? $("file").files[0].name : "");
        logStarted = true;
      }
      if (logStarted) turnLog.append(row);
    });
    await channel.open;
    const ready = await channel.request({
      type: "start", sample_rate: 16000, source_kind: state.kind, media_id: upload?.media_id,
      ...windowOptions, ...range
    });
    if (ready.type !== "ready" || ready.protocol !== "fourth") throw Error("服务未加载 Fourth 协议");
    if (upload) {
      const start = ready.file_start_seconds, end = ready.file_end_seconds;
      const tolerance = 1 / upload.sample_rate;
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start ||
          Math.abs(start - (range.file_start_seconds ?? 0)) > tolerance ||
          Math.abs(end - (range.file_end_seconds ?? upload.duration_seconds)) > tolerance) {
        throw Error("服务端未确认所选文件区间，请核对 Fourth 服务版本并重新开始。");
      }
    }
    if (health.context_policy === "window" && (ready.window_seconds !== windowOptions.window_seconds
        || ready.discard_seconds !== windowOptions.discard_seconds)) {
      throw Error("服务端未确认所选窗口，请核对 Fourth 服务版本并重新开始。");
    }
    transcript = [];
    currentParagraph = null;
    gapCount = 0;
    $("gapNotice").hidden = true;
    $("words").replaceChildren();
    $("windowStatus").hidden = true;
    $("heard").textContent = $("queued").textContent = "0.0 s";
    $("saveLog").disabled = $("saveText").disabled = true;
    status("正在转录");
    if (upload) await streamFile(state, channel, ready, upload);
    else await streamLive(state, channel, ready);
    status("转录结束");
  } catch (error) {
    status(error.message, true);
  } finally {
    player.pause();
    state.stream?.getTracks().forEach(t => t.stop());
    if (state.context && state.context.state !== "closed") await state.context.close();
    state.channel?.socket.close();
    running = null;
    $("stop").disabled = true;
    $("start").disabled = $("settings").disabled = false;
  }
};

fetch("/health", {cache: "no-store"}).then(r => r.json()).then(info => {
  if (!info.ready || info.protocol !== "fourth") throw Error("Fourth 服务尚未就绪");
  if (!$("windowSettings") || !$("gapNotice") || !$("fileStart") || !$("fileEnd")) {
    // Older HTML may still load this script through its unversioned URL.
    if (location.search === "?ui=speaker1") throw Error("服务仍返回旧版页面，请确认 Fourth 已更新并刷新。");
    location.replace("/?ui=speaker1");
    return;
  }
  const hasWindow = info.context_policy === "window";
  if (hasWindow && (!Number.isFinite(info.window_seconds) || !Number.isFinite(info.discard_seconds))) {
    throw Error("服务端未返回窗口设置，请核对 Fourth 服务版本。");
  }
  $("windowSettings").hidden = !hasWindow;
  if (hasWindow) {
    $("windowSeconds").value = String(info.window_seconds);
    $("windowSeconds").min = String(info.chunk_seconds);
    if (Number.isFinite(info.max_window_seconds)) $("windowSeconds").max = String(info.max_window_seconds);
    $("discardSeconds").value = String(info.discard_seconds);
    $("windowSeconds").disabled = $("discardSeconds").disabled = false;
  }
  health = info;
  $("health").textContent = [info.backend, info.action_protocol === "speaker" ? "说话人切换" : info.action_protocol === "explicit" ? "显式动作" : "纯正文",
    info.gap_recovery ? "有声等待补识别已开启" : "补识别已关闭",
    ...(hasWindow ? [] : ["当前策略 " + (info.context_policy || "未知") + "，此会话不支持自选窗口"])].join(" · ");
  $("step").textContent = Math.round(info.chunk_seconds * 1000) + " ms";
  $("start").disabled = false;
}).catch(error => { $("health").textContent = error.message; });
