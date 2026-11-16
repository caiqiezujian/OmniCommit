const $ = id => document.getElementById(id);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const selected = () => document.querySelector("input[name=source]:checked").value;
const player = $("player");
let health, running, mediaURL;
let events = [];
let transcript = [];

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
$("saveText").onclick = () => download("third.txt", "text/plain;charset=utf-8", transcript.join(""));
$("saveLog").onclick = () => download("third.jsonl", "application/x-ndjson", events.map(e => JSON.stringify(e) + "\n").join(""));

function showResult(result) {
  if (result.type !== "result") return;
  events.push(result);
  $("saveLog").disabled = false;
  if (result.action === "DIRECT") {
    const text = result.display_text;
    transcript.push(text);
    const box = $("words");
    const follow = box.scrollHeight - box.scrollTop - box.clientHeight < 32;
    box.append(document.createTextNode(text));
    if (follow) box.scrollTop = box.scrollHeight;
    $("saveText").disabled = false;
  }
  const li = document.createElement("li");
  li.textContent = (result.event === "end_of_audio" ? "音频结束" : result.end_seconds.toFixed(2) + " s") +
    "  " + result.text + (result.recovery ? "  [补识别：" + result.recovery.status + "]" : "");
  $("turns").prepend(li);
  if ($("turns").children.length > 60) $("turns").lastElementChild.remove();
  $("heard").textContent = result.end_seconds.toFixed(1) + " s";
  $("latency").textContent = result.generation_seconds.toFixed(2) + " s";
}

class Channel {
  constructor(onFailure) {
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
  if (!window.isSecureContext || !navigator.mediaDevices) throw Error("采音需要 HTTPS 或 localhost。");
  state.stream = state.kind === "system"
    ? await navigator.mediaDevices.getDisplayMedia({video: true, audio: true})
    : await navigator.mediaDevices.getUserMedia({audio: true});
  if (!state.stream.getAudioTracks().length) throw Error("未收到声音轨道。共享电脑声音时须勾选共享音频。");
  state.context = new AudioContext({sampleRate: 16000});
  await state.context.suspend();
  if (state.context.sampleRate !== 16000) throw Error("浏览器未提供所需的 16 kHz 采样率。");
  await state.context.audioWorklet.addModule("/web/capture.js");
}

async function streamLive(state, channel, ready) {
  const ac = state.context;
  const node = state.node = new AudioWorkletNode(ac, "third-capture", {
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

async function streamFile(state, channel, ready, upload) {
  const follow = $("playback").checked;
  const end = upload.duration_seconds;
  let release = 0, heard = 0, turn = 1;
  if (follow) {
    if (player.error) throw Error("浏览器无法播放此文件；取消勾选跟随播放器可逐块识别。");
    if (player.currentTime !== 0) {
      await new Promise(resolve => {
        player.addEventListener("seeked", resolve, {once: true});
        player.currentTime = 0;
      });
    }
    state.fileStarted = true;
    try { await player.play(); } catch { status("请点击播放器开始；转录将跟随播放。"); }
  }
  $("stop").disabled = false;
  while (true) {
    if (state.fileError) throw state.fileError;
    if (channel.error) throw channel.error;
    const nextFrame = Math.ceil(turn * Math.round(1000 * ready.chunk_seconds) * upload.sample_rate / 1000);
    const played = follow ? (player.ended ? end : player.currentTime) : nextFrame / upload.sample_rate;
    const target = state.stopping ? state.finalRelease : played;
    release = Math.max(release, Math.min(target, end));
    const final = state.stopping || release >= end;
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
  const state = running = {kind: selected(), queue: new CaptureQueue(), fileHeard: 0, stopping: false};
  $("start").disabled = $("settings").disabled = true;
  try {
    let upload;
    status("准备音频…");
    if (state.kind === "file") {
      if (!health.file_input) throw Error("服务器需要 FFmpeg 才能读取音视频文件。");
      const file = $("file").files[0];
      if (!file) throw Error("请选择音视频文件。");
      const response = await fetch("/api/media", {method: "POST", body: file});
      upload = await response.json();
      if (!response.ok) throw Error(upload.error || "文件上传失败");
    } else await openCapture(state);
    const channel = state.channel = new Channel(error => state.queue.fail(error));
    await channel.open;
    const ready = await channel.request({
      type: "start", sample_rate: 16000, source_kind: state.kind, media_id: upload?.media_id
    });
    if (ready.protocol !== "third") throw Error("服务未加载 Third 协议");
    events = []; transcript = [];
    $("words").replaceChildren(); $("turns").replaceChildren();
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

fetch("/health").then(r => r.json()).then(info => {
  if (!info.ready || info.protocol !== "third") throw Error("Third 服务尚未就绪");
  health = info;
  $("health").textContent = [info.backend, info.action_protocol === "explicit" ? "显式动作" : "纯正文",
    info.gap_recovery ? "有声等待补识别已开启" : "补识别已关闭"].join(" · ");
  $("step").textContent = Math.round(info.chunk_seconds * 1000) + " ms";
  $("start").disabled = false;
}).catch(error => { $("health").textContent = error.message; });
