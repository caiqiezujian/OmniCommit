// Browser-independent contract checks for file release and speaker corrections.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {webcrypto} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(here, '../omnicommit/web/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
assert.doesNotMatch(html, /id="token"|Authorization|访问口令/);
function element() {
  return {value:'', checked:false, hidden:false, disabled:false, textContent:'', dataset:{},
    classList:{toggle(){}}, children:[], append(...nodes){this.children.push(...nodes);},
    replaceChildren(...nodes){this.children=nodes;}, click(){}, pause(){this.paused=true;},
    currentTime:0, paused:false, files:[]};
}
const elements = new Map();
const tabs = ['microphone', 'system', 'file'].map(mode => ({...element(),dataset:{mode}}));
const document = {getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);},
  querySelectorAll(){return tabs;}, createElement(){return element();}, createTextNode(s){return s;}};
const sent=[];
const context = vm.createContext({document, console, crypto:webcrypto, WebSocket:{OPEN:1},
  URL:{createObjectURL(){return 'blob:x';}, revokeObjectURL(){}}, Blob, Int16Array,
  Float32Array, Math, Number, JSON, setTimeout, clearTimeout, setInterval, clearInterval,
  prompt(){return null;}});
for (const match of html.matchAll(/<script src="\/web\/([^"]+)"><\/script>/g)) {
  vm.runInContext(fs.readFileSync(path.join(here, '../omnicommit/web', match[1]), 'utf8'), context);
}
vm.runInContext(script, context);
assert.match(html, /\[hidden\]\s*\{\s*display:\s*none\s*!important/);
tabs[1].onclick();
assert.equal(elements.get('mixLabel').hidden, false);
tabs[0].onclick();
assert.equal(elements.get('mixLabel').hidden, true);
context.sent = sent;
// A file pre-seeked by the user must restart at zero before media submission.
const media = elements.get('media');
media.duration = 20;
let playhead = 10;
let seekDone;
media.addEventListener = (name, cb) => { if (name === 'seeked') seekDone = cb; };
Object.defineProperty(media, 'currentTime', {get(){return playhead;}, set(value){
  playhead = value;
  if (seekDone) queueMicrotask(seekDone);
}});
await vm.runInContext('resetPlayback()', context);
assert.equal(media.currentTime, 0);
vm.runInContext('initializing=true', context);
media.paused = false;
media.onplay();
assert.equal(media.paused, true);
vm.runInContext('initializing=false', context);
vm.runInContext(`
  mode='file'; fileDuration=10; socket={readyState:1,send(v){sent.push(JSON.parse(v));}};
  document.getElementById('media').currentTime=1.25;
  pollFile();`, context);
assert.equal(sent[0].type, 'file_next');
assert.equal(sent[0].release_seconds, 1.25);
vm.runInContext(`busy=false; fileStopping=true; fileFinalSeconds=1.25; document.getElementById('media').currentTime=1.75; pollFile();`, context);
assert.equal(sent[1].final, true);
assert.equal(sent[1].release_seconds, 1.25);
// A sub-millisecond unplayed tail must not be silently treated as complete.
vm.runInContext(`
  busy=false; mode='file'; failed=false; fileDuration=16008/16000;
  fileSampleRate=16000; fileSamples=16008;
  socket={readyState:1,close(){},send(v){sent.push(JSON.parse(v));}};
  document.getElementById('file').disabled=true;
  document.getElementById('media').currentTime=1;
  document.getElementById('media').paused=false;
  document.getElementById('media').onended();
`, context);
assert.equal(vm.runInContext('failed',context),true);
assert.equal(media.paused,true);
assert.equal(elements.get('file').disabled,false);
// Changing the input during preparation must leave the existing source alone.
vm.runInContext(`mediaURL='blob:locked'; document.getElementById('file').disabled=true; document.getElementById('file').onchange();`,context);
assert.equal(vm.runInContext('mediaURL',context),'blob:locked');
// Detect an unplayed tail in the original sample rate, not only 16 kHz.
vm.runInContext(`
  mode='file'; failed=false; fileStopping=false; ending=false;
  fileDuration=48024/48000; fileSampleRate=48000; fileSamples=48024;
  socket={readyState:1,close(){},send(v){sent.push(JSON.parse(v));}};
  document.getElementById('media').currentTime=1;
  document.getElementById('media').onended();
`,context);
assert.equal(vm.runInContext('failed',context),true);
// A failed audio worklet drains already received partial PCM and marks risk.
await vm.runInContext(`
  mode='microphone'; capturing=true; ending=false; captureError=false;
  streams=[]; context=null; queue=[]; partial=[0.2,0.3]; socket={readyState:0};
  captureFault();
`,context);
assert.equal(vm.runInContext('ending && captureError && queue.length === 1',context),true);
assert.equal(elements.get('warning').hidden,false);
vm.runInContext(`
  busy=false; finished=true; ending=true;
  speakers=[{speaker:'spk1',name:''},{speaker:'spk2',name:''}];
  segments=[{segment_id:'s1',model_speaker:'spk2',display_speaker:'spk2',language:'zh',text:'你好'}];
  control({op:'correct_segment',segment_id:'s1',speaker:'spk1'});
`, context);
assert.equal(vm.runInContext('segments[0].display_speaker', context), 'spk1');
assert.equal(vm.runInContext('segments[0].model_speaker', context), 'spk2');
vm.runInContext(`control({op:'rename_speaker',speaker:'spk1',name:'张老师'});`,context);
assert.equal(vm.runInContext('speakers[0].name',context), '张老师');

// Review the first page while new turns and file polling continue to arrive.
vm.runInContext(`
  turnLog.start('file');
  $('logPanel').open = true;
  $('logPanel').ontoggle();
  for (let n = 1; n <= 123; n++) {
    log({type:'file_wait'});
    log({type:'result', event:'audio', turn:n, action:'WAIT', text:'<|WAIT|>'});
  }
`, context);
assert.equal(elements.get('events').children.length, 50);
assert.equal(elements.get('logRange').textContent, '第 1–50 条，共 123 条');
elements.get('logNext').onclick();
assert.equal(elements.get('logRange').textContent, '第 51–100 条，共 123 条');
assert.match(html, /<details id="logPanel" open>/);
const panelHtml = html.slice(html.indexOf('<details id="logPanel"'));
for (const id of ['logTxt', 'logJsonl']) assert.ok(panelHtml.indexOf('id="'+id+'"') < panelHtml.indexOf('id="events"'));
const logBox = elements.get('events'), follow = elements.get('logFollow');
logBox.scrollHeight = 780;
follow.checked = true;
follow.onchange();
assert.equal(elements.get('logRange').textContent, '第 74–123 条，共 123 条');
const reading = logBox.children[0];
reading.open = true;
logBox.scrollTop = 224;
follow.checked = false;
follow.onchange();
assert.equal(logBox.scrollTop, 224);
assert.equal(logBox.children[0], reading);
elements.get('logFirst').onclick();
elements.get('logLatest').onclick();
assert.equal(logBox.scrollTop, logBox.scrollHeight);
assert.equal(follow.checked, false);
follow.checked = true;
logBox.onclick({target:{closest(selector){return selector === 'summary' ? reading : null;}}});
assert.equal(follow.checked, false);
const downloads = [];
context.saveLog = (name, type, body) => downloads.push({name, body});
vm.runInContext(`turnLog.download=saveLog; $('logJsonl').onclick(); $('logTxt').onclick();`,context);
assert.equal(downloads[0].body.trimEnd().split('\n').length, 246);
assert.equal(downloads[1].body.trimEnd().split('\n').length, 123);
assert.match(downloads[1].body, /第 1 轮 +<\|WAIT\|>/);
// Rendering and export preserve the original model assignment after local edits.
vm.runInContext(`
  const originalFrame = {type:'result', turn:124, action:'DIRECT', text:'<|DIRECT|><en><spk1>Hello',
    new_segments:[{display_speaker:'spk1'}]};
  log(originalFrame);
  originalFrame.new_segments[0].display_speaker='spk2';
  $('logFilter').value='DIRECT'; $('logFilter').onchange();
  $('logJsonl').onclick();
`, context);
assert.equal(elements.get('logRange').textContent, '第 1–1 条，共 1 条');
assert.equal(JSON.parse(downloads[2].body.trimEnd().split('\n').at(-1)).new_segments[0].display_speaker,'spk1');
vm.runInContext(`
  log({type:'result',event:'audio',turn:125,action:'DIRECT',start_seconds:59.5,end_seconds:60,
    segments:[{language:'zh',speaker:'spk1'}],text:'<|DIRECT|><zh><spk1>你好'});
  log({type:'result',event:'audio',turn:126,action:'WAIT',start_seconds:60,end_seconds:60.5});
  log({type:'file_wait'});
  log({type:'result',event:'audio',turn:127,action:'WAIT',start_seconds:60.5,end_seconds:61});
  log({type:'result',event:'end_of_audio',turn:128,action:'WAIT',start_seconds:61,end_seconds:61});
  log({type:'error',heard_seconds:61,error:'Context audio budget exhausted'});
`,context);
assert.match(elements.get('logProgress').textContent,/60\.0 秒（zh · spk1）.*连续 WAIT 2 个音频块 \/ 1\.0 秒/);
assert.equal(elements.get('logError').hidden,false);
assert.match(elements.get('logError').textContent,/61\.0 秒.*Context audio budget exhausted/);

// Logging and downloading a Second session preserves its sparse model text and spaces.
vm.runInContext(`
  turnLog.start('system');
  log({type:'ready',protocol:'second',chunk_seconds:.7});
  log({type:'result',event:'audio',turn:1,action:'DIRECT',text:'<en><spk4>Yes'});
  log({type:'result',event:'audio',turn:2,action:'DIRECT',text:' I will.'});
  $('logFilter').value='DIRECT'; $('logFilter').onchange();
  $('logJsonl').onclick();
`,context);
assert.equal(elements.get('brand').textContent,'OmniCommit Second');
assert.match(elements.get('serviceDescription').textContent,/700 ms/);
assert.match(elements.get('logTitle').textContent,/WAIT \/ 正文/);
assert.equal(elements.get('committedFilter').textContent,'仅正文');
assert.equal(elements.get('logCounts').textContent,'WAIT 0 · 正文 2');
assert.equal(JSON.parse(downloads.at(-1).body.trimEnd().split('\n').at(-1)).text,' I will.');

vm.runInContext(`
  finished=true; ending=true; busy=false; socket=null;
  speakers=[{speaker:'spk1',name:'甲'},{speaker:'spk2',name:'乙'}];
  segments=[{segment_id:1,display_speaker:'spk1',language:'en',text:'The total is'}];
  render();
`,context);
const transcript = elements.get('transcript');
const firstRow = transcript.children[0];
vm.runInContext(`
  for (const seg of [
    {segment_id:2,display_speaker:'spk1',language:'en',text:' 3.14. Next step?'},
    {segment_id:3,display_speaker:'spk1',language:'zh',text:'可以。继续吗？'},
    {segment_id:4,display_speaker:'spk2',language:'zh',text:'上半句\\r\\n下半句'}]) {
    segments.push(seg);
    appendTranscript($('transcript'), seg);
  }
`,context);
const visibleBodies = () => transcript.children.map(row => row.children.at(-1));
const displayedTxt = () => transcript.children.map(row => {
  const [speaker, language] = row.children[0].textContent.split(' · ');
  return speaker + ' [' + language + '] ' + row.children.at(-1);
}).join('\n') + '\n';
assert.equal(transcript.children[0],firstRow, 'appending new transcript must retain existing DOM');
assert.equal(transcript.children.length,7, 'each commit, sentence boundary and hard break starts a row');
assert.deepEqual(visibleBodies().map(s => s.trim()), [
  'The total is', '3.14.', 'Next step?', '可以。', '继续吗？', '上半句', '下半句'
]);
assert.equal(visibleBodies().slice(1,3).join(''), ' 3.14. Next step?');
assert.equal(vm.runInContext('transcriptText()',context), displayedTxt());
vm.runInContext("download=saveLog; $('exportTxt').onclick(); $('exportJson').onclick()", context);
assert.equal(downloads.at(-2).body, displayedTxt());
assert.deepEqual(JSON.parse(downloads.at(-1).body).segments.map(s => s.text),
  ['The total is',' 3.14. Next step?','可以。继续吗？','上半句\r\n下半句']);
context.prompt=()=> 'spk2';
transcript.children[2].children[1].onclick();  // Second sentence within original segment 2.
assert.equal(vm.runInContext('segments[1].display_speaker',context),'spk2');
assert.deepEqual(transcript.children.map(row => row.children[0].textContent), [
  '甲 · en', '乙 · en', '乙 · en', '甲 · zh', '甲 · zh', '乙 · zh', '乙 · zh'
]);
assert.equal(vm.runInContext('transcriptText()',context), displayedTxt());
// Some server clients use older browsers without a built-in sentence segmenter.
const segmentedBodies = visibleBodies();
vm.runInContext('const nativeSegmenter=Intl.Segmenter; Intl.Segmenter=undefined; render()', context);
assert.deepEqual(visibleBodies(), segmentedBodies);
vm.runInContext('Intl.Segmenter=nativeSegmenter', context);
// The original protocol keeps one row per segment even if that segment has multiple sentences.
vm.runInContext("turnLog.protocol='first';render()", context);
assert.equal(transcript.children.length,4);
assert.equal(visibleBodies()[1], ' 3.14. Next step?');

// Start via the actual page handler. Stub only the browser capture/upload
// devices and transport so lock timing and the real JSON handshake are covered.
assert.match(html, /<label id="chunkLabel" for="chunkMs">/);
assert.match(html, /<input id="chunkMs" type="number"/);
const picker = elements.get('chunkPresets'), field = elements.get('chunkMs');
const chunkSettings = document.getElementById('chunkSettings');
field.focus = () => { field.focused = true; };
chunkSettings.hidden = true;
const sockets = [];
context.location = {protocol:'http:', host:'localhost'};
context.WebSocket = class {
  constructor() { this.frames = []; this.readyState = 1; sockets.push(this); }
  send(frame) { this.frames.push(frame); }
  close() { this.readyState = 3; }
};
context.fetch = async () => ({ok:true, json:async () =>
  ({media_id:'file-7', duration_seconds:20, sample_rate:16000, samples:320000})});
media.play = async () => { media.paused = false; };
elements.get('file').files = [{name:'meeting.wav'}];
vm.runInContext(`
  turnLog.protocol = 'first'; acceptedChunkMs = null;
  prepareLive = async () => { context = {resume: async () => {}, close: async () => {}}; };
`, context);

async function startMeeting(source) {
  context.selectedSource = source;
  vm.runInContext('mode=selectedSource;',context);
  const starting = elements.get('start').onclick();
  assert.equal(picker.disabled, true, 'locked before capture permissions or file upload');
  assert.equal(field.disabled, true);
  await starting;
  const ws = sockets.at(-1);
  ws.onopen();
  return [ws, JSON.parse(ws.frames[0])];
}
vm.runInContext("identifyService({protocol:'first',chunk_seconds:.5})",context);
assert.equal(chunkSettings.hidden, true);
const [legacySocket, legacyStart] = await startMeeting('microphone');
assert.deepEqual(legacyStart, {type:'start',sample_rate:16000,source_kind:'microphone'});
legacySocket.close();
await vm.runInContext('cleanup()',context);
vm.runInContext("identifyService({protocol:'first',chunk_seconds:.5,chunk_ms_choices:[500,700]})",context);
assert.equal(field.hidden, true, 'preset-only servers still show their select');
picker.value = '700';
const [presetSocket, presetStart] = await startMeeting('microphone');
assert.equal(presetStart.chunk_ms, 700);
presetSocket.close();
await vm.runInContext('cleanup()',context);
for (const service of [
  {protocol:'first',chunk_seconds:.8,chunk_ms_choices:[500,700,800,1000],chunk_ms_min:200,chunk_ms_max:5000},
  {protocol:'second',chunk_seconds:.601,chunk_ms_choices:[500,601,700,800,1000],chunk_ms_min:100,chunk_ms_max:1800},
]) {
  // A fresh page receives its capabilities and default from its own service.
  picker.replaceChildren();
  chunkSettings.hidden = true;
  context.stepService = service;
  vm.runInContext('acceptedChunkMs=null; identifyService(stepService)',context);
  assert.equal(chunkSettings.hidden, false);
  assert.equal(field.hidden, false);
  assert.equal(field.value, String(Math.round(service.chunk_seconds * 1000)));
  assert.deepEqual(picker.children.map(option => option.textContent), ['常用值', ...service.chunk_ms_choices.map(ms => ms + ' ms')]);
  assert.equal(elements.get('brand').textContent, service.protocol === 'first' ? 'OmniCommit' : 'OmniCommit Second');
  for (const value of ['', 'x', '400.5', String(service.chunk_ms_min - 1), String(service.chunk_ms_max + 1)]) {
    field.value = value;
    const before = sockets.length;
    await elements.get('start').onclick();
    assert.equal(sockets.length, before, 'invalid step must not open a connection');
    assert.equal(picker.disabled, false);
    assert.match(elements.get('status').textContent, /整数毫秒/);
    assert.equal(field.focused, true);
  }
  for (const [source, ms] of [['microphone',200], ['system',800], ['file',377]]) {
    if (source === 'system') {
      picker.value = String(ms);
      picker.onchange();
      assert.equal(field.value, String(ms));
    } else {
      field.value = String(ms);
    }
    vm.runInContext('identifyService(stepService)',context); // delayed health never overwrites the selection
    const [ws, start] = await startMeeting(source);
    assert.equal(start.chunk_ms, ms);
    assert.equal(start.source_kind, source);
    if (source === 'file') assert.equal(start.media_id,'file-7');
    await ws.onmessage({data:JSON.stringify({...service,type:'ready',chunk_seconds:ms/1000})});
    assert.equal(vm.runInContext('step',context), ms * 16);
    assert.equal(picker.disabled, true);
    assert.match(elements.get('serviceDescription').textContent, new RegExp(ms + ' ms'));
    assert.equal(elements.get('logTitle').textContent, '逐轮日志 · WAIT / ' + (service.protocol === 'first' ? 'DIRECT' : '正文'));
    if (source !== 'file') {
      context.expectedFrames = ms * 16;
      vm.runInContext('capture(new Float32Array(expectedFrames - 1))',context);
      assert.equal(ws.frames.length, 1);
      vm.runInContext('capture(new Float32Array(1))',context);
      assert.equal(ws.frames[1].byteLength, ms * 32);
    }
    vm.runInContext('identifyService(stepService)',context);
    assert.equal(field.value, String(ms));
    await ws.onmessage({data:JSON.stringify({type:'result',event:'end_of_audio',action:'WAIT',text:'<|WAIT|>'})});
    assert.equal(picker.disabled, false);
    assert.equal(field.disabled, false);
  }
}
// Ranged playback sends source-clock release boundaries, even when inference
// is behind the player. Each new start has its own model session.
const rangeStart = elements.get('fileStart'), rangeEnd = elements.get('fileEnd');
rangeStart.focus = rangeEnd.focus = () => {};
media.duration = 100;
let uploads = 0;
context.fetch = async () => {
  uploads++;
  return {ok:true, json:async () =>
    ({media_id:'range-file', duration_seconds:100, sample_rate:44100, samples:4410000})};
};
async function frame(ws, row) {await ws.onmessage({data:JSON.stringify(row)});}
function last(ws) {return JSON.parse(ws.frames.at(-1));}
const readyRange = {type:'ready', chunk_seconds:.5, backend:'replay', file_start_seconds:58.125, file_end_seconds:96};
for (const protocol of ['first', 'second']) {
  rangeStart.value='58.125';rangeEnd.value='1:36';field.value='500';
  const [ws, start] = await startMeeting('file');
  assert.equal(start.file_start_seconds, 58.125);
  assert.equal(start.file_end_seconds, 96);
  assert.equal(rangeStart.disabled && rangeEnd.disabled, true);
  assert.equal(media.currentTime, 58.125);
  media.currentTime = 12;  // preview moved between upload and server confirmation
  await frame(ws, {...readyRange, protocol});
  assert.equal(media.currentTime, 58.125);
  media.currentTime=58.625;
  vm.runInContext('pollFile()',context);
  assert.deepEqual(last(ws), {type:'file_next', release_seconds:58.625});
  await frame(ws, {type:'result',event:'audio',turn:1,action:'WAIT',text:'<|WAIT|>',
    new_segments:[],start_seconds:0,end_seconds:.5,source_start_seconds:58.125,source_end_seconds:58.625});
  assert.equal(transcript.children.length,0, 'WAIT should be logged without a transcript row');
  assert.match(elements.get('processed').textContent, /已处理：0.5 秒 · 原文件 58.63 秒/);
  assert.match(elements.get('pending').textContent, /0.0 秒/);
  assert.match(vm.runInContext("logHeading(turnLog.current.model[0])",context), /原文件 58.125–58.625 s/);
  media.currentTime=96.157;
  media.ontimeupdate();  // model is still busy; playback should stop anyway
  assert.equal(media.paused, true);
  await frame(ws,{type:'file_wait'});
  vm.runInContext('transmit()',context);
  assert.deepEqual(last(ws),{type:'file_next',release_seconds:96,final:true});
  await frame(ws,{type:'file_done'});
  assert.deepEqual(last(ws),{type:'end'});
  await frame(ws,{type:'result',event:'end_of_audio',end_seconds:37.875,source_end_seconds:96});
  assert.equal(rangeStart.disabled || rangeEnd.disabled, false);
  ws.close();
}

for (const [start, end, message] of [['-1','','请输入'],['a','','请输入'],['1:60','','请输入'],
  ['2','1','晚于起点'],['2','2','晚于起点'],['100','','早于'],['1','101','超出']]) {
  rangeStart.value=start;rangeEnd.value=end;
  const count=sockets.length;
  const filesBefore=uploads;
  await elements.get('start').onclick();
  assert.equal(sockets.length,count);
  assert.equal(uploads, filesBefore);
  assert.match(elements.get('status').textContent,new RegExp(message));
  assert.equal(rangeStart.disabled||rangeEnd.disabled,false);
}
rangeStart.value='0:00:58.125';rangeEnd.value='';
const [openEnd, toEnd] = await startMeeting('file');
assert.equal(toEnd.file_start_seconds,58.125);
assert.equal('file_end_seconds' in toEnd,false);
await frame(openEnd,{...readyRange,file_end_seconds:100,protocol:'second'});
media.currentTime=59.375;
await elements.get('stop').onclick();
assert.deepEqual(last(openEnd),{type:'file_next',release_seconds:59.375,final:true});
await frame(openEnd,{type:'file_done'});
await frame(openEnd,{type:'result',event:'end_of_audio',end_seconds:1.25,source_end_seconds:59.375});
rangeStart.value='';rangeEnd.value='1:36';
const [openStart, until] = await startMeeting('file');
assert.equal('file_start_seconds' in until,false);
assert.equal(until.file_end_seconds,96);
await frame(openStart,{...readyRange,file_start_seconds:0,protocol:'first'});
await elements.get('stop').onclick();
await frame(openStart,{type:'file_done'});
await frame(openStart,{type:'file_end_empty',heard_seconds:0,source_heard_seconds:0});
assert.match(elements.get('status').textContent,/尚未播放音频/);
assert.equal(rangeEnd.disabled,false);

// An old server may ignore unknown start fields. Do not release the prefix
// under that server: only a matching ready handshake enables playback.
rangeStart.value='58';rangeEnd.value='96';
const [oldServer] = await startMeeting('file');
await frame(oldServer,{type:'ready',chunk_seconds:.5,protocol:'second'});
assert.equal(oldServer.frames.length,1);
assert.equal(oldServer.readyState,3);
assert.equal(media.paused,true);
assert.match(elements.get('status').textContent,/没有确认/);
// A new file clears the previous file's interval.
elements.get('file').onchange();
assert.equal(rangeStart.value,'');
assert.equal(rangeEnd.value,'');
elements.get('file').files = [];
await elements.get('start').onclick();
assert.equal(picker.disabled, false, 'unlocked after a failed file start');
console.log('web UI contract checks passed');
