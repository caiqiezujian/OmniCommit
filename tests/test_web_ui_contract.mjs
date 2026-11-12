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

// Second uses the same transport metadata for UI grouping; the visible log and
// downloaded session must preserve the sparse model text, including its spaces.
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
  segments=[{segment_id:1,display_speaker:'spk1',language:'en',text:'Yes.'}];
  render();
`,context);
const transcript = elements.get('transcript');
const firstGroup = transcript.children[0], firstPiece = firstGroup.children[1];
vm.runInContext(`
  for (const seg of [
    {segment_id:2,display_speaker:'spk1',language:'en',text:' I will.'},
    {segment_id:3,display_speaker:'spk1',language:'zh',text:'可以。'},
    {segment_id:4,display_speaker:'spk2',language:'zh',text:'好。'}]) {
    segments.push(seg);
    appendTranscript($('transcript'), seg);
  }
`,context);
assert.equal(transcript.children.length,3);
assert.equal(transcript.children[0],firstGroup);
assert.equal(firstGroup.children[1],firstPiece);
assert.equal(firstGroup.children.slice(1).map(e=>e.textContent).join(''),'Yes. I will.');
assert.equal(firstGroup.children[2].dataset.segmentId,2);
assert.equal(vm.runInContext('transcriptText()',context),'甲 [en] Yes. I will.\n甲 [zh] 可以。\n乙 [zh] 好。\n');
context.prompt=()=> 'spk2';
firstGroup.children[2].onclick();
assert.equal(vm.runInContext('segments[1].display_speaker',context),'spk2');
assert.equal(transcript.children.length,4);
assert.equal(vm.runInContext('transcriptText()',context),'甲 [en] Yes.\n乙 [en]  I will.\n甲 [zh] 可以。\n乙 [zh] 好。\n');

// Start via the actual page handler. Stub only the browser capture/upload
// devices and transport so lock timing and the real JSON handshake are covered.
assert.match(html, /<label id="chunkSettings" for="chunkMs" hidden>/);
const picker = elements.get('chunkMs'), chunkSettings = document.getElementById('chunkSettings');
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
for (const service of [
  {protocol:'first',chunk_seconds:.8,chunk_ms_choices:[500,700,800,1000]},
  {protocol:'second',chunk_seconds:.601,chunk_ms_choices:[500,601,700,800,1000]},
]) {
  // A fresh page receives its capabilities and default from its own service.
  picker.replaceChildren();
  chunkSettings.hidden = true;
  context.stepService = service;
  vm.runInContext('acceptedChunkMs=null; identifyService(stepService)',context);
  assert.equal(chunkSettings.hidden, false);
  assert.equal(picker.value, String(Math.round(service.chunk_seconds * 1000)));
  assert.deepEqual(picker.children.map(option => option.textContent), service.chunk_ms_choices.map(ms => ms + ' ms'));
  assert.equal(elements.get('brand').textContent, service.protocol === 'first' ? 'OmniCommit' : 'OmniCommit Second');
  for (const [source, ms] of [['microphone',700], ['system',800], ['file',1000]]) {
    picker.value = String(ms);
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
    assert.equal(picker.value, String(ms));
    await ws.onmessage({data:JSON.stringify({type:'result',event:'end_of_audio',action:'WAIT',text:'<|WAIT|>'})});
    assert.equal(picker.disabled, false);
  }
}
elements.get('file').files = [];
await elements.get('start').onclick();
assert.equal(picker.disabled, false, 'unlocked after a failed file start');
console.log('web UI contract checks passed');
