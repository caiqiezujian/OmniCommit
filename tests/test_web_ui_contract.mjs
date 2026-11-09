// Browser-independent contract checks for file release and speaker corrections.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(here, '../omnicommit/web/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
assert.doesNotMatch(html, /id="token"|Authorization|访问口令/);
function element() {
  return {value:'', checked:false, hidden:false, disabled:false, textContent:'', dataset:{},
    classList:{toggle(){}}, append(){}, replaceChildren(){}, click(){}, pause(){this.paused=true;},
    currentTime:0, paused:false, files:[]};
}
const elements = new Map();
const tabs = ['microphone', 'system', 'file'].map(mode => ({...element(),dataset:{mode}}));
const document = {getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);},
  querySelectorAll(){return tabs;}, createElement(){return element();}, createTextNode(s){return s;}};
const sent=[];
const context = vm.createContext({document, console, WebSocket:{OPEN:1},
  URL:{createObjectURL(){return 'blob:x';}, revokeObjectURL(){}}, Blob, Int16Array,
  Float32Array, Math, Number, JSON, setTimeout, clearTimeout, setInterval, clearInterval,
  prompt(){return null;}});
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
console.log('web UI contract checks passed');
