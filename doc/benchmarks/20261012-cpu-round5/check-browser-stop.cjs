const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('src/FunAsrNet.Server/wwwroot/app.js', 'utf8');
const release = source.slice(source.indexOf('async function releaseAudio'), source.indexOf('/**', source.indexOf('async function releaseAudio')));
const stop = source.slice(source.indexOf('async function stop()'), source.indexOf('$("mic-start").onclick'));
const log = [];
let releaseDevice;
const closed = new Promise(resolve => releaseDevice = resolve);
const session = {
  ws: {readyState: 1, send: value => log.push(JSON.parse(value).event)},
  ctx: {state: 'running', close: () => {log.push('device-close'); return closed;}},
  node: {port: {postMessage: () => log.push('flush-request')}, disconnect() {}},
};
const context = vm.createContext({current: session, performance: {now: () => 1000}, buttons() {}, status() {}, WebSocket: {OPEN: 1}, clearTimeout() {}, setTimeout: () => 1, fail: (_s, message) => {throw Error(message);}});
vm.runInContext(release + '\n' + stop, context);
(async () => {
  let finished = false;
  const operation = context.stop().then(() => finished = true);
  assert.deepEqual(log, ['flush-request']);
  log.push('pcm-tail');
  session.flushed();
  await Promise.resolve(); await Promise.resolve();
  assert.deepEqual(log, ['flush-request', 'pcm-tail', 'end', 'device-close']);
  assert.equal(session.stopTime, 1000);
  assert.equal(finished, false);
  releaseDevice(); await operation;
  assert.equal(finished, true);
  console.log('PASS tail PCM -> end -> device close; stop clock starts at click; cleanup awaited.');
})().catch(error => {console.error(error); process.exitCode = 1;});
