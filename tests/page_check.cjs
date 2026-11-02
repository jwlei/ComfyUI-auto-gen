// The real page, as Flask rendered it, with a stub DOM: its script runs from top to
// bottom, then the parts that build HTML get called with awkward data.
//   node tests/page_check.cjs page.html  (test_panel.py saves that page and runs this)
const fs = require('fs');
const assert = require('assert');
const vm = require('vm');

const html = fs.readFileSync(process.argv[2], 'utf8');
const script = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).find(s => s.includes('async function poll'));
assert(script, 'the page has its script');

// Every id the script reaches for has to be on the page.
const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]));
for (const [, id] of script.matchAll(/\$\('([A-Za-z]+)'\)/g)) assert(ids.has(id), `#${id} is on the page`);

function element(id) {
  return {
    id, dataset: {}, style: {}, hidden: false, innerHTML: '', textContent: '', value: '', className: '', elements: [],
    classList: { toggle() {}, add() {}, remove() {} },
    addEventListener() {}, querySelectorAll: selector => (selector === 'input' ? [element('low'), element('high')] : []), querySelector: () => element('part'), setAttribute() {},
    getBoundingClientRect: () => ({ top: 0, bottom: 100 }), appendChild() {}, offsetHeight: 90, insertAdjacentHTML() {}, showModal() { this.open = true; }, close() { this.open = false; }, scrollTo() {},
  };
}
const els = {};
const context = {
  document: { getElementById: id => (els[id] = els[id] || element(id)), documentElement: { style: { setProperty() {} } }, body: element('body'), addEventListener() {}, hidden: false },
  history: { replaceState() {} },
  window: { addEventListener() {}, innerHeight: 800 },
  localStorage: { getItem: () => null, setItem() {} },
  fetch: async () => { throw new Error('offline'); },
  FormData: class { delete() {} append() {} },
  location: { reload() { context.reloads++; } }, reloads: 0, AbortController,
  URL, setTimeout: () => 0, clearTimeout() {}, setInterval: () => 0, requestAnimationFrame: () => 0, cancelAnimationFrame() {},
  confirm: () => true, alert() {}, console, Date, Math, JSON, Set, Object, String,
};
vm.createContext(context);
vm.runInContext(script, context);
const run = code => vm.runInContext(code, context);

// Remembered options never touch a file input, which throws if you set its value.
for (const name of ['image', 'image_2', 'image_3']) assert(!run(`remembered({ name: '${name}', type: 'file' })`), name);
assert(run(`remembered({ name: 'llm', type: 'select-one' })`));
assert(!run(`remembered({ name: 'image_2_role', type: 'select-one' })`), 'each film starts from the default role');

// One box per image, built as a workflow needs them. The page's boxes, without a browser to parse them.
function imageBox(number) {
  const hiddenClass = () => ({ toggle(name, on) { this.on = on; } });
  const input = { name: number === 1 ? 'image' : `image_${number}`, value: '', files: [], disabled: false };
  const parts = { input, img: { src: '', classList: hiddenClass() }, button: { classList: hiddenClass() }, label: { textContent: '' } };
  return { hidden: false, parts, querySelector: tag => parts[tag] };
}
const boxes = [];
const holder = context.document.getElementById('imageBoxes');
holder.querySelectorAll = () => boxes;
holder.insertAdjacentHTML = (_, html) => {
  assert(/type="file"/.test(html) && /<img class="preview hidden"/.test(html) && /data-clear/.test(html));
  boxes.push(imageBox(boxes.length + 1));
};
const labels = () => boxes.filter(box => !box.hidden).map(box => box.parts.label.textContent);

run('syncImageInput()');
assert.deepStrictEqual(labels(), ['Image 1: first frame', 'Image 2: extra context (optional)'], 'the built-in workflow');
assert(els.secondImage.hidden, 'nothing to say about an image 2 nobody chose');
boxes[1].parts.input.files = [{}];
boxes[1].parts.input.value = 'two.png';
run('syncImageInput()');
assert(!els.secondImage.hidden, 'image 2 chosen: say what it is');
assert(els.secondImageHint.textContent.includes('never sees'), 'one input to fill: the LLM alone sees image 2');

run(`workflowTakes = { r2v: { min_images: 0, max_images: 3 }, t2v: { min_images: 0, max_images: 0 }, ti2v: { min_images: 0, max_images: 1 } }`);
els.videoWorkflow.value = 'r2v';
run('syncImageInput()');
assert.deepStrictEqual(labels(), ['Image 1: main subject (optional)', 'Image 2: extra context (optional)', 'Image 3: another reference (optional)']);
assert(els.secondImageHint.textContent.includes('Every clip'));
assert.strictEqual(boxes[1].parts.input.value, 'two.png', 'a workflow with room for it keeps the image');

els.videoWorkflow.value = 'ti2v';
run('syncImageInput()');
assert.deepStrictEqual(labels(), ['Image 1: first frame (optional: without one it starts from text)', 'Image 2: extra context (optional)']);
assert(boxes[2].parts.input.disabled, 'a hidden box sends nothing');

els.videoWorkflow.value = 't2v';
boxes[1].parts.input.files = [];
run('syncImageInput()');
assert.deepStrictEqual(labels(), [], 'text alone has no image to ask for');
assert.strictEqual(boxes[1].parts.input.value, '', 'and no image waiting in a hidden box');
assert(els.secondImage.hidden);
els.videoWorkflow.value = '';
run('workflowTakes = {}; syncImageInput()');

// Resolution presets: both shapes from 360p up to the workflow's own size, and that's the highest.
for (const size of ['640x352', '864x480', '960x544', '1280x736', '352x640', '480x864', '544x960', '736x1280']) {
  assert(html.includes(`<option value="${size}">`), size);
}
assert.strictEqual([...html.matchAll(/<option value="(\d+)x(\d+)">/g)].filter(m => +m[1] * +m[2] > 1280 * 736).length, 0);
els.sizePreset = element('sizePreset');
els.size = element('size');
els.sizePreset.value = '352x640';
run('syncSize()');
assert.strictEqual(els.size.value, '352x640', 'the preset is what the form sends');
assert(els.size.hidden);
els.sizePreset.value = 'custom';
run('syncSize()');
assert(!els.size.hidden, 'a custom size has somewhere to go');
assert.strictEqual(els.size.value, '352x640', 'and starts from the last one');
els.sizePreset.value = '';
run('syncSize()');
assert.strictEqual(els.size.value, '', 'the default sends no size, so it follows the image');

// The film length range comes from the server, and goes down to one clip.
assert(/id="duration"[^>]*min="5"[^>]*max="600"[^>]*step="5"/.test(html));
assert.strictEqual(run('clipCount(1)'), '1 clip');
assert.strictEqual(run('clipCount(75)'), '75 clips');

// Times.
assert.strictEqual(run('stepEta({state:"kept",seconds:0})'), 'kept');
assert.strictEqual(run('stepEta({state:"done",seconds:65})'), '1:05');
assert.strictEqual(run('stepEta({state:"next",seconds:1860})'), '~31 min');
assert.strictEqual(run('stepEta({state:"now",seconds:0})'), 'any moment');
assert.strictEqual(run('fmtEta(9300)'), '~2h 35m');

// A film's shots: the text people and models write can hold anything.
const shots = {
  id: 'run_20261101_120000', title: 'A <b>storm</b>', brief: 'say "hi"', start: 'chain', llm: 'claude', busy: false, finished: false,
  shots: [
    { title: 'One', seconds: 8, prompt: 'She says </textarea><script>alert(1)</script>', rendered: true },
    { title: 'Two', seconds: 7, prompt: 'Plain', rendered: false },
  ],
};
context.shots = shots;
const dialog = run('shotsHtml(shots)');
assert(!dialog.includes('<b>storm'), 'titles are escaped');
assert(!dialog.includes('<script>'), 'prompts are escaped');
assert(!dialog.includes('</textarea><script'), 'a prompt can\'t close its own box');
assert(dialog.includes('data-redo="1"'), 'a rendered shot can be rendered again');
assert(!dialog.includes('data-redo="2"'), 'one not rendered yet has nothing to redo');
assert(dialog.includes('preload="none"'), 'clips only load when played');
assert(dialog.includes('Render the 1 left'));
assert(dialog.includes('also renders every shot after it'));

shots.busy = true;
const busy = run('shotsHtml(shots)');
assert(busy.includes('readonly') && !busy.includes('data-redo') && !busy.includes('data-save') && !busy.includes('data-render'),
  'a run that is going can only be read');
shots.busy = false;

shots.start = 'image';
assert(run('shotsHtml(shots)').includes('leaves the others as they are'));
shots.finished = true;
assert(!run('shotsHtml(shots)').includes('data-render'), 'a finished film has nothing left to render');

// Films.
run("folder = 'video'");
const film = { name: 'A_run_20261101_120000.mp4', run: 'run_20261101_120000', title: 'A <i>film</i>', created: '01/11 12:00', size_mb: 40, mtime: 1 };
context.film = film;
const card = run('filmHtml(film)');
assert(card.includes('preload="none"'), 'a page of films must not hold the browser\'s connections');
assert(!card.includes('<i>film'), 'film titles are escaped');
assert(card.includes('data-shots="run_20261101_120000"') && !card.includes('data-up=') && card.includes('/download/video/'), 'a film can be downloaded, not uploaded');
film.run = null;
const orphan = run('filmHtml(film)');
assert(!orphan.includes('data-shots') && !orphan.includes('poster='), 'a film whose run is gone has no shots to open');

// The list of runs waiting: a plan is rendered, a stopped run is continued.
run(`renderStopped([{ id: 'run_1', pipeline: 'video', label: 'A <b>plan', detail: 'Plan ready, 8 shots', clips: 0, estimate: 1860 }])`);
const box = els.stoppedRuns;
assert.strictEqual(box.hidden, false);
assert(box.innerHTML.includes('A &lt;b&gt;plan') && box.innerHTML.includes('~31 min'));
assert(box.innerHTML.includes('data-shots="run_1"') && box.innerHTML.includes('data-continue="run_1"') && box.innerHTML.includes('>Render<'));
const written = box.innerHTML;
box.innerHTML = 'untouched';
run(`renderStopped([{ id: 'run_1', pipeline: 'video', label: 'A <b>plan', detail: 'Plan ready, 8 shots', clips: 0, estimate: 1860 }])`);
assert.strictEqual(box.innerHTML, 'untouched', 'a poll must not replace buttons that have not changed');
assert(written.length > 0);
run(`renderStopped([{ id: 'run_1', pipeline: 'video', label: 'x', detail: 'y', clips: 2, estimate: 60 }])`);
assert(box.innerHTML.includes('>Continue<'));
run('renderStopped([])');
assert.strictEqual(box.hidden, true);

// A stopped run must not read as failed, and an upload's status says what happened.
assert(script.indexOf('data.stopped)') < script.indexOf('Failed (exit'));
run(`renderUpload({ lines: [], running: false, started_at: '17:00', elapsed: 1, returncode: 1, stopped: true, queue: { items: [], held: false, results: [] } })`);
assert.strictEqual(els.uploadStatus.textContent, 'Upload stopped');
assert.strictEqual(els.uploadDot.className, 'dot');

// The header: a short status on Run, and a bar under it per job.
const idle = { running: false, stopped: false, returncode: null, elapsed: 0, progress: {}, upload: { running: false, elapsed: 0 } };
context.idle = idle;
assert.strictEqual(run('runShort(idle, false)'), 'Idle');
assert.strictEqual(run('runShort({ ...idle, returncode: 0 }, false)'), 'Done');
assert.strictEqual(run('runShort({ ...idle, returncode: 1 }, false)'), 'Failed');
assert.strictEqual(run('runShort({ ...idle, returncode: 1, stopped: true }, false)'), 'Stopped');
assert.strictEqual(run('runShort(idle, true)'), 'Next job');
const rendering = { ...idle, running: true, elapsed: 400, progress: { stage: [2, 4, 'Clips'], detail: 'Clip 2 of 75: The harbour, 8s',
  eta: { total_left: 600, adjusted_total: 1000 } } };
context.rendering = rendering;
assert.strictEqual(run('runShort(rendering, false)'), 'Clip 2/75');
assert.strictEqual(run('runShort({ ...idle, running: true }, false)'), 'Starting');
run('renderStrips(idle)');
assert(els.runStrip.hidden && els.uploadStrip.hidden, 'nothing going, no bars');
const barOf = job => { const bar = context.document.getElementById(`${job}StripBar`); bar.parentElement = element('bar'); return bar; };
barOf('run'); barOf('upload');
run('renderStrips(rendering)');
assert.strictEqual(els.runStrip.hidden, false);
assert.strictEqual(els.runStripText.textContent, 'Clips');
assert.strictEqual(els.runStripRight.textContent, '~10 min left');
assert.strictEqual(els.runStripBar.style.width, '40%');
assert(els.runStripHint.textContent.startsWith('Clip 2 of 75'));
context.uploading = { ...idle, upload: { running: true, elapsed: 75, progress: { upload: [2, 3, 'A.mp4'] } } };
run('renderStrips(uploading)');
assert.strictEqual(els.uploadStripText.textContent, 'Uploading 2/3');
assert.strictEqual(els.uploadStripHint.textContent, 'A.mp4');
assert.strictEqual(els.uploadStripBar.style.width, '33%');
assert.deepStrictEqual([...html.matchAll(/<a href="#(\w+)"/g)].map(m => m[1]), ['runCard', 'optionsCard', 'quotaCard', 'videosCard']);

// Nothing a video does can hold up the status: no video loads until you press play.
assert(!/preload="(metadata|auto)"/.test(script) && !/preload="(metadata|auto)"/.test(html), 'videos must not hold the browser\'s connections');

// Each side only previews its own videos.
const made = name => ({ name, created: '01/11 12:00', size_mb: 1, mtime: 1 });
run(`lastVideos = ${JSON.stringify({ output: [made('a_short.mp4')], uploaded: [made('uploaded_short.mp4')], video: [made('a_film.mp4')] })}`);
run("setPipeline('video', false)");
assert(els.videos.innerHTML.includes('a_film.mp4') && !els.videos.innerHTML.includes('short.mp4'), 'Video shows films only');
run("setPipeline('shorts', false)");
assert(els.videos.innerHTML.includes('a_short.mp4') && !els.videos.innerHTML.includes('a_film.mp4'), 'Shorts shows Shorts only');
assert(!els.videos.innerHTML.includes('uploaded_short.mp4'), 'and what is ready, until you open Uploaded');
run("setTab('uploaded')");
assert(els.videos.innerHTML.includes('uploaded_short.mp4') && !els.videos.innerHTML.includes('a_film.mp4'));
run("setPipeline('shorts', false)");

// Every link goes somewhere, in the page's own order, or the wrong section lights up.
const links = [...html.matchAll(/<a href="#(\w+)"/g)].map(m => m[1]);
for (const id of links) assert(ids.has(id), `#${id} is on the page`);
assert.deepStrictEqual(links.map(id => html.indexOf(`id="${id}"`)), links.map(id => html.indexOf(`id="${id}"`)).sort((a, b) => a - b));
// Status comes first, Generate is called just that, and it sits with the options.
assert(html.indexOf('id="runCard"') < html.indexOf('id="optionsCard"'));
assert(/id="go">Generate</.test(html) && !script.includes("'Add to queue'"));
assert(html.indexOf('id="stop"') < html.indexOf('id="optionsCard"'), 'Stop stays with what is running');

// The switch is the top of the page, with the title gone and two icons beside it.
assert(html.indexOf('id="switchShorts"') < html.indexOf('id="stick"') && !/<h1>/.test(html.slice(0, html.indexOf('<script>'))));
assert(/id="shutdown" aria-label="Shut down"[^>]*><svg/.test(html));
// Queue and log live in Status, the queue first and the log folded away, and the footer no longer goes to them.
const status = html.slice(html.indexOf('id="runCard"'), html.indexOf('id="optionsCard"'));
assert(status.indexOf('id="runQueue"') > 0 && status.indexOf('id="runQueue"') < status.indexOf('id="logBox"') && status.indexOf('id="logBox"') < status.indexOf('id="log"'));
assert(/<details class="log-box" id="logBox">\s*<summary>Expand to see/.test(status) && !/id="logBox" open/.test(html));
assert(/<a href="#optionsCard">Generate<\/a>/.test(html) && !html.includes('href="#queueCard"') && !html.includes('href="#logCard"'));

// The bars only show beside what they are about.
run('stripActive.run = true; stripActive.upload = false; syncStrips()');
assert.strictEqual(els.runStrip.hidden, false);
const realRect = els.runCard ? els.runCard.getBoundingClientRect : null;
context.document.getElementById('runCard').getBoundingClientRect = () => ({ top: 5000, bottom: 5100 });
run('syncStrips()');
assert.strictEqual(els.runStrip.hidden, true, 'scrolled away from status');
context.document.getElementById('runCard').getBoundingClientRect = () => ({ top: 0, bottom: 100 });
run('stripActive.run = false; syncStrips()');

// The switch: each side only shows what applies to it, and a job on the other still shows.
assert(/class="[^"]*only-shorts[^"]*" id="uploadQueue"/.test(html) && /class="[^"]*only-shorts[^"]*" id="uploadLog"/.test(html));
assert(/class="only-shorts" id="uploadAll"/.test(html) && /class="tabs only-shorts"/.test(html));
assert(/id="formVideo"/.test(html) && !/name="upload"[^>]*>[\s\S]*id="formVideo"[\s\S]*name="upload"/.test(html), 'Video has no upload option');

// Queue: held jobs offer Resume, and every item can be removed.
const queue = element('q');
queue.dataset.title = 'Runs';
context.queue = queue;
run(`renderQueue(queue, { items: [{ id: 3, label: 'a <film>', detail: 'd', estimate: 600 }], results: [], held: true, current: null }, 0)`);
assert(queue.innerHTML.includes('<b>Runs (1)</b>'));
assert(queue.innerHTML.includes('data-resume') && queue.innerHTML.includes('data-remove="3"') && queue.innerHTML.includes('a &lt;film&gt;'));

// The status recovers by itself.
(async () => {
  const quiet = { ...idle, lines: [], queue: { items: [], results: [], held: false, current: null }, stopped_runs: [], videos: { output: [], uploaded: [], video: [] },
    comfyui: { online: true }, quota: { groups: [], resets_in: 100 }, gpu: null, upload: { lines: [], running: false, elapsed: 0, queue: { items: [], results: [], held: false, current: null } } };
  const answer = (body, extra = {}) => async () => ({ json: async () => body, redirected: false, ...extra });

  // A poll that never answers is given up on, and the next one still gets through.
  context.fetch = (url, options) => new Promise((_, reject) => options.signal.addEventListener('abort', () => reject(new Error('aborted'))));
  context.setTimeout = fn => { fn(); return 0; };
  await run('poll()');
  context.setTimeout = () => 0;
  context.fetch = answer({ ...quiet, server_started: 1, returncode: 0 });
  await run('poll()');
  assert.strictEqual(els.jumpRun.textContent, 'Done', 'it works again straight after a stuck one');

  // Signed out: the page goes to the login rather than sit there stale.
  context.fetch = answer({}, { redirected: true });
  await run('poll()');
  assert.strictEqual(context.reloads, 1);

  // A new server means new code, unless that would lose what is being typed.
  context.reloads = 0;
  context.document.getElementById('topic').value = 'Half typed';
  context.fetch = answer({ ...quiet, server_started: 2 });
  await run('poll()');
  assert.strictEqual(context.reloads, 0, 'a typed topic is not lost');
  run("serverStarted = 1");
  context.document.getElementById('topic').value = '';
  await run('poll()');
  assert.strictEqual(context.reloads, 1);

  console.log('all page checks passed');
})().catch(e => { console.error(e); process.exit(1); });
