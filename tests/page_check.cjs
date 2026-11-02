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
    addEventListener() {}, querySelectorAll: () => [], querySelector: () => null,
    insertAdjacentHTML() {}, showModal() { this.open = true; }, close() { this.open = false; }, scrollTo() {},
  };
}
const els = {};
const context = {
  document: { getElementById: id => (els[id] = els[id] || element(id)) },
  window: { addEventListener() {} },
  localStorage: { getItem: () => null, setItem() {} },
  fetch: async () => { throw new Error('offline'); },
  FormData: class { delete() {} append() {} },
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
els.workflow.value = 'r2v';
run('syncImageInput()');
assert.deepStrictEqual(labels(), ['Image 1: main subject (optional)', 'Image 2: extra context (optional)', 'Image 3: another reference (optional)']);
assert(els.secondImageHint.textContent.includes('Every clip'));
assert.strictEqual(boxes[1].parts.input.value, 'two.png', 'a workflow with room for it keeps the image');

els.workflow.value = 'ti2v';
run('syncImageInput()');
assert.deepStrictEqual(labels(), ['Image 1: first frame (optional: without one it starts from text)', 'Image 2: extra context (optional)']);
assert(boxes[2].parts.input.disabled, 'a hidden box sends nothing');

els.workflow.value = 't2v';
boxes[1].parts.input.files = [];
run('syncImageInput()');
assert.deepStrictEqual(labels(), [], 'text alone has no image to ask for');
assert.strictEqual(boxes[1].parts.input.value, '', 'and no image waiting in a hidden box');
assert(els.secondImage.hidden);
els.workflow.value = '';
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
const film = { name: 'A_run_20261101_120000.mp4', run: 'run_20261101_120000', title: 'A <i>film</i>', created: '01/11 12:00', size_mb: 40, mtime: 1 };
context.film = film;
const card = run('filmHtml(film)');
assert(card.includes('preload="none"'), 'a page of films must not hold the browser\'s connections');
assert(!card.includes('<i>film'), 'film titles are escaped');
assert(card.includes('data-shots="run_20261101_120000"') && card.includes('data-up=') && card.includes('/download/output/'));
film.run = null;
const orphan = run('filmHtml(film)');
assert(!orphan.includes('data-shots') && !orphan.includes('poster='), 'a film whose run is gone has no shots to open');

// The list of runs waiting: a plan is rendered, a stopped run is continued.
run(`renderStopped([{ id: 'run_1', label: 'A <b>plan', detail: 'Plan ready, 8 shots', clips: 0, estimate: 1860 }])`);
const box = els.stoppedRuns;
assert.strictEqual(box.hidden, false);
assert(box.innerHTML.includes('A &lt;b&gt;plan') && box.innerHTML.includes('~31 min'));
assert(box.innerHTML.includes('data-shots="run_1"') && box.innerHTML.includes('data-continue="run_1"') && box.innerHTML.includes('>Render<'));
const written = box.innerHTML;
box.innerHTML = 'untouched';
run(`renderStopped([{ id: 'run_1', label: 'A <b>plan', detail: 'Plan ready, 8 shots', clips: 0, estimate: 1860 }])`);
assert.strictEqual(box.innerHTML, 'untouched', 'a poll must not replace buttons that have not changed');
assert(written.length > 0);
run(`renderStopped([{ id: 'run_1', label: 'x', detail: 'y', clips: 2, estimate: 60 }])`);
assert(box.innerHTML.includes('>Continue<'));
run('renderStopped([])');
assert.strictEqual(box.hidden, true);

// A stopped run must not read as failed, and an upload's status says what happened.
assert(script.indexOf('data.stopped)') < script.indexOf('Failed (exit'));
run(`renderUpload({ lines: [], running: false, started_at: '17:00', elapsed: 1, returncode: 1, stopped: true, queue: { items: [], held: false, results: [] } })`);
assert.strictEqual(els.uploadStatus.textContent, 'Upload stopped');
assert.strictEqual(els.uploadDot.className, 'dot');

// Queue: held jobs offer Resume, and every item can be removed.
const queue = element('q');
context.queue = queue;
run(`renderQueue(queue, { items: [{ id: 3, label: 'a <film>', detail: 'd', estimate: 600 }], results: [], held: true, current: null }, 0)`);
assert(queue.innerHTML.includes('data-resume') && queue.innerHTML.includes('data-remove="3"') && queue.innerHTML.includes('a &lt;film&gt;'));

console.log('all page checks passed');
