// The real page, as Flask rendered it, with a stub DOM: its script runs from top to
// bottom, then the parts that build HTML get called with awkward data.
//   node tests/page_check.cjs page.html  (test_panel.py saves that page and runs this)
const fs = require('fs');
const assert = require('assert');
const vm = require('vm');

const html = fs.readFileSync(process.argv[2], 'utf8');
const script = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).find(s => s.includes('async function poll'));
assert(script, 'the page has its script');

const status0 = () => html.slice(html.indexOf('id="runCard"'), html.indexOf('id="optionsCard"'));
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
  document: { getElementById: id => (els[id] = els[id] || element(id)), querySelectorAll: () => [], documentElement: { style: { setProperty() {} } }, body: element('body'), addEventListener() {}, hidden: false },
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
assert(!run(`remembered({ name: 'image_5_role', type: 'select-one' })`));

// One box per image, built as a workflow needs them. The page's boxes, without a browser to parse them.
function imageBox(number) {
  const hiddenClass = () => ({ toggle(name, on) { this.on = on; } });
  // As in a browser, clearing an input's value empties its files.
  const input = { name: number === 1 ? 'image' : `image_${number}`, files: [], disabled: false, text: '',
    get value() { return this.text; }, set value(text) { this.text = text; if (!text) this.files = []; } };
  const parts = { input, img: { src: '', classList: hiddenClass() }, button: { classList: hiddenClass() }, label: { textContent: '' },
    select: { disabled: true, value: 'view', classList: hiddenClass() } };
  return { hidden: false, parts, querySelector: tag => parts[tag], classList: hiddenClass() };
}
const boxes = [];
const holder = context.document.getElementById('imageBoxes');
holder.querySelectorAll = () => boxes;
holder.insertAdjacentHTML = (_, html) => {
  assert(/type="file"/.test(html) && /<img class="preview hidden"/.test(html) && /data-clear/.test(html));
  assert(/<div class="image-line">[\s\S]*type="file"[\s\S]*data-clear[\s\S]*<\/div>/.test(html), 'the image and what it is share one line');
  const number = boxes.length + 1;
  assert.strictEqual(html.includes(`name="image_${number}_role"`), number > 1, 'every image after the first has a role');
  if (number > 1) assert(/value="view">Same character, new view<[\s\S]*value="character">Another character<[\s\S]*value="place">Environment/.test(html));
  boxes.push(imageBox(boxes.length + 1));
};
const labels = () => boxes.filter(box => !box.hidden).map(box => box.parts.label.textContent);

context.URL = { createObjectURL: () => 'blob:x', revokeObjectURL() {} };
const choose = (index, file = {}) => {
  boxes[index].parts.input.files = [file];
  boxes[index].parts.input.value = 'chosen.png';
  run(`showImage(imageBoxes()[${index}]); syncImageInput()`);
};
const pressPlus = () => {
  assert(!els.addImage.hidden, 'the + is there to press');
  run('imagesShown++; syncImageInput()');
};
run('syncImageInput()');
assert.deepStrictEqual(labels(), ['Image 1: first frame'], 'one box to start with');
assert(els.addImage.hidden, 'no + until that box holds an image');
assert(els.imagesHint.hidden, 'nothing to say about an image 2 nobody chose');
choose(0);
pressPlus();
assert.deepStrictEqual(labels(), ['Image 1: first frame', 'Image 2 (optional)'], 'the built-in workflow');
assert(els.addImage.hidden, 'the next + waits for this box');
choose(1);
assert.strictEqual(boxes[1].parts.input.value, 'chosen.png');
assert(els.addImage.hidden, 'and one input has room for image 2, no more');
assert(boxes[1].parts.select.classList.on === false && boxes[1].parts.select.disabled === false, 'image 2 chosen: its role shows and is sent');
run('syncImageInput()');
assert(!els.imagesHint.hidden);
assert(els.imagesHint.textContent.includes('never sees'), 'one input to fill: the LLM alone sees image 2');

run(`workflowTakes = { r2v: { min_images: 0, max_images: 3 }, t2v: { min_images: 0, max_images: 0 }, ti2v: { min_images: 0, max_images: 1 } }`);
els.videoWorkflow.value = 'r2v';
run('syncImageInput()');
assert.deepStrictEqual(labels(), ['Image 1: main character (optional)', 'Image 2 (optional)'], 'the boxes open stay open');
assert.strictEqual(boxes[1].parts.input.value, 'chosen.png', 'a workflow with room for it keeps the image');
pressPlus();
assert.deepStrictEqual(labels(), ['Image 1: main character (optional)', 'Image 2 (optional)', 'Image 3 (optional)']);
assert(els.imagesHint.textContent.includes('Every clip'));
assert(els.imagesHint.textContent.includes('takes 3 at most'), 'it says why the + stops');

els.videoWorkflow.value = 'ti2v';
run('syncImageInput()');
assert.deepStrictEqual(labels(), ['Image 1: first frame (optional: without one it starts from text)', 'Image 2 (optional)']);
assert(boxes[2].parts.input.disabled, 'a hidden box sends nothing');

els.videoWorkflow.value = 't2v';
boxes[1].parts.input.files = [];
run('syncImageInput()');
assert.deepStrictEqual(labels(), [], 'text alone has no image to ask for');
assert.strictEqual(boxes[1].parts.input.value, '', 'and no image waiting in a hidden box');
assert(els.imagesHint.hidden);

// Up to 20 boxes, one + at a time, and however many a workflow could take.
context.DataTransfer = class { constructor() { this.files = []; this.items = { add: file => this.files.push(file) }; } };
run('workflowTakes = { many: { min_images: 0, max_images: 30 }, minimax: { min_images: 0, max_images: 9 } }');
els.videoWorkflow.value = 'many';
run('imagesShown = 1; syncImageInput()');
for (let index = 0; index < 20; index++) {
  assert.strictEqual(labels().length, index + 1);
  choose(index, { number: index + 1 });
  if (index < 19) pressPlus();
}
assert(els.addImage.hidden, 'twenty is the most');
assert(!els.imagesHint.textContent.includes('at most'), 'the form\'s own limit, not the workflow\'s');

// Taking one out moves the rest up, each with its role, and leaves no gap.
boxes[3].parts.select.value = 'place';
boxes[4].parts.select.value = 'character';
run('removeImage(imageBoxes()[2]); syncImageInput()');
const chosen = () => boxes.filter(box => !box.hidden).map(box => (box.parts.input.files[0] || {}).number);
assert.deepStrictEqual(chosen().slice(0, 4), [1, 2, 4, 5]);
assert.strictEqual(chosen().length, 19);
assert.deepStrictEqual([boxes[2].parts.select.value, boxes[3].parts.select.value], ['place', 'character']);
assert.strictEqual(boxes[19].parts.input.value, '', 'the last box is emptied');
assert(!els.addImage.hidden, 'and there is room for one more');

// A workflow with less room keeps what fits, and an emptied list leaves one box.
els.videoWorkflow.value = 'minimax';
run('syncImageInput()');
assert.strictEqual(labels().length, 9);
assert(els.addImage.hidden && els.imagesHint.textContent.includes('takes 9 at most'));
assert(boxes[9].parts.input.disabled && !boxes[9].parts.input.files.length, 'what no longer fits is not sent');
run('imagesShown = 1; for (const box of imageBoxes()) clearImage(box); syncImageInput()');
assert.strictEqual(labels().length, 1);
assert.strictEqual(boxes[0].parts.button.classList.on, true, 'an empty first box has nothing to remove');

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

// One ETA for the run, on its bar: the status line and the step line don't repeat it.
assert(!script.includes('totalEta(') && !script.includes('Original ETA') && !script.includes('Adjusted ETA'));
assert(!/stageTime'\)\.textContent = [^;]*stepEta/.test(script));
run(`renderEtaDetail({ changes: [], original_total: 600, adjusted_total: 600 })`);
assert.strictEqual(els.etaDetail.hidden, true, 'on track: nothing to explain');
run(`renderEtaDetail({ changes: [{ text: 'Clip <b>3</b> was slow', seconds: 120 }], original_total: 600, adjusted_total: 720 })`);
assert(!els.etaDetail.hidden && els.etaDetail.innerHTML.includes('<details') && els.etaDetail.innerHTML.includes('+2 min') && !els.etaDetail.innerHTML.includes('<b>3'));

// Shorts in YouTube red; the phone address is folded under More, at the bottom of Status; no voice hint.
assert(/body\.on-shorts \{ --accent: #ff0033; \}/.test(html));
assert(/<details class="misc" id="misc">\s*<summary>More<\/summary>\s*<div class="phone">/.test(status0()));
assert(!html.includes('If both Gemini voice models fail'));

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

// How a run or upload ended is said once, under Finished, newest first; the status itself goes back to Idle.
run(`renderUpload({ lines: [{ n: 1, t: '17:00', text: 'x' }], running: false, started_at: '17:00', elapsed: 1, returncode: 1, stopped: true, queue: { items: [], held: false, results: [] } })`);
assert.strictEqual(els.uploadBox.hidden, true, 'an upload that ended has nothing to show here');
run(`renderFinished([{ label: 'A <b>run', outcome: 'done', at: '17:05' }, { label: 'B', outcome: 'failed', note: 'exit 1', at: '17:10' }],
  [{ label: 'C', outcome: 'stopped', at: '17:20' }])`);
assert.strictEqual(els.finished.hidden, false);
assert(els.finishedTitle.innerHTML.startsWith('Finished (3)') && els.finishedTitle.innerHTML.includes('1 failed'));
const order = ['Upload: C', 'B', 'A &lt;b&gt;run'].map(text => els.finishedList.innerHTML.indexOf(text));
assert(order.every(i => i >= 0) && order[0] < order[1] && order[1] < order[2], 'newest first, escaped');
run('renderFinished([], [])');
assert.strictEqual(els.finished.hidden, true);
assert(!script.includes("'Finished ✔'") && !script.includes('Upload finished') && !script.includes('Failed (exit'));

// The header: a short status on Run, and a bar under it per job.
const idle = { running: false, stopped: false, returncode: null, elapsed: 0, progress: {}, upload: { running: false, elapsed: 0 } };
context.idle = idle;
assert.strictEqual(run('runShort(idle, false)'), 'Idle');
assert.strictEqual(run('runShort({ ...idle, returncode: 0 }, false)'), 'Idle', 'finished is idle');
assert.strictEqual(run('runShort({ ...idle, returncode: 1 }, false)'), 'Idle');
assert.strictEqual(run('runShort(idle, true)'), 'Next job');
const rendering = { ...idle, running: true, elapsed: 400, progress: { stage: [2, 4, 'Clips'], detail: 'Clip 2 of 75: The harbour, 8s',
  eta: { total_left: 600, adjusted_total: 1000 } } };
context.rendering = rendering;
assert.strictEqual(run('runShort(rendering, false)'), 'Clip 2/75');
assert.strictEqual(run('runShort({ ...idle, running: true }, false)'), 'Starting');
run('renderStrips(idle)');
assert(els.runStrip.hidden && els.uploadStrip.hidden, 'nothing going, no bars');
const barOf = id => { const bar = context.document.getElementById(id); bar.parentElement = element('bar'); return bar; };
for (const id of ['runStripBar', 'uploadStripBar', 'runProgressBar', 'uploadProgressBar']) barOf(id);
context.document.getElementById('runCard').getBoundingClientRect = () => ({ top: 5000, bottom: 5100 });
run('renderStrips(rendering)');
assert.strictEqual(els.runStrip.hidden, false, 'away from Status, the footer has the bar');
assert.strictEqual(els.runProgress.hidden, false, 'and Status has it in full');
assert.strictEqual(els.runProgressBar.style.width, '40%');
assert(/^40% · ~10 min left, done around /.test(els.runProgressRight.textContent), 'Status has the one ETA, and when it ends');
assert.strictEqual(els.runProgressText.textContent, 'This run', 'the step line under it names the stage');
assert.strictEqual(els.runStripText.textContent, 'Clips');
assert.strictEqual(els.runStripRight.textContent, '40% · ~10 min left');
assert.strictEqual(els.runStripBar.style.width, '40%');
assert(els.runStripHint.textContent.startsWith('Clip 2 of 75'));
context.uploading = { ...idle, upload: { running: true, elapsed: 75, progress: { upload: [2, 3, 'A.mp4'] } } };
run('renderStrips(uploading)');
assert.strictEqual(els.uploadStripText.textContent, 'Uploading 2/3');
assert.strictEqual(els.uploadStripHint.textContent, 'A.mp4');
assert.strictEqual(els.uploadStripBar.style.width, '33%');
assert.strictEqual(els.uploadProgressBar.style.width, '33%');
assert.strictEqual(els.uploadProgressHint.textContent, 'A.mp4');
assert.strictEqual(els.uploadProgressRight.textContent, '33%', 'the status line above it has the time so far');
assert.strictEqual(els.uploadProgress.hidden, false);
context.document.getElementById('runCard').getBoundingClientRect = () => ({ top: 0, bottom: 100 });
run('syncStrips()');
assert(els.runStrip.hidden && els.uploadStrip.hidden, 'on Status the footer bars would only repeat it');
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
assert(/id="go">Generate</.test(html) && script.includes("'Add to queue'"), 'Generate, or Add to queue while something runs');
assert(/<a href="#videosCard">Videos<\/a>/.test(html) && !html.includes('jumpCount'), 'the footer has no count on Videos');
// Each heading holds its section, and the first is called Generation.
assert(!html.includes('What to make'));
assert(/<details class="section" id="formTopic" open>\s*<summary class="section-title">Generation</.test(html));
assert(/<details class="section" id="videoGeneration" open>\s*<summary class="section-title">Generation</.test(html));
assert(!/<div class="section-title">(Voice|Publish|How)/.test(html));
// On Generate the footer keeps the button and the estimate.
assert(html.indexOf('id="goStick"') > html.indexOf('id="stick"') && html.indexOf('id="goStick"') < html.indexOf('class="dashboard"'));
assert(html.indexOf('id="stop"') < html.indexOf('id="optionsCard"'), 'Stop stays with what is running');

// Whichever mode is on is marked, for the eye and for screen readers, and the other isn't.
const marks = () => {
  const made = {};
  for (const id of ['switchShorts', 'switchVideo']) {
    const button = context.document.getElementById(id);
    button.active = null;
    button.classList = { toggle: (name, on) => { if (name === 'active') button.active = on; } };
    button.setAttribute = (name, value) => { if (name === 'aria-selected') button.selected = value; };
    made[id] = button;
  }
  return made;
};
const buttons = marks();
run("setPipeline('video', false)");
assert(buttons.switchVideo.active === true && buttons.switchShorts.active === false && buttons.switchVideo.selected === true);
run("setPipeline('shorts', false)");
assert(buttons.switchShorts.active === true && buttons.switchVideo.active === false && buttons.switchShorts.selected === true);
assert(/\.switch button\.active \{ background: var\(--accent\)/.test(html) && /body\.on-video \{ --accent:/.test(html));

// The switch is the top of the page, with the title gone and two icons beside it.
assert(html.indexOf('id="switchShorts"') < html.indexOf('id="stick"') && !/<h1>/.test(html.slice(0, html.indexOf('<script>'))));
assert(/id="shutdown" aria-label="Shut down"[^>]*><svg/.test(html));
// Queue and log live in Status, the queue first and the log folded away, and the footer no longer goes to them.
const status = html.slice(html.indexOf('id="runCard"'), html.indexOf('id="optionsCard"'));
assert(status.indexOf('id="runQueue"') > 0 && status.indexOf('id="runQueue"') < status.indexOf('id="logBox"') && status.indexOf('id="logBox"') < status.indexOf('id="log"'));
assert(/<details class="log-box" id="logBox">\s*<summary>Expand to see/.test(status) && !/id="logBox" open/.test(html));
assert(/<a href="#optionsCard">Generate<\/a>/.test(html) && !html.includes('href="#queueCard"') && !html.includes('href="#logCard"'));

// The footer bars show away from Status, which has its own.
run('stripActive.run = true; stripActive.upload = false; syncStrips()');
assert.strictEqual(els.runStrip.hidden, true, 'on Status');
context.document.getElementById('runCard').getBoundingClientRect = () => ({ top: 5000, bottom: 5100 });
run('syncStrips()');
assert.strictEqual(els.runStrip.hidden, false, 'scrolled away from Status');
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
run(`renderQueue(queue, { items: [], results: [{ label: 'x', outcome: 'done', at: '17:00' }, { label: 'y', outcome: 'done', at: '17:01' }], held: false, current: null }, 0)`);
assert.strictEqual(queue.hidden, true, 'what finished is under Finished, not in the queue');
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
  assert.strictEqual(els.jumpRun.textContent, 'Idle', 'it works again straight after a stuck one');
  assert.strictEqual(els.statusText.textContent, 'Idle', 'a finished run leaves the status idle');
  assert.strictEqual(els.go.textContent, 'Generate');
  context.fetch = answer({ ...quiet, server_started: 1, queue: { ...quiet.queue, items: [{ id: 1, label: 'x', estimate: 60 }] } });
  await run('poll()');
  assert.strictEqual(els.go.textContent, 'Add to queue');
  assert.strictEqual(els.goStick.textContent, 'Add to queue');

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
