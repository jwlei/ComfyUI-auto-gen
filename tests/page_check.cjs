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
    classList: { toggle() {}, add() {}, remove() {}, contains: () => false },
    addEventListener() {}, querySelectorAll: selector => (selector === 'input' ? [element('low'), element('high')] : []), querySelector: () => element('part'), setAttribute() {},
    getBoundingClientRect: () => ({ top: 0, bottom: 100 }), appendChild() {}, offsetHeight: 90, insertAdjacentHTML() {}, showModal() { this.open = true; }, close() { this.open = false; }, scrollTo() {},
    pause() {}, play() { return Promise.resolve(); }, removeAttribute(name) { delete this[name]; },
  };
}
const els = {};
const wide = { matches: false, addEventListener() {} };
const context = {
  document: { getElementById: id => (els[id] = els[id] || element(id)), querySelectorAll: () => [], documentElement: { style: { setProperty() {} } }, body: element('body'), addEventListener() {}, hidden: false },
  history: { replaceState() {} },
  window: { addEventListener() {}, innerHeight: 800, matchMedia: () => wide, scrollTo() {} },
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
const filmSizes = html.slice(html.indexOf('id="sizePreset"'), html.indexOf('id="size"'));
assert.strictEqual([...filmSizes.matchAll(/<option value="(\d+)x(\d+)">/g)].filter(m => +m[1] * +m[2] > 1280 * 736).length, 0);
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
assert(!html.includes('stageName') && !html.includes('stageTime') && !script.includes('Step ${n}'), 'the stepper says which step, with no line repeating it');
run(`renderEtaDetail({ changes: [], original_total: 600, adjusted_total: 600 })`);
assert.strictEqual(els.etaDetail.hidden, true, 'on track: nothing to explain');
run(`renderEtaDetail({ changes: [{ text: 'Clip <b>3</b> was slow', seconds: 120 }], original_total: 600, adjusted_total: 720 })`);
assert(!els.etaDetail.hidden && els.etaDetail.innerHTML.includes('<details') && els.etaDetail.innerHTML.includes('+2 min') && !els.etaDetail.innerHTML.includes('<b>3'));

// Shorts in YouTube red; the phone address is hidden at the bottom of Status until its header is clicked; no voice hint.
assert(/body\.on-shorts \{ --accent: #ff0033; \}/.test(html));
assert(/<div class="misc" id="misc" hidden>\s*<div class="phone">/.test(status0()));
assert(!/<summary>More<\/summary>/.test(html), 'no More dropdown');
assert(/id="statusHead"[^>]*aria-controls="misc"/.test(status0()));
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

// The Image side: a set's titles can hold anything, and its size is worked out as the pipeline does.
const sets = [{ id: 'image_20261102_120000', label: 'A <b>set', total: 3, size: '1024x1024', when: '02 Nov', model: 'krea2_<SFW>',
  images: [{ position: 1, title: '<i>Look', version: 5 }, { position: 3, title: 'Two', version: 6 }] }];
context.sets = sets;
const setHtml = run('imageSetHtml(sets[0])');
assert(!setHtml.includes('<b>set') && !setHtml.includes('<i>Look'), 'titles are escaped');
assert(setHtml.includes('/thumbnails/1?v=5') && setHtml.includes('/files/1?v=5') && setHtml.includes('loading="lazy"'), 'small pictures, that open full size');
assert(setHtml.includes('data-delete-set=') && setHtml.includes('data-shots=') && setHtml.includes('data-download-set='));
assert(setHtml.includes('krea2_&lt;SFW&gt; · 2 of 3'), 'the workflow by its short name, first');
assert(!setHtml.includes('target="_blank"') && setHtml.includes('data-view-position="3"'), 'a picture opens in the viewer, not a new tab');
const inRow = setHtml.slice(setHtml.indexOf('class="image-set-row'));
assert(inRow.includes('<img') && inRow.indexOf('<img') < inRow.indexOf('data-download-set') && inRow.indexOf('data-shots') < inRow.indexOf('data-delete-set'), 'the pictures and their buttons share one line');

// Download on a set takes each image as a file of its own; the viewer's takes just the one it shows.
const clicked = [];
context.document.createElement = () => ({ click() { clicked.push(this.href); }, remove() {} });
context.document.body.appendChild = () => {};
context.setTimeout = fn => { fn(); return 0; };
run("downloadImages('image_20261102_120000', [1, 3])");
assert.deepStrictEqual(clicked, ['/image-runs/image_20261102_120000/files/1?download=1', '/image-runs/image_20261102_120000/files/3?download=1'], 'no zip');
context.setTimeout = () => 0;
assert(/<dialog id="imageViewer"[\s\S]*?id="viewerImage"[\s\S]*?id="viewerDownload"[\s\S]*?id="viewerDelete"[\s\S]*?id="viewerClose"[\s\S]*?<\/dialog>/.test(html), 'the picture, then Download, Delete and Close under it');
assert(/dialog\.viewer::backdrop \{ background: rgba\(0, 0, 0, \.85\)/.test(html), 'over a dimmed page');
run('imageSets = sets');
run("openViewer('image_20261102_120000', 3)");
assert.strictEqual(els.imageViewer.open, true);
assert.strictEqual(els.viewerDownload.href, '/image-runs/image_20261102_120000/files/3?v=6&download=1');
assert.strictEqual(els.viewerTitle.textContent, 'Two · 2 of 2', 'which one of the set it is');
// A swipe, or an arrow key, goes through the set, and stops at its ends.
run("openViewer('image_20261102_120000', 1)");
assert.strictEqual(els.viewerTitle.textContent, '<i>Look · 1 of 2');
assert.strictEqual(run('stepViewer(-1)'), false, 'nothing before the first');
assert.strictEqual(run('stepViewer(1)'), true);
assert.strictEqual(run('viewing.position'), 3, 'the next one in the set, wherever it was numbered');
assert.strictEqual(els.viewerTitle.textContent, 'Two · 2 of 2');
assert.strictEqual(run('stepViewer(1)'), false, 'nothing after the last');
assert(/touchend[\s\S]*?Math\.abs\(dx\) > 50 && Math\.abs\(dx\) > Math\.abs\(dy\)/.test(script) && /ArrowRight/.test(script), 'swipes, and arrow keys on a desktop');
assert(/#viewerImage \{[^}]*touch-action: pan-y pinch-zoom/.test(html), 'a sideways swipe is the page\'s, not the browser\'s');
run('closeViewer()');
assert.strictEqual(els.imageViewer.open, false);
assert(/\.image-set-row \{ display: flex/.test(html) && /\.image-grid img \{ width: 56px/.test(html), 'and small enough to');
// An album: its cover, as tall as the strip of the rest above its buttons; one image stays on a line with its buttons.
assert(/<div class="image-set-row album">\s*<a [^>]*class="album-cover"[^>]*><img[^>]*\/thumbnails\/1\?v=5"><span class="album-count">2<\/span><\/a>\s*<div class="album-side">\s*<div class="album-strip"><a [^>]*data-view-position="3"[\s\S]*?<\/div>\s*<div class="image-set-actions">/.test(setHtml),
  'cover beside the strip, the strip above the buttons');
const single = run("imageSetHtml({ id: 'image_9', label: 'One', total: 1, size: '1x1', when: 'now', images: [{ position: 1, title: 'x', version: 1 }] })");
assert(single.includes('class="album-cover"') && !single.includes('album-count') && !single.includes('album-strip'), 'one image: the same picture box, no count, no strip');
assert(/class="icon-btn" data-download-set="image_9" title="Download" aria-label="Download"><svg/.test(single) && /class="icon-btn del" data-delete-set="image_9"[^>]*aria-label="Delete"><svg/.test(single), 'icons, named');
assert(/data-download-set="image_9"[^>]*>[\s\S]*?<\/button><button type="button" class="icon-btn" data-again-set="image_9"[^>]*aria-label="The same again"><svg/.test(single), 'the same again, second');
// It puts every option back as the set had it, the rest at the workflow's own default.
{
  const fields = {};
  const field = (name, type, extra = {}) => (fields[name] = { name, type, value: '', checked: false, ...extra });
  field('opt_up', 'checkbox'); field('opt_steps', 'number'); field('opt_lora_on', 'checkbox'); field('opt_lora_strength', 'number', { value: '1' });
  context.optBox2 = { dataset: {}, querySelector: selector => fields[/name="([^"]+)"/.exec(selector)[1]] || null };
  context.optDefs = [{ id: 'up', kind: 'switch', default: '0' }, { id: 'steps', kind: 'number', default: '20' }, { id: 'lora', kind: 'strength', default: '0' },
    { id: 'prompt_enhancer', kind: 'switch', default: '0', role: 'llm' }];
  run(`applyOptions(optBox2, optDefs, { up: '1', lora: '0.8', prompt_enhancer: '1' })`);
  assert(fields.opt_up.checked === true && fields.opt_steps.value === '20' && fields.opt_lora_on.checked === true && fields.opt_lora_strength.value === '0.8');
}
assert(/\.image-set-actions \{[^}]*justify-content: flex-end;[^}]*margin-top: auto;/.test(html) && /\.album-cover \{[^}]*width: 92px; height: 92px;/.test(html), 'the icons sit in the same place on every set');
assert(/\.image-set-row\.album \{ align-items: stretch; \}/.test(html) && /\.album-strip \{ display: flex;[^}]*overflow-x: auto/.test(html), 'the strip scrolls sideways');
assert(/\.album-cover img \{ position: absolute; inset: 0;/.test(html) && !/\.album-cover \{[^}]*min-height/.test(html), 'the cover is as tall as the strip and buttons, not its picture');
assert(/input\[type=file\]::file-selector-button/.test(html), 'choosing a file has a real button');
run('imageSets = sets; renderImageSets()');
const once = els.imageSets.innerHTML;
assert(once.includes('2 of 3'));
els.imageSets.innerHTML = 'untouched';
run('renderImageSets()');
assert.strictEqual(els.imageSets.innerHTML, 'untouched', 'a poll with nothing new reloads no pictures');
const imageRun = { id: 'image_1', is_image: true, title: 'T', brief: 'b', llm: 'claude', direct: false, size: '1024x1024', busy: false, finished: false,
  shots: [{ title: 'One', prompt: '</textarea><script>x</script>', rendered: true, version: 1 }, { title: 'Two', prompt: 'p', rendered: false, version: 0 }] };
context.imageRun = imageRun;
const imageDialog = run('shotsHtml(imageRun)');
assert(!imageDialog.includes('<script>'), 'prompts are escaped');
assert(imageDialog.includes('data-redo="1"') && !imageDialog.includes('data-redo="2"') && imageDialog.includes('Make the 1 left') && imageDialog.includes('data-delete-image="2"'));
// A set's prompts: a carousel to swipe through, one image to a page, its count, and no long hint; the buttons in the middle.
assert(/<div class="shots-list carousel" id="shotsCarousel">/.test(imageDialog) && imageDialog.includes('1 of 2 · swipe for the others'));
assert(!imageDialog.includes('Making an image again leaves') && imageDialog.includes('<div class="shots-foot centred">'));
assert(/\.shots-list\.carousel \{ display: flex;[^}]*scroll-snap-type: x mandatory/.test(html) && /\.shots-list\.carousel > \.shot \{ flex: 0 0 100%/.test(html));
assert(imageDialog.includes('<span class="finished">finished</span>') && !imageDialog.includes('>made<') && /\.shot-title \.finished \{ color: var\(--ok\); \}/.test(html), 'finished, in green');
// The seed, quietly on the right of Make again and Delete, grows up out of its chevron.
assert(/<div class="shot-actions">\s*<button type="button" data-redo="1">Make again<\/button><button type="button" data-delete-image="1">Delete<\/button>\s*<button type="button" class="seed-toggle" data-seed-toggle="1" aria-expanded="false"[^>]*>Seed<\/button>\s*<div class="seed-fold" id="seedFold_1" hidden>\s*<div class="seed-line">/.test(imageDialog));
assert(/\.seed-fold \{[^}]*bottom: calc\(100% \+ 6px\)/.test(html) && /\.seed-toggle \{ position: absolute; right: 0;/.test(html));
assert(/\$\('shotsDialog'\)\.addEventListener\('click', e => \{\s*if \(e\.target\.closest && \(e\.target\.closest\('\.seed-fold'\) \|\| e\.target\.closest\('\[data-seed-toggle\]'\)\)\) return;\s*closeSeedFolds\(\);/.test(script),
  'a tap anywhere else puts the seed away');
assert.strictEqual(run("runUrl('image_1')"), '/image-runs/image_1');
assert.strictEqual(run("runUrl('run_1')"), '/runs/run_1');
const picked = (aspect, megapixels, source) => {
  els.imageAspect.value = aspect;
  els.imageMegapixels.value = megapixels;
  run(`sourceSize = ${JSON.stringify(source)}`);
  return run('pickedSize().join("x")');
};
assert.strictEqual(picked('1:1', '1', null), '1024x1024');
assert.strictEqual(picked('16:9', '1', null), '1360x768');
assert.strictEqual(picked('image', '1', null), '1024x1024', 'nothing to match: square');
assert.strictEqual(picked('image', 'image', [800, 1200]), '800x1200', 'the image\'s own resolution');
assert.strictEqual(picked('image', '1', [800, 1200]), '832x1248', 'its shape at the pixel count asked for');
assert.strictEqual(picked('1:1', 'image', [800, 1200]), '976x976', 'its pixel count in the shape asked for');
run('sourceSize = null');
// Automatic picks as the server does, and a model's limits grey out what it doesn't make well.
run(`imageWorkflowList = [{ min_images: 0, max_images: 0, size: true, model: 'krea2' }, { min_images: 1, max_images: 1, size: true, model: 'qwen_image_edit' },
  { min_images: 0, max_images: 16, size: true, model: 'fibo2' }]`);
assert.deepStrictEqual(['0', '1', '2', '17'].map(n => run(`(automaticTakes(${n}) || {}).model`)), ['krea2', 'qwen_image_edit', 'fibo2', undefined],
  'made for it first, Fibo 2.0 when nothing else fits');
els.imageAspect.options = [{ value: '1920x1080' }, { value: '1024x1024' }, { value: '3840x2160' }, { value: '16:9' }];
els.imageMegapixels.options = [{ value: 'image' }, { value: '1' }, { value: '2' }, { value: '6' }];
run(`imageWorkflows = { 'fibo.json': imageWorkflowList[2], 'krea.json': imageWorkflowList[0] }`);
els.imageWorkflow.value = 'fibo.json';
const fiboLimit = run('syncImageLimits()');
assert.deepStrictEqual(els.imageAspect.options.map(o => o.disabled), [true, false, true, undefined], 'Fibo 2.0: no screen over 1.1 MP');
assert.deepStrictEqual(els.imageMegapixels.options.map(o => o.disabled), [undefined, false, true, true]);
assert(fiboLimit.includes('Fibo 2.0 makes images up to 1.1 MP'));
els.imageWorkflow.value = 'krea.json';
run('syncImageLimits()');
assert.deepStrictEqual(els.imageAspect.options.map(o => o.disabled), [false, false, true, undefined], 'Krea-2: 1080p yes, 4K no');
assert.deepStrictEqual(els.imageMegapixels.options.map(o => o.disabled), [undefined, false, false, true]);
els.imageWorkflow.value = '';
run('imageWorkflows = {}; imageWorkflowList = []');

// A workflow's own options: a checkbox says off too, what goes with a switch sits beside it, and a problem locks it.
context.options = [
  { id: 'group_2nd_pass', kind: 'switch', label: '2nd <b>pass', default: '1' },
  { id: 'group_up', kind: 'switch', label: 'Upscaler', default: '0' },
  { id: 'group_up.choice', kind: 'choice', label: 'Upscaler: choice', default: '4K', choices: ['1080p', '4K'], parent: 'group_up' },
  { id: 'face_lora_strength', kind: 'number', label: 'LoRA strength', default: '0.8', min: -2, max: 2, step: 0.05, parent: 'group_up' },
  { id: 'group_cc', kind: 'switch', label: 'Color correction', default: '0', problem: 'Needs <i>LayerColor' },
  { id: 'lora_k', kind: 'strength', label: 'LoRA: K', default: '0', min: -2, max: 2, step: 0.05, usual: '1' },
  { id: 'steps', kind: 'number', label: 'Steps', default: '8', min: 1, max: 150, step: 1 },
];
const optionsHtml = run('options.map(option => imageOptionHtml(option, options)).join("")');
assert(!optionsHtml.includes('<b>pass') && !optionsHtml.includes('<i>LayerColor'), 'labels and problems are escaped');
assert(/name="opt_group_2nd_pass" value="0"><input type="checkbox" name="opt_group_2nd_pass" value="1" checked/.test(optionsHtml), 'unticked still says so');
const upscaler = optionsHtml.slice(optionsHtml.indexOf('Upscaler'), optionsHtml.indexOf('Color correction'));
assert(upscaler.includes('name="opt_group_up.choice"') && upscaler.includes('<option selected>4K</option>') && upscaler.includes('name="opt_face_lora_strength"'),
  'what goes with a switch is beside it');
assert(/name="opt_group_cc" value="1" disabled/.test(optionsHtml) && optionsHtml.includes("Can't be changed"), "one that can't run is locked, saying why");
assert(/name="opt_lora_k_on" value="1">/.test(optionsHtml) && /name="opt_lora_k_strength"[^>]*value="1"/.test(optionsHtml), 'an unused LoRA offers its usual strength');
assert(/<label for="opt_steps">Steps<\/label>/.test(optionsHtml) && /name="opt_steps"[^>]*value="8"/.test(optionsHtml));
assert(!run("remembered({ name: 'opt_steps', type: 'number' })"), 'each workflow starts as it was saved');
for (const size of ['1920x1080', '2560x1440', '3840x2160', '1272x2772']) assert(html.includes(`<option value="${size}">`), size);
els.imageAspect.value = '1920x1080';
els.imageMegapixels.value = '1';
run('syncImageSize()');
assert(els.imageSizeHint.textContent.includes('Makes 1920x1080 exactly'), 'a screen is its exact size, whatever the grid');
assert(!els.imageSizeHint.textContent.includes('far above'));
els.imageAspect.value = '3840x2160';
run('syncImageSize()');
assert(els.imageSizeHint.textContent.includes('far above'), 'it says what asking for 4K costs');
run("setPipeline('image', false)");
assert.strictEqual(run('folder'), 'image');
assert.strictEqual(els.jumpResults.textContent, 'Images');
run("setPipeline('shorts', false)");
assert.strictEqual(els.jumpResults.textContent, 'Videos');
assert(/body\.on-image \{ --accent/.test(html));

// A stopped run reads top to bottom in its own box: title, details, then the buttons.
run(`renderStopped([{ id: 'image_1', pipeline: 'image', label: 'Wardrobe', detail: '1 of 2 images · 1376x768', when: '02 Nov 23:46', clips: 1, estimate: 190 }])`);
const stoppedBox = els.stoppedRuns.innerHTML;
assert(stoppedBox.includes('class="stopped-run"'), 'in a box of its own');
assert(stoppedBox.indexOf('Wardrobe') < stoppedBox.indexOf('Image · 1 of 2 images') && stoppedBox.indexOf('Image · 1 of 2 images') < stoppedBox.indexOf('class="stopped-actions"'), 'in that order');
assert(stoppedBox.includes('<div class="hint">02 Nov 23:46</div>') && stoppedBox.indexOf('Image · 1 of 2 images') < stoppedBox.indexOf('02 Nov 23:46') && stoppedBox.indexOf('02 Nov 23:46') < stoppedBox.indexOf('class="stopped-actions"'), 'the date on its own line, under the details');
assert(stoppedBox.indexOf('class="stopped-actions"') < stoppedBox.indexOf('data-shots') && stoppedBox.indexOf('data-shots') < stoppedBox.indexOf('data-continue') && stoppedBox.indexOf('data-continue') < stoppedBox.indexOf('data-discard'), 'the buttons last');
assert(/\.stopped-run \{[^}]*border: 1px solid/.test(html));
run('renderStopped([])');

// Each form starts with its workflow, above the line its sections start with, and there's no Check setup.
for (const form of ['formShorts', 'formVideo', 'formImage']) {
  const start = html.indexOf(`id="${form}"`), section = html.indexOf('<details class="section"', start);
  const top = html.indexOf('class="workflow-top"', start);
  assert(top > start && top < section, `${form}: Workflow first`);
  assert(/<label for="\w+">Workflow<\/label>/.test(html.slice(top, section)), `${form}: called Workflow`);
}
assert(!html.includes('first-child > .section:first-child'), 'every form has the line above Generation');
assert(!html.includes('id="check"') && !script.includes('/check/'), 'no Check setup');

// A phone shows one section at a time, from the footer, and the bars over it ease in and out.
for (const page of ['status', 'generate', 'quota', 'output']) {
  assert(new RegExp(`data-page="${page}"`).test(status0() + html), page);
  assert(html.includes(`body[data-page="${page}"] .dashboard > [data-page="${page}"]`), `${page} shows alone`);
}
run("showPage('status')");
assert.strictEqual(context.document.body.dataset.page, 'status');
run("showPage('generate')");
assert.strictEqual(context.document.body.dataset.page, 'generate');
assert(/\.stick \.strip\[hidden\] \{ display: block; max-height: 0; opacity: 0;/.test(html), 'a bar eases out rather than vanishing');
assert(/@keyframes page-in/.test(html) && /prefers-reduced-motion/.test(html), 'pages slide in, unless motion is turned down');

// Films.
run("folder = 'video'");
const film = { name: 'A_run_20261101_120000.mp4', run: 'run_20261101_120000', title: 'A <i>film</i>', created: '01/11 12:00', size_mb: 40, mtime: 1 };
context.film = film;
const card = run('filmHtml(film)');
assert(!card.includes('<video') && card.includes('loading="lazy"'), 'a page of films must not hold the browser\'s connections: a poster, playing only when asked');
assert(!card.includes('<i>film'), 'film titles are escaped');
assert(card.includes('data-shots="run_20261101_120000"') && !card.includes('data-up=') && card.includes('data-download-film='), 'a film can be downloaded, not uploaded');
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
run(`renderFinished([{ label: 'A <b>run', outcome: 'done', at: '17:05', pipeline: 'video' }, { label: 'B', outcome: 'failed', note: 'exit 1', at: '17:10', pipeline: 'image' }],
  [{ label: 'C', outcome: 'done', at: '17:20' }])`);
assert.strictEqual(els.finished.hidden, false);
assert.strictEqual(els.finishedToggle.hidden, false);
assert(els.finishedToggle.innerHTML.startsWith('Finished (3)') && els.finishedToggle.innerHTML.includes('1 failed'));
const order = ['>C<', '>B (exit 1)<', '>A &lt;b&gt;run<'].map(text => els.finishedList.innerHTML.indexOf(text));
assert(order.every(i => i >= 0) && order[0] < order[1] && order[1] < order[2], 'newest first, escaped');
assert(/data-goto="video"><span class="when">17:05<\/span><span class="result-mark">✔<\/span><span class="result-type">Video<\/span><span class="result-label">A &lt;b&gt;run/.test(els.finishedList.innerHTML), 'its type before its title');
assert(/\.finished \.results \.result \{ display: flex; align-items: center;/.test(html) && /\.result \+ \.result \{ border-top: 1px solid/.test(html)
  && /\.result-label \{[^}]*-webkit-line-clamp: 2/.test(html), 'centred beside a title of two lines at most, a rule between');
assert(/id="skip"[^>]*>Skip<\/button>/.test(html) && /id="stop"[^>]*>Stop<\/button>/.test(html) && /\.run-actions \{ display: flex; justify-content: center;/.test(html), 'Skip and Stop, small and centred');
assert(/data-goto="shorts" data-tab="uploaded">[\s\S]*?<span class="result-type">Upload</.test(els.finishedList.innerHTML), 'an upload goes to what was uploaded');
// One tap takes you to that side's output.
run("goToOutput('image')");
assert.strictEqual(run('pipeline'), 'image');
assert.strictEqual(context.document.body.dataset.page, 'output');
run("setPipeline('shorts', false)");
// Log on the left, Finished on the right, on one line: one open at a time.
assert(/<div class="status-toggles" id="statusToggles">\s*<button[^>]*id="logToggle"[^>]*>Log<\/button>\s*<button[^>]*id="finishedToggle"/.test(html), 'Log, then Finished');
assert(!html.includes('Expand to see the log'));
run("showStatusPanel('log')");
assert(els.logBox.open === true && els.finished.open === false);
run("showStatusPanel('finished')");
assert(els.logBox.open === false && els.finished.open === true, 'opening one closes the other');
run('showStatusPanel(null)');
assert(!els.logBox.open && !els.finished.open);
// What a set going now has made is there too, as it comes, newest first.
run(`renderFinished([{ label: 'Old set', outcome: 'done', at: '09:00', t: 100, pipeline: 'image' }], [],
  [{ label: 'Looks · Version 1', outcome: 'done', at: '11:20', t: 300, pipeline: 'image' }, { label: 'Looks · Version 2', outcome: 'done', at: '11:22', t: 400, pipeline: 'image' }])`);
assert(els.finishedToggle.innerHTML.startsWith('Finished (3)'));
const madeOrder = ['Version 2', 'Version 1', 'Old set'].map(text => els.finishedList.innerHTML.indexOf(text));
assert(madeOrder.every(i => i >= 0) && madeOrder[0] < madeOrder[1] && madeOrder[1] < madeOrder[2], 'each image as it is made, newest first');
// Only the last five.
context.many = Array.from({ length: 7 }, (_, i) => ({ label: `Job ${i}`, outcome: 'done', at: '10:00', t: i, pipeline: 'video' }));
run('renderFinished(many, [])');
assert(els.finishedToggle.innerHTML.startsWith('Finished (5)') && els.finishedList.innerHTML.includes('Job 6') && !els.finishedList.innerHTML.includes('Job 1<'));
run('renderFinished([], [])');
assert.strictEqual(els.finished.hidden, true);
assert.strictEqual(els.finishedToggle.hidden, true);
assert(!script.includes("'Finished ✔'") && !script.includes('Upload finished') && !script.includes('Failed (exit'));

// The header: a short status on Run, and a bar under it per job.
const idle = { running: false, stopped: false, returncode: null, elapsed: 0, progress: {}, upload: { running: false, elapsed: 0 } };
context.idle = idle;
// The footer's Status is just that, with its dot: what is being made shows in Status and the bars.
assert(/<a href="#runCard" data-page="status" class="run"><span class="dot" id="jumpDot"><\/span>Status<\/a>/.test(html) && !html.includes('jumpRun'));
const rendering = { ...idle, running: true, elapsed: 400, progress: { stage: [2, 4, 'Clips'], detail: 'Clip 2 of 75: The harbour, 8s',
  eta: { total_left: 600, adjusted_total: 1000 } } };
context.rendering = rendering;
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
assert.strictEqual(els.runProgressText.textContent, 'This run', 'the stepper under it names the stage');
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

// Folded, a footer bar is one line: the bar, how far along and when it ends. The rest opens when tapped.
assert(/id="runStrip" role="button"[^>]*>\s*<div class="strip-line"><div class="bar"><div id="runStripBar"><\/div><\/div><span id="runStripRight"><\/span><\/div>\s*<div class="strip-more">[\s\S]*?id="runStripText"[\s\S]*?id="runStripHint"[\s\S]*?id="runStripTotal"/.test(html));
assert(/^Done around \d/.test(els.runStripTotal.textContent), 'opened, it says when the run ends');
const classes = () => { const on = new Set(); return { add: c => on.add(c), remove: c => on.delete(c), contains: c => on.has(c), toggle(c, yes) { if (yes) on.add(c); else on.delete(c); } }; };
els.runStrip.classList = classes();
const timers = [];
context.setTimeout = (fn, ms) => { timers.push([fn, ms]); return timers.length; };
context.document.body.dataset.page = 'generate';
run("tapStrip('run')");
assert(els.runStrip.classList.contains('open'), 'a tap opens it');
assert.strictEqual(timers[timers.length - 1][1], 5000);
timers[timers.length - 1][0]();
assert(!els.runStrip.classList.contains('open'), 'it folds again by itself after five seconds');
assert.strictEqual(context.document.body.dataset.page, 'generate');
run("tapStrip('run')");
run("tapStrip('run')");
assert.strictEqual(context.document.body.dataset.page, 'status', 'a tap while open goes to Status');
assert(!els.runStrip.classList.contains('open'));
context.setTimeout = () => 0;
assert(/\.go-bar button \{[^}]*padding: 8px 14px; font-size: \.85rem;/.test(html), 'the footer button is the smaller one');

// A log keeps where you are reading: the upload log only follows when it's at its end, and old lines going off the
// top don't move what you're looking at.
assert(!script.includes("logWriter($('uploadLog'), () => true)"), 'the upload log does not always jump to its end');
context.fakeLog = { rows: [], scrollTop: 0,
  get scrollHeight() { return this.rows.length * 10; }, get children() { return this.rows; },
  get firstElementChild() { const rows = this.rows; return rows.length ? { remove: () => rows.shift() } : null; },
  set innerHTML(text) { this.rows = text.match(/class="ln/g) || []; },
  insertAdjacentHTML(_, text) { this.rows.push(...(text.match(/class="ln/g) || [])); } };
context.lines = (a, b) => Array.from({ length: b - a + 1 }, (_, i) => ({ n: a + i, t: '', text: 'x' }));
run('var writeFake = logWriter(fakeLog, () => false)');
run('writeFake(lines(1, 100))');
context.fakeLog.scrollTop = 800;
run('writeFake(lines(1, 100))');
assert.strictEqual(context.fakeLog.scrollTop, 800, 'nothing new, nothing moves');
run('writeFake(lines(51, 120))');
assert.strictEqual(context.fakeLog.rows.length, 70);
assert.strictEqual(context.fakeLog.scrollTop, 300, 'line 81 is still at the top of the view');
context.document.getElementById('runCard').getBoundingClientRect = () => ({ top: 0, bottom: 100 });
run('syncStrips()');
assert(els.runStrip.hidden && els.uploadStrip.hidden, 'on Status the footer bars would only repeat it');
assert.deepStrictEqual([...html.matchAll(/<a href="#(\w+)"/g)].map(m => m[1]), ['runCard', 'optionsCard', 'videosCard', 'quotaCard']);
assert(!html.includes('name="avatar_source"') && !html.includes('name="lang"'), 'Shorts always uses ComfyUI, and the voice says the language');

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
// Shorts have the same rows as films and image sets; a Short ready to go has an upload icon first.
run("setTab('output')");
assert(/<div class="image-set film" data-name="a_short\.mp4">[\s\S]*?class="icon-btn" data-up="a_short\.mp4"[^>]*aria-label="Upload to YouTube"><svg[\s\S]*?data-download-film="a_short\.mp4"/.test(els.videos.innerHTML));
run("setTab('uploaded')");
assert(!els.videos.innerHTML.includes('data-up='), 'nothing to upload once it is up');
run("setPipeline('shorts', false)");

// Every link goes somewhere. The footer's order is its own: a desktop lights up by where the sections are.
const links = [...html.matchAll(/<a href="#(\w+)"/g)].map(m => m[1]);
for (const id of links) assert(ids.has(id), `#${id} is on the page`);
// Deleting takes two taps, on every image and video delete button.
for (const asked of ["confirm('Delete ' + d.del", "confirm('Delete this image?", "confirm('Delete this set?", 'confirm(`Delete image ${d.deleteImage}']) {
  assert(!script.includes(asked), `${asked} is a second tap now`);
}
context.tapped = { textContent: 'Delete', innerHTML: 'Delete', dataset: {}, classList: classes(), style: {}, getBoundingClientRect: () => ({ width: 72 }) };
assert.strictEqual(run('secondTap(tapped)'), false, 'the first tap only asks');
assert(context.tapped.classList.contains('armed') && context.tapped.textContent === 'Confirm');
assert.strictEqual(context.tapped.style.width, '72px', 'at the size it was');
assert.strictEqual(run('secondTap(tapped)'), true, 'the second deletes');
assert.strictEqual(context.tapped.innerHTML, 'Delete');
run('secondTap(tapped)');
run('disarm(tapped)');
assert(!context.tapped.classList.contains('armed') && context.tapped.innerHTML === 'Delete', 'and left alone it goes back');
// An icon turns red with a tick instead: "Confirm" wouldn't fit, and it keeps its size too.
context.iconTapped = { innerHTML: '<svg>bin</svg>', dataset: {}, classList: classes(), style: {}, getBoundingClientRect: () => ({ width: 40 }) };
context.iconTapped.classList.add('icon-btn');
run('secondTap(iconTapped)');
assert(context.iconTapped.innerHTML.includes('m5 12 5 5 9-10') && context.iconTapped.style.width === '40px');
run('disarm(iconTapped)');
assert.strictEqual(context.iconTapped.innerHTML, '<svg>bin</svg>');
assert.strictEqual(context.tapped.style.width, '', 'and its own size again');

// Moving on in the queue: the one starting leaves it at once, sliding up, and Status shows it arriving; never a blank.
const queueEl = { dataset: { title: 'Runs', lane: 'run' }, hidden: false, innerHTML: '', nodes: [],
  querySelectorAll() { return this.nodes; } };
const queueTimers = [];
context.queueEl = queueEl;
context.setTimeout = (fn, ms) => { queueTimers.push([fn, ms]); return 1; };
context.twoQueued = { items: [{ id: 1, label: 'First', pipeline: 'video', detail: 'd' }, { id: 2, label: 'Second', pipeline: 'image', detail: 'd' }], held: false };
run('renderQueue(queueEl, twoQueued, 0)');
assert(queueEl.innerHTML.includes('data-id="1"') && queueEl.innerHTML.includes('data-id="2"'));
const first = { dataset: { id: '1' }, style: {}, offsetHeight: 40, classList: classes() };
queueEl.nodes = [first, { dataset: { id: '2' }, style: {}, offsetHeight: 40, classList: classes() }];
run('renderQueue(queueEl, twoQueued, 0, true)');
assert(first.classList.contains('leaving') && first.style.height === '40px', 'the next one slides out of the queue');
assert(queueEl.innerHTML.includes('data-id="1"'), 'drawn again only once it has gone');
queueEl.nodes = [];
queueTimers[queueTimers.length - 1][0]();
assert(!queueEl.innerHTML.includes('data-id="1"') && queueEl.innerHTML.includes('data-id="2"') && queueEl.innerHTML.includes('Runs (1)'));
context.setTimeout = () => 0;
run("showWork('Video: First')");
assert(els.currentWork.hidden === false && els.currentWork.textContent === 'Video: First');
assert(/nextUp \? `\$\{SIDE_NAMES\[queue\.items\[0\]\.pipeline\]/.test(script), 'between two jobs, the next one already shows');
assert(/#currentWork\.arrive \{ animation: work-arrive/.test(html) && /\.queue-item\.leaving \{[^}]*animation: queue-leave/.test(html));
assert(/dialog\.viewer \.viewer-actions > \* \{ flex: 1 1 0; min-width: 96px; height: 40px; margin: 0;/.test(html), 'Download, Delete and Close are one size');
assert(/\.image-set-actions button\.armed, \.acts button\.armed[^{]*\{\s*background: var\(--err\) !important;/.test(html), 'Confirm is red in Output too');
context.withLoras = { ...context.oneImage0 = { id: 'image_2', label: 'L', total: 1, size: '1x1', when: 'now', images: [] }, loras: ['KNPV3_1 (0.8)', '<b>x'] };
const loraHtml = run('imageSetHtml(withLoras)');
assert(/<\/div><div class="meta loras"[^>]*>LoRA: KNPV3_1 \(0\.8\), &lt;b&gt;x<\/div>/.test(loraHtml), 'LoRAs on a line of their own, escaped');
assert(!run('imageSetHtml(oneImage0)').includes('LoRA'), 'none, no line');
assert(run("filmHtml({ name: 'f.mp4', created: 'now', size_mb: 1, loras: ['a'] })").includes('>LoRA: a</div>'), 'films too');
// A run that stopped short says so under Stopped runs: a small word on its title's line, and why under a fold.
// Output is only what was made.
run(`renderStopped([{ id: 'run_1', pipeline: 'video', label: 'Jungle Flight', detail: 'Plan ready', when: 'now', estimate: 60, clips: 0,
  outcome: { outcome: 'not rendered', why: 'Asked to read\\n<b>x', at: 5 } }, { id: 'run_2', pipeline: 'video', label: 'Fine', detail: 'd', when: 'now', estimate: 1, clips: 0 }])`);
const stoppedHtml = els.stoppedRuns.innerHTML;
assert(/<div class="run-head"><div class="what">Jungle Flight<\/div><span class="run-tag">not rendered<\/span><\/div>/.test(stoppedHtml), 'on the same line as its title');
assert(/<details class="why"><summary>Why<\/summary><div class="why-text">Asked to read\n&lt;b&gt;x<\/div><\/details>/.test(stoppedHtml), 'why, escaped, under a fold');
assert.strictEqual((stoppedHtml.match(/run-tag/g) || []).length, 1, 'nothing to say for one you stopped yourself');
assert(!script.includes('failed_films') && !script.includes('filmsHtml'), 'Output lists what was made, nothing else');

context.oneImage = { id: 'image_1', label: 'One', total: 1, size: '1024x1024', when: 'now', images: [{ position: 1, title: 'x', version: 1 }] };
assert(!run('imageSetHtml(oneImage)').includes('1 of 1'), 'one image on its own says no "1 of 1"');

// The footer: tapping the page you're on again offers the same page on the other sides.
context.document.body.dataset.page = 'generate';
run("setPipeline('shorts', false)");
const genLink = { dataset: { page: 'generate' }, offsetWidth: 80, offsetLeft: 100, offsetTop: 40, setAttribute() {}, removeAttribute() {} };
context.genLink = genLink;
run('openSideMenu(genLink)');
assert.strictEqual(els.sideMenu.hidden, false);
assert(els.sideMenu.innerHTML.includes('data-side="video"') && els.sideMenu.innerHTML.includes('data-side="image"') && !els.sideMenu.innerHTML.includes('data-side="shorts"'),
  'the other two sides, not the one you are on');
assert(/\.side-menu \{[^}]*animation: menu-in/.test(html) && /@keyframes menu-in \{ from \{ opacity: 0; transform: translateY\(8px\) scale\(\.94\)/.test(html), 'grows and fades in, upwards');
assert(/if \(page === document\.body\.dataset\.page && SIDE_PAGES\.includes\(page\)\)/.test(script) && /const SIDE_PAGES = \['status', 'generate', 'output'\]/.test(script), 'Status, Generate and Output; not Quota');
assert(/\.switch button \{[^}]*padding: 4px 12px; font-size: \.78rem;/.test(html), 'the side switch at the top is small now');

// Each workflow's options come back as you last set them, and another workflow's don't spill over.
{
  const stored = {};
  context.localStorage = { getItem: key => stored[key] || null, setItem: (key, value) => { stored[key] = value; } };
  const steps = { name: 'opt_steps', type: 'number', value: '20', disabled: false };
  const upscale = { name: 'opt_up', type: 'checkbox', checked: false, disabled: false };
  const hiddenZero = { name: 'opt_up', type: 'hidden', value: '0' };
  const box = { dataset: { workflow: 'imageOptions|krea.json' }, querySelectorAll: () => [steps, hiddenZero, upscale] };
  context.optBox = box;
  steps.value = '12';
  upscale.checked = true;
  run('rememberOptions(optBox)');
  assert.deepStrictEqual(JSON.parse(stored['comfyui-auto-gen.workflow-options'])['imageOptions|krea.json'], { opt_steps: '12', opt_up: true });
  steps.value = '20';
  upscale.checked = false;
  run('restoreOptions(optBox)');
  assert(steps.value === '12' && upscale.checked === true, 'as you left them');
  box.dataset.workflow = 'imageOptions|qwen.json';
  steps.value = '20';
  run('restoreOptions(optBox)');
  assert.strictEqual(steps.value, '20', "another workflow's settings are its own");
  context.localStorage = { getItem: () => null, setItem() {} };
}
assert(/syncOptions\('imageOptions', 'imageOptionsBox', imageTakes\(\)\.options \|\| \[\], baseImageTakes\(\)\.name\)/.test(script), "Automatic's pick by its name");

// Why Generate was turned down shows beside the footer's button too, where it was pressed.
run("showFormError('This workflow makes images from text alone: leave the image out.')");
assert.strictEqual(els.goError.hidden, false);
assert.strictEqual(els.goError.textContent, els.formError.textContent);
run("showFormError('')");
assert(els.goError.hidden && els.formError.hidden);
assert(/<div class="go-error" id="goError" role="alert" hidden><\/div>/.test(html) && /\.go-error:not\(\[hidden\]\) \+ \.go-eta \{ display: none; \}/.test(html));
// From the tap until the panel has it: the buttons count up, and Status shows it on its way to the queue.
run('showSending(0.45)');
assert.strictEqual(els.goStick.textContent, 'Sending… 45%');
assert.strictEqual(els.queuePending.hidden, false);
assert.strictEqual(els.queuePendingBar.style.width, '45%');
run('showSending(1)');
assert(els.queuePendingText.textContent.endsWith('adding to the queue…'), 'sent, and being taken in');
assert(/if \(!submitting\) \$\('go'\)\.textContent/.test(script), 'a poll meanwhile leaves it saying so');
assert(/xhr\.upload\.onprogress/.test(script) && /<div class="queue-pending" id="queuePending" hidden>/.test(html));

// A film has the same again too, beside Download, while its run is kept.
// Films are laid out as image sets: the poster as the picture, which plays it, and the same icons, the same again second.
const filmRow = run("filmHtml({ name: 'f.mp4', created: 'now', size_mb: 1, run: 'run_1', model: 'wan' })");
assert(/<div class="image-set film" data-name="f\.mp4">[\s\S]*?<div class="image-set-row album">\s*<a [^>]*class="album-cover film-cover" data-play-film="f\.mp4"/.test(filmRow));
assert(/data-download-film="f\.mp4"[\s\S]*?<\/button><button type="button" class="icon-btn" data-again-film="run_1"[\s\S]*?data-shots="run_1"[\s\S]*?class="icon-btn del" data-del="f\.mp4"/.test(filmRow), 'download, the same again, shots, delete');
assert(!run("filmHtml({ name: 'f.mp4', created: 'now', size_mb: 1 })").includes('data-again-film'), 'not once its run is gone');
// It plays in the viewer, and a swipe goes to the next film.
run(`lastVideos = { output: [], uploaded: [], video: [{ name: 'a.mp4', title: 'A' }, { name: 'b.mp4', title: 'B' }] }; folder = 'video'`);
run("openFilm('a.mp4')");
assert(els.viewerVideo.hidden === false && els.viewerImage.hidden === true && els.viewerVideo.src === '/videos/video/a.mp4');
assert.strictEqual(els.viewerTitle.textContent, 'A · 1 of 2');
assert.strictEqual(run('stepViewer(1)'), true);
assert.strictEqual(els.viewerVideo.src, '/videos/video/b.mp4');
assert.strictEqual(run('stepViewer(1)'), false, 'nothing after the last');
run('closeViewer()');
run(`folder = 'output'`);

// The viewer never shows both: [hidden] wins over each one's own display.
assert(/#viewerImage\[hidden\], #viewerVideo\[hidden\] \{ display: none !important; \}/.test(html));
assert(/\.viewer-title \{[^}]*-webkit-line-clamp: 2/.test(html), 'a long title stays on two lines, the buttons in view');
// LoRAs on one line, cut short, the whole of it a tap away.
assert(/class="meta loras" role="button" tabindex="0" title="LoRA: a, b">LoRA: a, b<\/div>/.test(run("lorasLine(['a', 'b'])")));
assert(/\.meta\.loras \{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis;/.test(html) && /\.meta\.loras\.open \{ white-space: normal;/.test(html));

// Discarding a stopped run takes two taps, with no pop-up; deleting a whole set from Output has its own address.
assert(!script.includes('Discard this run? Its') && /else if \(d\.discard\) \{\s*\/\/[^\n]*\n\s*if \(!secondTap\(e\.target\)\) return;/.test(script));
assert(script.includes("stopJob(`/image-runs/${encodeURIComponent(d.deleteSet)}/delete`)"));

// An exact size with nothing in its box falls back on 1:1 at one megapixel.
els.imageAspect.value = 'custom';
els.imageSize.value = '';
run('sizeFallback()');
assert(els.imageAspect.value === '1:1' && els.imageMegapixels.value === '1');
els.imageAspect.value = 'custom';
els.imageSize.value = '1920x1080';
run('sizeFallback()');
assert.strictEqual(els.imageAspect.value, 'custom', 'a size typed in stays');
els.imageAspect.value = 'image';

assert(/\.status-queue:has\(> :not\(\[hidden\]\)\) \{[^}]*border-top: 1px solid var\(--line\)/.test(html), 'a line between Status and the runs');

const base = fs.readFileSync(require('path').join(__dirname, '..', 'panel', 'templates', 'base.html'), 'utf8');
assert(/maximum-scale=1, user-scalable=no/.test(base) && /html \{ overflow-x: clip; touch-action: manipulation; \}/.test(base), 'no zooming the page out from under the footer');
assert(/html \{ overflow-x: clip;/.test(base) && /input, select, textarea \{ max-width: 100%; \}/.test(base) && /\.row > \* \{ flex: 1; min-width: 0; \}/.test(base),
  'nothing makes a phone page wider than its screen, so the footer stays put');
assert(/\.check > span \{ min-width: 0; overflow-wrap: anywhere; \}/.test(html) && /\.option-choice \{[^}]*max-width: 100%/.test(html), 'options wrap rather than widen it');
assert(/JUMPS\.filter\(a => top\(a\) <= line\)\.sort\(/.test(script), 'by position, not the footer order');
// Status comes first, Generate is called just that, and it sits with the options.
assert(html.indexOf('id="runCard"') < html.indexOf('id="optionsCard"'));
assert(/id="go">Generate</.test(html) && script.includes("'Add to queue'"), 'Generate, or Add to queue while something runs');
assert(/<a href="#videosCard" data-page="output" id="jumpResults">Videos<\/a>/.test(html) && !html.includes('jumpCount'), 'the footer has no count on Videos');
// Each heading holds its section, and the first is called Generation.
assert(!html.includes('What to make'));
assert(/<details class="section" id="formTopic">\s*<summary class="section-title">Generation</.test(html));
assert(/<details class="section" id="videoGeneration">\s*<summary class="section-title">Generation</.test(html));
assert(!/<div class="section-title">(Voice|Publish|How)/.test(html));
// Every heading starts folded, nothing brings back how they were left, and changing side folds them again.
assert(!/<details class="section"[^>]* open/.test(html) && !script.includes('localStorage.setItem(SECTIONS_KEY'));
{
  const section = { open: true };
  const all = context.document.querySelectorAll;
  context.document.querySelectorAll = selector => (selector === 'details.section' ? [section] : []);
  run("setPipeline('shorts', false)");
  assert.strictEqual(section.open, true, 'pressing the side you are on folds nothing');
  run("setPipeline('video', false)");
  assert.strictEqual(section.open, false, 'to Video');
  section.open = true;
  run("setPipeline('shorts', false)");
  assert.strictEqual(section.open, false, 'and back');
  context.document.querySelectorAll = all;
}
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
assert(/\.switch button\.active \{[^}]*color: var\(--accent\)/.test(html) && /body\.on-video \{ --accent:/.test(html));

// The switch is the top of the page, with the title gone and two icons beside it.
assert(html.indexOf('id="switchShorts"') < html.indexOf('id="stick"') && !/<h1>/.test(html.slice(0, html.indexOf('<script>'))));
assert(/id="shutdown" aria-label="Shut down"[^>]*><svg/.test(html));
// Queue and log live in Status, the queue first, and the footer no longer goes to them.
const status = html.slice(html.indexOf('id="runCard"'), html.indexOf('id="optionsCard"'));
assert(status.indexOf('id="runQueue"') > 0 && status.indexOf('id="runQueue"') < status.indexOf('id="logBox"') && status.indexOf('id="logBox"') < status.indexOf('id="log"'));
assert(/<details class="log-box" id="logBox">\s*<summary>Log<\/summary>/.test(status) && !/id="logBox" open/.test(html), 'opened by the Log toggle above it');
assert(/<a href="#optionsCard" data-page="generate">Generate<\/a>/.test(html) && !html.includes('href="#queueCard"') && !html.includes('href="#logCard"'));
// The log is folded on a phone. On a desktop it is always open, with no heading to fold it by, beside the run at half the width.
assert.strictEqual(els.logBox.open, false, 'folded on a phone');
wide.matches = true;
run('syncLogBox()');
assert.strictEqual(els.logBox.open, true, 'open on a desktop');
wide.matches = false;
run('syncLogBox()');
assert.strictEqual(els.logBox.open, false, 'and folded again once the window is narrow');
assert(/\.log-box > summary \{ display: none; \}/.test(html));
assert(/\.run-card \{[^}]*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\)/.test(html));
assert(status.indexOf('id="logBox"') < status.indexOf('id="finished"'), 'on a phone it comes before Finished and More');

// The GPU line is right under the running line, on both.
assert(status.indexOf('id="statusText"') < status.indexOf('id="gpuLine"') && status.indexOf('id="gpuLine"') < status.indexOf('id="runProgress"'));

// The clip being rendered is said quietly in ComfyUI's box, which no longer names ComfyUI.
assert(!status.includes('<span>ComfyUI:</span>'));
context.atStage = (detail, comfy) => ({ running: true, pipeline: 'shorts', gpu: null, progress: { stage: [4, 7, 'Avatar clips'], detail },
  comfyui: comfy ? { active: true, connected: true, label: 'Generating video', value: 3, max: 8, elapsed: 65, queue: 1 } : {} });
run(`renderNow(atStage('Clip 2 of 5 - scene 3, 6s lipsync', true))`);
assert.strictEqual(els.comfyMeta.textContent, 'Clip 2 of 5 - scene 3, 6s lipsync · This clip: 1:05');
assert.strictEqual(els.stageDetail.textContent, '', 'and not said again above it');
run(`renderNow(atStage('Clip 2 of 5 - scene 3, 6s lipsync', false))`);
assert.strictEqual(els.stageDetail.textContent, 'Clip 2 of 5 - scene 3, 6s lipsync', 'no box up yet: it still says it');
run(`renderNow(atStage('Scene 4 of 9', true))`);
assert.strictEqual(els.stageDetail.textContent, 'Scene 4 of 9', 'only a clip goes in the box');
assert.strictEqual(els.comfyMeta.textContent, 'This clip: 1:05');

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
  assert.strictEqual(els.statusText.textContent, 'Idle', 'it works again straight after a stuck one, and a finished run leaves it idle');
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
