// The instrument. The board edits the pattern, this plays it, and the two
// never wait for each other: vision can be slow and careful while the clock
// stays solid.

const COLOURS = ["pink", "yellow", "blue", "green"];
const LOOKAHEAD = 0.1;      // seconds of notes scheduled in advance
const TICK = 25;            // milliseconds between scheduling passes

// ---------------------------------------------------------------- the pattern

// What the board says, in player order. Replaced wholesale by the server; the
// clock only ever reads it, so a slow vision frame cannot stall playback.
let grid = Array.from({ length: 4 }, () => Array(16).fill("empty"));

function coloursInColumn(step) {
  // Rows carry nothing in drum mode; they only let you stack several balls on
  // the same sixteenth. So a column is the set of colours present in it.
  return [...new Set(grid.map((row) => row[step]).filter((c) => c !== "empty"))];
}

// ---------------------------------------------------------------- the voices

const audio = new (window.AudioContext || window.webkitAudioContext)();
const master = audio.createGain();
master.connect(audio.destination);

// One voice per colour. Each starts as a synthesised sound so the thing makes
// noise the moment you open it, and holds a recorded or uploaded buffer once
// you give it one.
const voices = Object.fromEntries(COLOURS.map((name) => [name, { buffer: null, source: "built in" }]));

function builtIn(name, when) {
  const g = audio.createGain();
  g.connect(master);
  if (name === "pink") {                       // kick
    const o = audio.createOscillator();
    o.frequency.setValueAtTime(160, when);
    o.frequency.exponentialRampToValueAtTime(48, when + 0.11);
    g.gain.setValueAtTime(1, when);
    g.gain.exponentialRampToValueAtTime(0.001, when + 0.3);
    o.connect(g); o.start(when); o.stop(when + 0.32);
  } else if (name === "yellow") {              // snare
    const n = noise(0.2);
    const f = audio.createBiquadFilter();
    f.type = "highpass"; f.frequency.value = 1200;
    g.gain.setValueAtTime(0.7, when);
    g.gain.exponentialRampToValueAtTime(0.001, when + 0.18);
    n.connect(f); f.connect(g); n.start(when);
  } else if (name === "blue") {                // closed hat
    const n = noise(0.06);
    const f = audio.createBiquadFilter();
    f.type = "highpass"; f.frequency.value = 7000;
    g.gain.setValueAtTime(0.35, when);
    g.gain.exponentialRampToValueAtTime(0.001, when + 0.05);
    n.connect(f); f.connect(g); n.start(when);
  } else {                                     // green, a woodblock tone
    const o = audio.createOscillator();
    o.type = "triangle";
    o.frequency.setValueAtTime(880, when);
    g.gain.setValueAtTime(0.5, when);
    g.gain.exponentialRampToValueAtTime(0.001, when + 0.12);
    o.connect(g); o.start(when); o.stop(when + 0.14);
  }
}

function noise(seconds) {
  const frames = Math.floor(audio.sampleRate * seconds);
  const buffer = audio.createBuffer(1, frames, audio.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < frames; i++) data[i] = Math.random() * 2 - 1;
  const src = audio.createBufferSource();
  src.buffer = buffer;
  return src;
}

function hit(name, when) {
  const voice = voices[name];
  if (!voice.buffer) return builtIn(name, when);
  const src = audio.createBufferSource();
  src.buffer = voice.buffer;
  src.connect(master);
  src.start(when);
}

// ---------------------------------------------------------------- the clock

// Scheduled ahead against audio time, never setInterval. A timer fires when
// the browser gets round to it; this decides when the sound happens, which is
// the whole difference between a sequencer and a toy.
let playing = false;
let step = 0;
let nextNoteAt = 0;
let timer = null;
let tempo = 110;

function stepLength() { return 60 / tempo / 4; }   // sixteenth notes

function schedule() {
  while (nextNoteAt < audio.currentTime + LOOKAHEAD) {
    for (const colour of coloursInColumn(step)) hit(colour, nextNoteAt);
    lightColumn(step, nextNoteAt);
    nextNoteAt += stepLength();
    step = (step + 1) % 16;
  }
}

function start() {
  audio.resume();
  playing = true;
  step = 0;
  nextNoteAt = audio.currentTime + 0.05;
  timer = setInterval(schedule, TICK);
  playButton.textContent = "Stop";
  playButton.classList.add("on");
}

function stop() {
  playing = false;
  clearInterval(timer);
  document.querySelectorAll(".col-now").forEach((e) => e.classList.remove("col-now"));
  playButton.textContent = "Play";
  playButton.classList.remove("on");
}

// The playhead is drawn from a separate queue, because the scheduler runs
// ahead of what you are hearing and lighting a column early looks wrong.
const upcoming = [];
function lightColumn(which, when) { upcoming.push({ which, when }); }

function drawPlayhead() {
  while (upcoming.length && upcoming[0].when <= audio.currentTime) {
    const { which } = upcoming.shift();
    document.querySelectorAll(".col-now").forEach((e) => e.classList.remove("col-now"));
    document.querySelectorAll(`[data-step="${which}"]`)
      .forEach((e) => e.classList.add("col-now"));
  }
  requestAnimationFrame(drawPlayhead);
}
requestAnimationFrame(drawPlayhead);

// ---------------------------------------------------------------- the screen

const gridEl = document.getElementById("grid");
const noteEl = document.getElementById("note");
const linkEl = document.getElementById("link");
const playButton = document.getElementById("play");

const cellEls = [];
for (let r = 0; r < 4; r++) {
  const row = [];
  for (let c = 0; c < 16; c++) {
    const el = document.createElement("div");
    el.className = "cell";
    el.dataset.step = c;
    gridEl.appendChild(el);
    row.push(el);
  }
  cellEls.push(row);
}
const marks = document.createElement("div");
marks.className = "beat-marks";
for (let c = 0; c < 16; c++) {
  const s = document.createElement("span");
  s.textContent = c % 4 === 0 ? c / 4 + 1 : "·";
  if (c % 4 === 0) s.className = "strong";
  marks.appendChild(s);
}
gridEl.after(marks);

function paint() {
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 16; c++) {
      const colour = grid[r][c];
      cellEls[r][c].className = "cell" + (colour === "empty" ? "" : " " + colour);
    }
}

playButton.onclick = () => {
  playing ? stop() : start();
  // Let go of the button, or it keeps focus and the next space both presses
  // it and fires the shortcut below.
  playButton.blur();
};

// Space starts and stops. Not while a button has focus, where space already
// means press this one, and not while typing.
addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.repeat) return;
  const on = document.activeElement;
  if (on && (on.tagName === "BUTTON" || on.isContentEditable)) return;
  e.preventDefault();
  playing ? stop() : start();
});

const tempoEl = document.getElementById("tempo");
tempoEl.oninput = () => {
  tempo = +tempoEl.value;
  document.getElementById("tempoOut").value = tempo;
};
const volumeEl = document.getElementById("volume");
volumeEl.oninput = () => {
  master.gain.value = (+volumeEl.value / 100) ** 2;   // ears are not linear
  document.getElementById("volumeOut").value = volumeEl.value;
};
master.gain.value = 0.64;

// ---------------------------------------------------------------- samples

// IndexedDB, so an upload or a recording survives a reload. Nothing here is
// worth a round trip to the Python side; it belongs to the instrument.
let db = null;
const open = indexedDB.open("bubblegum", 1);
open.onupgradeneeded = () => open.result.createObjectStore("samples");
open.onsuccess = () => { db = open.result; COLOURS.forEach(loadSample); };

function keep(name, bytes) {
  if (db) db.transaction("samples", "readwrite").objectStore("samples").put(bytes, name);
}
function loadSample(name) {
  const ask = db.transaction("samples").objectStore("samples").get(name);
  ask.onsuccess = () => { if (ask.result) decode(name, ask.result, "saved"); };
}
async function decode(name, bytes, source) {
  try {
    voices[name].buffer = await audio.decodeAudioData(bytes.slice(0));
    voices[name].source = source;
    describe(name);
  } catch { describe(name, "could not read that file"); }
}

const voicesEl = document.getElementById("voices");
for (const name of COLOURS) {
  const card = document.createElement("div");
  card.className = "voice";
  card.innerHTML = `
    <h2><span class="dot" style="background: var(--${name})"></span>${name}</h2>
    <p class="source" id="src-${name}">built in</p>
    <div class="row">
      <button data-act="play">Hear it</button>
      <button data-act="upload">Upload</button>
      <button data-act="record">Record</button>
      <button data-act="reset">Built in</button>
    </div>
    <input type="file" accept="audio/*" hidden>`;
  voicesEl.appendChild(card);

  const file = card.querySelector("input");
  file.onchange = async () => {
    if (!file.files[0]) return;
    const bytes = await file.files[0].arrayBuffer();
    keep(name, bytes);
    decode(name, bytes, file.files[0].name);
  };

  card.querySelector('[data-act="play"]').onclick = () => {
    audio.resume(); hit(name, audio.currentTime + 0.01);
  };
  card.querySelector('[data-act="upload"]').onclick = () => file.click();
  card.querySelector('[data-act="reset"]').onclick = () => {
    voices[name].buffer = null; voices[name].source = "built in";
    if (db) db.transaction("samples", "readwrite").objectStore("samples").delete(name);
    describe(name);
  };
  card.querySelector('[data-act="record"]').onclick = (e) => record(name, e.target);
}

function describe(name, override) {
  document.getElementById("src-" + name).textContent = override || voices[name].source;
}

async function record(name, button) {
  if (button.dataset.busy) return;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch { return describe(name, "no microphone"); }
  const recorder = new MediaRecorder(stream);
  const parts = [];
  recorder.ondataavailable = (e) => parts.push(e.data);
  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    const bytes = await new Blob(parts).arrayBuffer();
    keep(name, bytes);
    decode(name, bytes, "recorded");
    button.textContent = "Record";
    button.classList.remove("recording");
    delete button.dataset.busy;
  };
  button.dataset.busy = "1";
  button.textContent = "Stop";
  button.classList.add("recording");
  recorder.start();
  setTimeout(() => recorder.state !== "inactive" && recorder.stop(), 2000);
  button.onclick = () => recorder.state !== "inactive" && recorder.stop();
}

// ---------------------------------------------------------------- the board

const events = new EventSource("/events");
events.onmessage = (e) => {
  const state = JSON.parse(e.data);
  grid = state.grid;
  paint();
  noteEl.textContent = state.note;
  linkEl.textContent = state.blind ? "cannot see the board"
    : state.settling ? "settling" : "live";
  linkEl.className = "state " + (state.blind ? "blind" : "live");
};
events.onerror = () => {
  linkEl.textContent = "no connection";
  linkEl.className = "state";
};

paint();

// ---------------------------------------------------------------- pipeline

// A second view that walks through what the vision side does to a frame, for
// explaining it to people. Each stage is there because of something visibly
// wrong in the one before, so going through them in order is the argument.
//
// It asks the server for the next set only once it has the last, which keeps
// it live without a timer and without competing with the instrument: the
// stages cost several times what simply reading the board does.

const modeButton = document.getElementById("mode");
const pipelineEl = document.getElementById("pipeline");
const stepsEl = document.getElementById("steps");
const shotEl = document.getElementById("shot");
const titleEl = document.getElementById("stageTitle");
const captionEl = document.getElementById("stageCaption");
const playable = [...document.querySelectorAll(".transport, .board, .voices")];

let showingPipeline = false;
let stages = [];
let atStage = 0;

modeButton.onclick = () => {
  showingPipeline = !showingPipeline;
  pipelineEl.hidden = !showingPipeline;
  playable.forEach((el) => (el.hidden = showingPipeline));
  modeButton.textContent = showingPipeline ? "Back to the sequencer"
                                           : "Show the pipeline";
  modeButton.blur();
  if (showingPipeline) pump();
};

async function pump() {
  while (showingPipeline) {
    try {
      const got = await fetch("/stages.json");
      if (!got.ok) throw new Error(got.status);
      stages = (await got.json()).stages;
      if (stepsEl.childElementCount !== stages.length) buildSteps();
      showStage(Math.min(atStage, stages.length - 1));
    } catch {
      captionEl.textContent = "waiting for a frame from the camera";
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
}

function buildSteps() {
  stepsEl.replaceChildren();
  stages.forEach((s, i) => {
    const b = document.createElement("button");
    b.textContent = `${i + 1}. ${s.title}`;
    b.onclick = () => { showStage(i); b.blur(); };
    stepsEl.appendChild(b);
  });
}

function showStage(i) {
  if (!stages.length) return;
  atStage = (i + stages.length) % stages.length;
  const s = stages[atStage];
  shotEl.src = "data:image/jpeg;base64," + s.jpeg;
  titleEl.textContent = s.title;
  captionEl.textContent = s.caption;
  [...stepsEl.children].forEach((b, n) => b.classList.toggle("on", n === atStage));
}

addEventListener("keydown", (e) => {
  if (!showingPipeline) return;
  if (e.key === "ArrowRight") { e.preventDefault(); showStage(atStage + 1); }
  if (e.key === "ArrowLeft") { e.preventDefault(); showStage(atStage - 1); }
});
