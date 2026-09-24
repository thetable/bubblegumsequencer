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
// Trim the silence a sample starts and ends with.
//
// You spend the first half second of a recording reaching back to the
// keyboard, so the sound begins somewhere in the middle of the buffer. Played
// untrimmed, that silence is a delay applied to every single hit, and on
// sixteenth notes at 120 bpm a step is only 125 ms, so a quarter second of
// dead air is two steps late. The tail matters for the opposite reason: a
// two second recording left whole overlaps the next fifteen steps.
//
// Measured against the sample's own peak rather than an absolute level, so it
// behaves the same whether you recorded loudly or quietly.
const TRIM_FLOOR = 0.02;    // counts as sound, as a fraction of the peak
const TRIM_SILENT = 0.002;  // below this peak the whole thing is silence
const TRIM_LEAD = 0.005;    // seconds kept before the transient
const TRIM_TAIL = 0.050;    // seconds kept after the last sound
const TRIM_FADE = 0.0015;   // a hair of fade, so a cut mid-wave cannot click

function trim(buffer) {
  const channels = [];
  let peak = 0;
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const data = buffer.getChannelData(c);
    channels.push(data);
    for (let i = 0; i < data.length; i++) {
      const v = Math.abs(data[i]);
      if (v > peak) peak = v;
    }
  }
  // Nothing but noise: leave it alone rather than trim it to nothing, so a
  // failed recording still sounds like a failed recording instead of silence.
  if (peak < TRIM_SILENT) return buffer;

  const floor = peak * TRIM_FLOOR;
  const loud = (i) => channels.some((d) => Math.abs(d[i]) >= floor);
  let first = 0, last = buffer.length - 1;
  while (first < last && !loud(first)) first++;
  while (last > first && !loud(last)) last--;

  const rate = buffer.sampleRate;
  const start = Math.max(0, first - Math.round(TRIM_LEAD * rate));
  const end = Math.min(buffer.length, last + 1 + Math.round(TRIM_TAIL * rate));
  if (start === 0 && end === buffer.length) return buffer;

  const out = audio.createBuffer(buffer.numberOfChannels, end - start, rate);
  const fade = Math.min(Math.round(TRIM_FADE * rate), (end - start) >> 1);
  for (let c = 0; c < channels.length; c++) {
    const to = out.getChannelData(c);
    to.set(channels[c].subarray(start, end));
    for (let i = 0; i < fade; i++) {
      to[i] *= i / fade;
      to[to.length - 1 - i] *= i / fade;
    }
  }
  return out;
}

// Bring every sample to the same peak, so a quiet recording is not simply
// lost under the synthesised voices.
//
// Peak rather than loudness. Matching perceived loudness would suit sustained
// sounds better, but these are drum hits: one big transient over a quiet body,
// and an RMS match would push that transient straight through the ceiling.
// Peak is the predictable choice and cannot clip.
//
// Capped, because gain is not free: a recording with nothing in it has a noise
// floor, and boosting silence thirty fold just gives you loud silence.
const LEVEL_TARGET = 0.89;   // leaves about a decibel of headroom
const LEVEL_MAX = 20;        // 26 dB, past which it is only noise being lifted

function level(buffer) {
  let peak = 0;
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const d = buffer.getChannelData(c);
    for (let i = 0; i < d.length; i++) {
      const v = Math.abs(d[i]);
      if (v > peak) peak = v;
    }
  }
  if (peak < TRIM_SILENT) return 1;
  const gain = Math.min(LEVEL_TARGET / peak, LEVEL_MAX);
  if (Math.abs(gain - 1) < 0.01) return 1;
  // In place: this buffer was decoded a moment ago and nothing else holds it.
  // The bytes it came from are untouched in IndexedDB either way.
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const d = buffer.getChannelData(c);
    for (let i = 0; i < d.length; i++) d[i] *= gain;
  }
  return gain;
}

async function decode(name, bytes, source) {
  try {
    const whole = await audio.decodeAudioData(bytes.slice(0));
    const cut = trim(whole);
    const gain = level(cut);
    voices[name].buffer = cut;
    // Says what it did, because a trim or a boost that fires when it should
    // not is otherwise invisible until the thing sounds wrong.
    const dB = 20 * Math.log10(gain);
    voices[name].source = `${source}, ${cut.duration.toFixed(2)} s`
      + (cut === whole ? "" : ` of ${whole.duration.toFixed(2)}`)
      + (gain === 1 ? "" : `, ${dB > 0 ? "+" : ""}${dB.toFixed(0)} dB`);
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
const detailEl = document.getElementById("stageDetail");
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
      detailEl.textContent = "";
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
  detailEl.textContent = s.detail || "";
  [...stepsEl.children].forEach((b, n) => b.classList.toggle("on", n === atStage));
}

addEventListener("keydown", (e) => {
  if (!showingPipeline) return;
  if (e.key === "ArrowRight") { e.preventDefault(); showStage(atStage + 1); }
  if (e.key === "ArrowLeft") { e.preventDefault(); showStage(atStage - 1); }
});
