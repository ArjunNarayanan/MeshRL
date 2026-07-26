/* Mesh Quest — Copyright 2026 Arjun Narayanan. All rights reserved.
 * Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).
 * Free for noncommercial use; commercial use requires a separate license.
 */
// --- UI ---------------------------------------------------------------------

const storage = {
  _read() {
    try { return JSON.parse(localStorage.getItem("meshquest-records") || "{}"); }
    catch { return {}; }
  },
  get(shape) {
    const r = this._read();
    return shape in r ? r[shape] : null;
  },
  set(shape, moves) {
    try {
      const r = this._read();
      r[shape] = moves;
      localStorage.setItem("meshquest-records", JSON.stringify(r));
    } catch { /* private-mode browsers: records just don't persist */ }
  },
};

const game = new Game("L-shape", storage);

// activity log: everything the player does, exportable from the ? panel
const activityLog = (() => {
  try { return JSON.parse(localStorage.getItem("meshquest-log") || "[]"); }
  catch { return []; }
})();
let logSaveQueued = false;
function logEvent(type, data = {}) {
  activityLog.push({ t: new Date().toISOString(), shape: game.shape, type, ...data });
  if (activityLog.length > 3000) activityLog.splice(0, activityLog.length - 3000);
  if (!logSaveQueued) {
    logSaveQueued = true;
    setTimeout(() => {
      logSaveQueued = false;
      try { localStorage.setItem("meshquest-log", JSON.stringify(activityLog)); } catch {}
    }, 400);
  }
}
let state = null;
let mode = "insert_vertex";
let selectedVertex = null;
let wasWon = false;
let wasNearWin = false;
let hitEdges = [], hitVerts = [], hoverTarget = null;
// user-adjusted board offset, remembered across levels and sessions
const view = (() => {
  try { return { x: 0, y: 0, ...JSON.parse(localStorage.getItem("meshquest-view") || "{}") }; }
  catch { return { x: 0, y: 0 }; }
})();
function saveView() {
  try { localStorage.setItem("meshquest-view", JSON.stringify({ x: view.x, y: view.y })); } catch {}
}
let renderQueued = false;
function queueRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => { renderQueued = false; render(); });
}
const coarsePointer = window.matchMedia("(pointer: coarse)").matches;
const EDGE_PICK_RADIUS = coarsePointer ? 48 : 40;
const VERTEX_PICK_RADIUS = coarsePointer ? 38 : 30;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

// --- tutorial + level notes -------------------------------------------------

const TUT_SVG = {
  tiles: `<svg viewBox="0 0 120 84">
    <polygon class="tvis-tile t1" points="14,14 60,14 60,49 14,49"></polygon>
    <polygon class="tvis-tile t2" points="60,14 106,14 106,49 60,49"></polygon>
    <polygon class="tvis-tile t3" points="14,49 60,49 60,79 14,79"></polygon>
    <path class="tvis-line" d="M14,14 L106,14 L106,49 L60,49 L60,79 L14,79 Z"></path>
    <path class="tvis-line" d="M60,14 L60,49 L14,49"></path>
  </svg>`,
  wants: `<svg viewBox="0 0 120 84">
    <line class="tvis-line tvis-ray r1" pathLength="1" x1="60" y1="42" x2="60" y2="12"></line>
    <line class="tvis-line tvis-ray r2" pathLength="1" x1="60" y1="42" x2="94" y2="42"></line>
    <line class="tvis-line tvis-ray r3" pathLength="1" x1="60" y1="42" x2="60" y2="72"></line>
    <line class="tvis-line tvis-ray r4" pathLength="1" x1="60" y1="42" x2="26" y2="42"></line>
    <circle class="tvis-dot tvis-blush" cx="60" cy="42" r="8"></circle>
  </svg>`,
  pop: `<svg viewBox="0 0 120 84">
    <line class="tvis-line" x1="14" y1="42" x2="106" y2="42"></line>
    <circle class="tvis-dot tvis-pop" cx="60" cy="42" r="8"></circle>
  </svg>`,
  draw: `<svg viewBox="0 0 120 84">
    <circle class="tvis-dot" cx="20" cy="42" r="8"></circle>
    <circle class="tvis-dot" cx="100" cy="42" r="8"></circle>
    <line class="tvis-line tvis-draw" pathLength="1" x1="20" y1="42" x2="100" y2="42"></line>
  </svg>`,
  happy: `<svg viewBox="0 0 120 84">
    <circle class="tvis-good g1" cx="30" cy="42" r="9"></circle>
    <circle class="tvis-good g2" cx="60" cy="42" r="9"></circle>
    <circle class="tvis-good g3" cx="90" cy="42" r="9"></circle>
  </svg>`,
  par: `<svg viewBox="0 0 120 84">
    <path class="tvis-line" d="M60,14 L98,72 L22,72 Z"></path>
    <circle class="tvis-badge" cx="60" cy="52" r="13"></circle>
    <text class="tvis-badge-text" x="60" y="57">1</text>
  </svg>`,
  hubs: `<svg viewBox="0 0 236 84">
    <path class="tvis-line" d="M 58.0,12.0 L 51.3,32.7 L 29.5,32.7 L 47.1,45.5 L 40.4,66.3 L 58.0,53.5 L 75.6,66.3 L 68.9,45.5 L 86.5,32.7 L 64.7,32.7 Z"></path>
    <line class="tvis-spoke" x1="51.3" y1="32.7" x2="58" y2="42"></line><line class="tvis-spoke" x1="47.1" y1="45.5" x2="58" y2="42"></line><line class="tvis-spoke" x1="58.0" y1="53.5" x2="58" y2="42"></line><line class="tvis-spoke" x1="68.9" y1="45.5" x2="58" y2="42"></line><line class="tvis-spoke" x1="64.7" y1="32.7" x2="58" y2="42"></line>
    <circle class="tvis-dot tvis-blush tvis-pulse" cx="58" cy="42" r="7"></circle>
    <text class="tvis-flat-label" x="58" y="72" text-anchor="middle">5 lines, wants 4</text>
    <path class="tvis-line" d="M 178.0,12.0 L 171.3,32.7 L 149.5,32.7 L 167.1,45.5 L 160.4,66.3 L 178.0,53.5 L 195.6,66.3 L 188.9,45.5 L 206.5,32.7 L 184.7,32.7 Z"></path>
    <line class="tvis-spoke" x1="167.1" y1="45.5" x2="170.0" y2="46.0"></line><line class="tvis-spoke" x1="178.0" y1="53.5" x2="170.0" y2="46.0"></line><line class="tvis-spoke" x1="188.9" y1="45.5" x2="170.0" y2="46.0"></line><line class="tvis-spoke" x1="184.7" y1="32.7" x2="187.0" y2="38.0"></line><line class="tvis-spoke" x1="171.3" y1="32.7" x2="187.0" y2="38.0"></line><line class="tvis-spoke" x1="170.0" y1="46.0" x2="187.0" y2="38.0"></line>
    <circle class="tvis-good" cx="170.0" cy="46.0" r="6"></circle>
    <circle class="tvis-good g2" cx="187.0" cy="38.0" r="6"></circle>
    <text class="tvis-ok-label" x="178" y="72" text-anchor="middle">nobody over-stuffed</text>
  </svg>`,
  flatfix: `<svg viewBox="0 0 240 84">
    <path class="tvis-line" d="M10,74 L90,74 L50,14 Z"></path>
    <circle class="tvis-dot tvis-blush tvis-pulse" cx="30" cy="44" r="7"></circle>
    <text class="tvis-flat-label" x="42" y="40">180&#176;</text>
    <path class="tvis-line" d="M150,74 L230,74 L190,14 Z"></path>
    <line class="tvis-spoke" x1="170" y1="44" x2="190" y2="52"></line>
    <line class="tvis-spoke" x1="210" y1="44" x2="190" y2="52"></line>
    <line class="tvis-spoke" x1="190" y1="74" x2="190" y2="52"></line>
    <circle class="tvis-good" cx="170" cy="44" r="6"></circle>
    <circle class="tvis-good g2" cx="210" cy="44" r="6"></circle>
    <circle class="tvis-good g3" cx="190" cy="74" r="6"></circle>
    <circle class="tvis-dot tvis-pop" cx="190" cy="52" r="7"></circle>
  </svg>`,
};

const TUTORIAL_L = [
  { svg: "tiles", chip: "r-score", next: "Let's go!",
    text: "Cut the shape into <b>4-sided tiles</b>." },
  { svg: "wants", vertex: 3, next: "Help it!",
    text: "Every dot wants just-right lines.<br>Red dot = unhappy." },
  { svg: "pop", mode: "insert_vertex", require: { op: "insert_vertex", edge: [0, 1] },
    text: "Tap the glowing edge — pop a new dot!" },
  { svg: "pop", mode: "insert_vertex", require: { op: "insert_vertex", edge: [5, 0] },
    text: "Now the left edge. One more dot!" },
  { svg: "draw", mode: "insert_edge", require: { op: "insert_edge", pair: [6, 3] },
    text: "Tap the new dot, then the red dot." },
  { svg: "draw", mode: "insert_edge", require: { op: "insert_edge", pair: [7, 3] },
    text: "Last line!" },
  { svg: "happy", chip: "r-par", next: "Whoa",
    text: "All green! The corners <i>always</i> add up — 200-year-old math promises." },
  { svg: "par", next: "Play!",
    text: "<b>Par</b> = best score possible.<br>Match it on every level." },
];

// step-1 illustration drawn from the actual level: its outline, being tiled
function makeShapeTilesSvg(st) {
  const xs = st.vertices.map(v => v.x), ys = st.vertices.map(v => v.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const sc = Math.min(100 / Math.max(maxX - minX, 1e-9), 64 / Math.max(maxY - minY, 1e-9));
  const X = x => 60 + (x - (minX + maxX) / 2) * sc;
  const Y = y => 42 - (y - (minY + maxY) / 2) * sc;
  const pos = {};
  st.vertices.forEach(v => { pos[v.id] = X(v.x).toFixed(1) + "," + Y(v.y).toFixed(1); });
  const polys = st.faces.map(f => f.vertices.map(id => pos[id]).join(" "));
  const clip = "tclip" + Math.random().toString(36).slice(2, 7);
  let cells = "";
  for (let gx = 0; gx < 5; gx++) {
    for (let gy = 0; gy < 4; gy++) {
      cells += `<rect class="tvis-cell" style="animation-delay:${(((gx + gy) % 5) * 0.3).toFixed(1)}s"
        x="${8 + gx * 21}" y="${5 + gy * 18.5}" width="19" height="16.5"></rect>`;
    }
  }
  return `<svg viewBox="0 0 120 84">
    <clipPath id="${clip}">${polys.map(p => `<polygon points="${p}"></polygon>`).join("")}</clipPath>
    <g clip-path="url(#${clip})">${cells}</g>
    ${polys.map(p => `<polygon class="tvis-line" points="${p}" fill="none"></polygon>`).join("")}
  </svg>`;
}

// levels whose tutorial walks the actual (certified) solve
const SOLUTION_TUTORIALS = {
  "Star": st => [
    { svgRaw: makeShapeTilesSvg(st), chip: "r-par", next: "Show me",
      text: "Every point needs its own tile. <b>Par is 4</b>: four dots must stay unhappy — you only choose <i>which</i>." },
    { svg: "draw", mode: "insert_edge", require: { op: "insert_edge", pair: [3, 7] },
      text: "Cut straight across, inside corner to inside corner." },
    { svg: "pop", mode: "insert_vertex", require: { op: "insert_vertex", edge: [3, 7] },
      text: "Pop a dot in the middle of that new line. This is a <b>hub</b>." },
    { svg: "draw", mode: "insert_edge", require: { op: "insert_edge", pair: [5, 10] },
      text: "Join the inside corner between two points to the hub — two tiles done!" },
    { svg: "draw", mode: "insert_edge", require: { op: "insert_edge", pair: [3, 7] },
      text: "Same cut again, this time across the <i>other</i> side." },
    { svg: "pop", mode: "insert_vertex", require: { op: "insert_vertex", edge: [3, 7] },
      text: "Pop a second hub in the middle of it." },
    { svg: "draw", mode: "insert_edge", require: { op: "insert_edge", pair: [11, 9] },
      text: "Join an inside corner to this hub." },
    { svg: "draw", mode: "insert_edge", require: { op: "insert_edge", pair: [11, 1] },
      text: "And the last one!" },
    { svg: "hubs", next: "Play!",
      text: "Why two hubs? One hub in the middle is prettier — but it ends up with <b>5</b> lines when it wants 4. Too many is just as unhappy as too few, so that costs 6. With two hubs nobody is over-stuffed: 4 dots are just a little short, and 4 is the best anyone can do here." },
  ],
  "Triangle": st => [
    { svgRaw: makeShapeTilesSvg(st), chip: "r-par", next: "Let's go!",
      text: "The Triangle's <b>Par is 1</b> — one dot must stay imperfect. Here's the classic way." },
    { svg: "pop", mode: "insert_vertex", require: { op: "insert_vertex", edge: [0, 1] },
      text: "A dot on every side. First: the glowing edge." },
    { svg: "pop", mode: "insert_vertex", require: { op: "insert_vertex", edge: [1, 2] },
      text: "Second side!" },
    { svg: "pop", mode: "insert_vertex", require: { op: "insert_vertex", edge: [2, 0] },
      text: "Third side!" },
    { svg: "draw", mode: "insert_edge", require: { op: "insert_edge", pair: [3, 5] },
      text: "Draw a line between two of the new dots." },
    { svg: "pop", mode: "insert_vertex", require: { op: "insert_vertex", edge: [3, 5] },
      text: "Pop a dot in the middle of that new line." },
    { svg: "draw", mode: "insert_edge", require: { op: "insert_edge", pair: [6, 4] },
      text: "Connect the middle dot to the last side dot." },
    { svg: "happy", next: "Play!",
      text: "Three tiles — par! The middle dot wants 4 lines but has 3: the unavoidable 1. This trick is called <b>Catmull&ndash;Clark</b>." },
  ],
};

function buildTutorial(shape, st) {
  if (shape === "L-shape") return { steps: TUTORIAL_L };
  if (SOLUTION_TUTORIALS[shape]) {
    return { steps: SOLUTION_TUTORIALS[shape](st) };
  }
  // one hint card, tailored to this shape. No scripted actions: a generic
  // "pop a dot first" prompt walks players off levels whose solves are pure
  // chords (Pac-Man, Plus, Gear).
  return {
    steps: [{
      svgRaw: makeShapeTilesSvg(st),
      chip: "r-par",
      next: "Play!",
      text: (LEVEL_HINTS[shape] || "Cut it into 4-sided tiles.")
        + `<br><br>${LEVEL_NOTES[shape] || ""}`,
    }],
  };
}

// what to try on each level — sets the right tool expectation, no spoilers
const LEVEL_HINTS = {
  "T-bracket": "Cut across where the stem meets the bar. Some cuts need a new dot to land on.",
  "I-bracket": "One bar at a time. Pop a dot wherever a cut has nowhere to land.",
  "U-channel": "Give the floor and each prong their own tiles.",
  "Z-shape": "Cut <i>with</i> the lean, not against it.",
  "Plus": "Lines only — no new dots needed!",
  "Staircase": "One step at a time. Each step is a little L.",
  "Pentagon": "Five sides is odd! Pop one dot to make it even, then cut.",
  "Semicircle": "Pop a dot on the flat side first. Then pick two round dots to be corners.",
  "Pac-Man": "No new dots needed — try cutting straight out from the middle.",
  "Star": "Give every point its own tile, then sort out the middle.",
  "Square hole": "Work around the ring, tile by tile. The seam line belongs — keep it!",
  "Triforce ring": "Tile around the ring. The little seam line already belongs — keep it!",
  "Gear": "One cut across each tooth makes six tiles fast. Then the ring of dots left in the middle is the real puzzle.",
};

const LEVEL_NOTES = {
  "T-bracket": "Two grumpy inside corners now. A perfect 0 is still possible.",
  "I-bracket": "Eight corners to fix. A plan beats random cuts.",
  "U-channel": "Each prong wants its own tiles. Par 0.",
  "Z-shape": "The two inside corners lean opposite ways. Cut with the lean.",
  "Plus": "Four arms, four inside corners. There's a beautiful symmetric answer.",
  "Staircase": "Three L-shapes in a trench coat. The L trick works on every step.",
  "Triangle": "3 sides — an odd number! Tiles need an even rim, so pop a dot first. And one dot must stay sad: par 1. You pick which.",
  "Pentagon": "A bit too pointy overall: par 1. One dot takes the hit — where will you hide it?",
  "Semicircle": "Round shapes secretly want corners. Pick 2 arc dots to be them — par 2.",
  "Pac-Man": "Par 3 — odd, so no fair sharing. Psst: seen this shape before, in disguise?",
  "Star": "The star loves symmetry — but symmetric answers score 6, not 4. Break the pattern to win. Really.",
  "Square hole": "A hole! Inside corners cancel outside corners: perfect 0 possible. Holes help.",
  "Triforce ring": "Pointy corners and hole corners cancel exactly: par 0, a perfect mesh.",
  "Gear": "The teeth are the easy part — one cut each. What is left is a 12-sided ring, and a round ring secretly wants to be a square: exactly 4 dots must stay unhappy. That is the par.",
};

let tut = null; // { i, steps } while a card sequence is active
const seenNotes = (() => {
  try { return new Set(JSON.parse(localStorage.getItem("meshquest-notes") || "[]")); }
  catch { return new Set(); }
})();

function tutStep() { return tut ? tut.steps[tut.i] : null; }

function sameSet(a, b) {
  return a.length === b.length && a.every(x => b.includes(x));
}

function tutAllows(kind, id) {
  const s = tutStep();
  if (!s) return true;
  if (!s.require) return false; // info steps: board locked
  const r = s.require;
  if (r.op === "insert_vertex" || r.op === "delete_edge") {
    if (kind !== "edge") return false;
    if (!r.edge) return true; // unconstrained step: any edge
    const e = state.edges.find(e => e.id === id);
    return !!e && sameSet([e.v1, e.v2], r.edge);
  }
  if (r.op === "insert_edge") {
    if (kind !== "vertex") return false;
    return !r.pair || r.pair.includes(id);
  }
  return false;
}

function showCard(eyebrow, html, nextLabel, onNext, onSkip) {
  const card = document.getElementById("tutcard");
  document.getElementById("tut-eyebrow").textContent = eyebrow;
  document.getElementById("tut-body").innerHTML = html;
  const nextBtn = document.getElementById("tut-next");
  const skipBtn = document.getElementById("tut-skip");
  nextBtn.classList.toggle("hidden", !nextLabel);
  if (nextLabel) nextBtn.textContent = nextLabel;
  nextBtn.onclick = onNext || null;
  skipBtn.onclick = onSkip || (() => hideCard());
  card.classList.add("show");
  if (state) render();
}

function hideCard() {
  document.getElementById("tutcard").classList.remove("show");
  if (state) render();
}

function tutShow() {
  const s = tutStep();
  if (s.mode) setMode(s.mode, true);
  const dots = tut.steps.length > 1
    ? tut.steps.map((_, k) => (k <= tut.i ? "●" : "○")).join(" ")
    : game.shape;
  const body = (s.svgRaw || (s.svg ? TUT_SVG[s.svg] : "")) + `<div>${s.text}</div>`;
  showCard(dots, body, s.next || null, s.next ? tutAdvance : null, tutEnd);
  render();
}

function tutAdvance() {
  tut.i += 1;
  if (tut.i >= tut.steps.length) tutEnd();
  else tutShow();
}

function tutEnd() {
  logEvent("tutorial_end", { at_step: tut ? tut.i : null });
  tut = null;
  try { localStorage.setItem("meshquest-tutorial", "done"); } catch {}
  hideCard();
  // any moves made during a card sequence are real: progress is kept
  render();
}

function tutStart() {
  hideCard();
  document.getElementById("autosmooth").checked = true;
  // scripted walkthroughs need their exact starting board; hint cards run on
  // whatever the player already has, so in-progress work is never lost
  const scripted = game.shape === "L-shape" || !!SOLUTION_TUTORIALS[game.shape];
  if (scripted) game.reset(game.shape);
  state = game.serialize();
  selectedVertex = null;
  const { steps } = buildTutorial(game.shape, state);
  tut = { i: 0, steps };
  logEvent("tutorial_start", { steps: steps.length });
  tutShow();
}

function tutCheck(desc) {
  const s = tutStep();
  if (!s || !s.require) return;
  const r = s.require;
  if (r.op !== desc.op) return;
  const want = r.edge || r.pair;
  if (!want || sameSet(desc.verts, want)) tutAdvance();
}

function maybeShowNote(shape) {
  if (tut || !LEVEL_NOTES[shape] || seenNotes.has(shape)) return;
  const dismiss = () => {
    seenNotes.add(shape);
    try { localStorage.setItem("meshquest-notes", JSON.stringify([...seenNotes])); } catch {}
    hideCard();
  };
  showCard(`Level note — ${shape}`, LEVEL_NOTES[shape], "Got it", dismiss, dismiss);
}

const HINTS = {
  insert_vertex: "Click any edge to split it with a midpoint vertex.",
  delete_vertex: "Click a highlighted vertex — degree 2 and not an original corner.",
  insert_edge: "Click two vertices on the same face to connect them.",
  delete_edge: "Click an interior edge to delete it and merge its two faces.",
};

function toast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("show"), 2600);
}

function refresh() {
  state = game.serialize();
  selectedVertex = null;
  render();
}

function doOp(op, params) {
  const auto = document.getElementById("autosmooth").checked;
  let verts = [];
  if (op === "insert_vertex" || op === "delete_edge") {
    const e = state.edges.find(e => e.id === params.edge);
    if (e) verts = [e.v1, e.v2];
  } else if (op === "insert_edge") {
    verts = [params.a, params.b];
  }
  try {
    game.applyOp(op, params, auto);
  } catch (e) {
    logEvent("op_rejected", { op, verts, error: e.message });
    toast(e.message);
    return;
  }
  refresh();
  logEvent("op", { op, verts, moves: state.moves, scores: state.scores });
  tutCheck({ op, verts });
}

function partnerMap() {
  const map = {};
  for (const [a, b] of state.insertable_pairs) {
    (map[a] = map[a] || new Set()).add(b);
    (map[b] = map[b] || new Set()).add(a);
  }
  return map;
}

function render() {
  const svg = document.getElementById("canvas");
  svg.setAttribute("class", mode === "insert_vertex" || mode === "delete_edge" ? "edge-mode" : "");
  // narrow screens: while a card is open the header/toolbar are dead weight
  // (the tool is auto-selected), so give that space to the board — except on
  // steps that spotlight a readout chip, which need the header visible.
  const cardEl = document.getElementById("tutcard");
  const cardOpen = cardEl.classList.contains("show");
  const stepNow = tutStep();
  document.body.classList.toggle("card-open", cardOpen);
  document.body.classList.toggle("chip-step", !!(stepNow && stepNow.chip));
  const W = svg.clientWidth, H = svg.clientHeight;
  const M = Math.min(60, Math.max(28, Math.min(W, H) * 0.08));
  // when the tutorial/note card is open on a wide screen it sits on the left;
  // reserve that band so it never covers the geometry
  const leftPad = cardOpen && W > 720 ? cardEl.offsetWidth + 20 + 36 : M;
  const xs = state.vertices.map(v => v.x), ys = state.vertices.map(v => v.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  // every level scales into one fixed square box, so the layout — and the
  // nav arrows anchored to the box — is identical across all levels
  const availW = W - leftPad - M;
  // on narrow screens the card is a bottom sheet: reserve its band vertically
  const bottomPad = cardOpen && W <= 720 ? cardEl.offsetHeight + 16 : 0;
  const availH = H - 2 * M - bottomPad;
  const box = Math.max(110, Math.min(availW - (W > 720 ? 150 : 110), availH));
  const boxCX = leftPad + availW / 2;
  const boxCY = M + availH / 2;
  const scale = Math.min(box / Math.max(maxX - minX, 1e-9),
                         box / Math.max(maxY - minY, 1e-9));
  const X = x => boxCX + (x - (minX + maxX) / 2) * scale + view.x;
  const Y = y => boxCY - (y - (minY + maxY) / 2) * scale + view.y;

  const pos = {};
  for (const v of state.vertices) pos[v.id] = [X(v.x), Y(v.y)];

  const partners = partnerMap();
  let out = "";

  for (const f of state.faces) {
    const pts = f.vertices.map(id => pos[id].join(",")).join(" ");
    let cls = f.degree === 4 ? "d4" : f.degree === 3 ? "d3" : f.degree === 2 ? "d2" : "dbig";
    if (f.quality < state.quality_threshold) cls = "degen";
    out += `<polygon class="face ${cls}" points="${pts}"></polygon>`;
  }

  const groups = {};
  for (const e of state.edges) {
    const key = Math.min(e.v1, e.v2) + "-" + Math.max(e.v1, e.v2);
    (groups[key] = groups[key] || []).push(e);
  }
  hitEdges = [];
  for (const group of Object.values(groups)) {
    group.forEach((e, i) => {
      const [x1, y1] = pos[e.v1], [x2, y2] = pos[e.v2];
      const off = (i - (group.length - 1) / 2) * 26;
      const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
      const cx = (x1 + x2) / 2 - dy / len * off, cy = (y1 + y2) / 2 + dx / len * off;
      const d = `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`;
      let dim = mode === "delete_edge" && !e.deletable ? "dim" : "";
      let tutCls = "";
      const ts = tutStep();
      if (ts && ts.require) {
        if (tutAllows("edge", e.id)) tutCls = "tut-target";
        else dim = "dim";
      }
      const samples = [];
      for (let t = 0; t <= 1.001; t += 0.125) {
        const u = 1 - t;
        samples.push([u * u * x1 + 2 * t * u * cx + t * t * x2,
                      u * u * y1 + 2 * t * u * cy + t * t * y2]);
      }
      hitEdges.push({ id: e.id, deletable: e.deletable, samples });
      // soft glow marks every legal target for the current tool
      const eligible = mode === "insert_vertex"
        || (mode === "delete_edge" && e.deletable);
      const glow = eligible && !dim && !tutCls && tutAllows("edge", e.id) ? "glow" : "";
      out += `<g class="${dim}">
        <path class="edge ${e.boundary ? "boundary" : ""} ${tutCls} ${glow}" data-edge="${e.id}" d="${d}"></path>
      </g>`;
    });
  }

  hitVerts = [];
  for (const v of state.vertices) {
    const defect = v.degree - v.desired;
    const cls = defect === 0 ? "ok" : Math.abs(defect) === 1 ? "off1" : "off2";
    let actionable = false, dim = false;
    if (mode === "delete_vertex") { actionable = v.deletable; dim = !v.deletable; }
    if (mode === "insert_edge") {
      actionable = true;
      if (selectedVertex !== null && selectedVertex !== v.id) {
        dim = !(partners[selectedVertex] && partners[selectedVertex].has(v.id));
        actionable = !dim;
      }
    }
    let tutCls = "";
    const ts = tutStep();
    if (ts) {
      if (ts.require && ts.require.op === "insert_edge") {
        if (tutAllows("vertex", v.id)) tutCls = "tut-target";
        else dim = true;
      } else if (ts.require) {
        dim = false; // edge steps: leave vertices visible but inert
      } else if (ts.vertex === v.id) {
        tutCls = "tut-target"; // info-step spotlight
      }
    }
    const sel = v.id === selectedVertex ? "selected" : "";
    const [x, y] = pos[v.id];
    const r = v.user ? 10 : 8.5;
    hitVerts.push({ id: v.id, x, y, eligible: actionable });
    const heat = v.quality < state.quality_threshold
      ? `<circle class="heat" cx="${x}" cy="${y}" r="19"
           opacity="${Math.min(0.4, (state.quality_threshold - Math.max(v.quality, -1)) * 0.3).toFixed(2)}"></circle>`
      : "";
    out += `<g class="${dim ? "dim" : ""}">
      ${heat}
      <circle class="vertex ${cls} ${v.user ? "user" : ""} ${sel} ${tutCls} ${
        actionable && !dim && !tutCls && tutAllows("vertex", v.id) ? "glow" : ""}"
              data-vertex="${v.id}" cx="${x}" cy="${y}" r="${r}">
        <title>v${v.id} — degree ${v.degree} / desired ${v.desired}${v.user ? " — corner" : ""}</title>
      </circle>
      ${defect !== 0 ? `<text class="vlabel" x="${x}" y="${y + 3}">${defect > 0 ? "+" : ""}${defect}</text>` : ""}
    </g>`;
  }

  svg.innerHTML = out;
  hoverTarget = null;

  const s = state.scores;
  document.getElementById("s-vertex").textContent = s.vertex;
  document.getElementById("s-par").textContent = state.par;
  document.getElementById("s-quality").textContent = s.min_quality.toFixed(2);
  document.getElementById("s-moves").textContent = state.moves;
  document.getElementById("s-best").textContent = state.best ?? "–";
  document.getElementById("r-score").classList.toggle("at-par", s.vertex === state.par);
  document.getElementById("r-quality").classList.toggle("alert", s.min_quality < state.quality_threshold);
  const locked = !!tut;
  document.getElementById("undo").disabled = !state.can_undo;
  document.getElementById("smoothbtn").disabled = locked;
  document.getElementById("reset").disabled = locked;
  document.getElementById("shape").disabled = locked;
  document.getElementById("autosmooth").disabled = locked;
  document.getElementById("hint").textContent = HINTS[mode];
  const ts2 = tutStep();
  document.getElementById("r-score").classList.toggle("tut-target", !!ts2 && ts2.chip === "r-score");
  document.getElementById("r-par").classList.toggle("tut-target", !!ts2 && ts2.chip === "r-par");

  const sel2 = document.getElementById("shape");
  if (sel2.options.length === 0) {
    for (const name of state.shapes) {
      const o = document.createElement("option");
      o.value = o.textContent = name;
      sel2.appendChild(o);
    }
  }
  sel2.value = state.shape;

  const isWon = s.vertex === state.par && s.face === 0
      && s.min_quality >= state.quality_threshold;
  const shapes = state.shapes;
  const idx = shapes.indexOf(state.shape);
  const nextShape = shapes[(idx + 1) % shapes.length];
  const prevShape = shapes[(idx - 1 + shapes.length) % shapes.length];
  const nextBtn = document.getElementById("next");
  const prevBtn = document.getElementById("prev");
  nextBtn.classList.toggle("show", !tut);
  prevBtn.classList.toggle("show", !tut);
  nextBtn.classList.toggle("win", isWon);
  nextBtn.title = "Next level: " + nextShape;
  prevBtn.title = "Previous level: " + prevShape;
  // arrows anchor to the fixed box, never to the mesh: globally constant
  prevBtn.style.left = Math.max(boxCX - box / 2 - 70, 8) + "px";
  nextBtn.style.left = Math.min(boxCX + box / 2 + 24, W - 66) + "px";
  prevBtn.style.top = boxCY + "px";
  nextBtn.style.top = boxCY + "px";
  if (isWon && !wasWon) {
    confettiBurst();
    logEvent("win", { moves: state.moves, new_record: state.new_record });
    if (state.new_record) toast(`\u{1F3C6} New record: ${state.moves} moves`);
  }
  wasWon = isWon;

  // the classic trap: at par with all quads, but a degenerate corner blocks
  // the win. Teach it the moment it happens.
  const isNearWin = !isWon && s.vertex === state.par && s.face === 0
      && s.min_quality < state.quality_threshold;
  if (isNearWin && !wasNearWin) {
    wasNearWin = true;
    logEvent("flat_corner_card", { min_quality: s.min_quality });
    if (!tut) {
      showCard("Good start!",
        TUT_SVG.flatfix
        + "<div>The numbers add up — but that new dot lies <b>flat</b> (180&deg;), too squished for a real tile. We can do a little better: a dot on every side, meeting in the middle.</div>",
        "Keep going", hideCard, hideCard);
    }
  } else {
    wasNearWin = isNearWin;
  }
}

function confettiBurst() {
  if (reducedMotion.matches) return;
  const canvas = document.getElementById("confetti");
  const stage = document.getElementById("stage");
  canvas.width = stage.clientWidth;
  canvas.height = stage.clientHeight;
  const ctx = canvas.getContext("2d");
  const style = getComputedStyle(document.documentElement);
  const colors = ["--good", "--accent", "--warn", "--bad"].map(t => style.getPropertyValue(t).trim());
  const parts = [];
  for (let i = 0; i < 140; i++) {
    parts.push({
      x: canvas.width / 2 + (Math.random() - 0.5) * canvas.width * 0.5,
      y: canvas.height * (0.25 + Math.random() * 0.2),
      vx: (Math.random() - 0.5) * 9,
      vy: -4 - Math.random() * 7,
      w: 5 + Math.random() * 5,
      h: 3 + Math.random() * 4,
      rot: Math.random() * Math.PI,
      vr: (Math.random() - 0.5) * 0.3,
      color: colors[i % colors.length],
    });
  }
  const t0 = performance.now();
  function tick(t) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const alive = (t - t0) < 2600;
    for (const p of parts) {
      p.vy += 0.18; p.x += p.vx; p.y += p.vy; p.rot += p.vr;
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot);
      ctx.fillStyle = p.color;
      ctx.globalAlpha = Math.max(0, 1 - (t - t0) / 2600);
      ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
      ctx.restore();
    }
    if (alive) requestAnimationFrame(tick);
    else ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  requestAnimationFrame(tick);
}

// --- nearest-target picking -------------------------------------------------

function distToSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const lsq = dx * dx + dy * dy;
  const t = lsq ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lsq)) : 0;
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

function pickTarget(mx, my) {
  if (mode === "insert_vertex" || mode === "delete_edge") {
    let best = null, bestDist = EDGE_PICK_RADIUS;
    for (const e of hitEdges) {
      if (mode === "delete_edge" && !e.deletable) continue;
      if (!tutAllows("edge", e.id)) continue;
      for (let i = 0; i + 1 < e.samples.length; i++) {
        const d = distToSegment(mx, my, ...e.samples[i], ...e.samples[i + 1]);
        if (d < bestDist) { bestDist = d; best = { kind: "edge", id: e.id }; }
      }
    }
    return best;
  }
  let best = null, bestDist = VERTEX_PICK_RADIUS;
  for (const v of hitVerts) {
    if (!v.eligible) continue;
    if (!tutAllows("vertex", v.id)) continue;
    const d = Math.hypot(mx - v.x, my - v.y);
    if (d < bestDist) { bestDist = d; best = { kind: "vertex", id: v.id }; }
  }
  return best;
}

function setHover(target) {
  const svg = document.getElementById("canvas");
  if (hoverTarget && (!target || target.kind !== hoverTarget.kind || target.id !== hoverTarget.id)) {
    const old = svg.querySelector(
      hoverTarget.kind === "edge"
        ? `[data-edge="${hoverTarget.id}"]` : `circle[data-vertex="${hoverTarget.id}"]`);
    if (old) old.classList.remove("near");
  }
  if (target) {
    const el = svg.querySelector(
      target.kind === "edge"
        ? `[data-edge="${target.id}"]` : `circle[data-vertex="${target.id}"]`);
    if (el) el.classList.add("near");
  }
  hoverTarget = target;
  svg.classList.toggle("has-target", !!target);
}

function mousePos(e) {
  const rect = document.getElementById("canvas").getBoundingClientRect();
  return [e.clientX - rect.left, e.clientY - rect.top];
}

const canvasEl = document.getElementById("canvas");
let panState = null;
let suppressClick = false;

canvasEl.addEventListener("mousemove", e => {
  if (!state || panState?.moved) return;
  setHover(pickTarget(...mousePos(e)));
});
canvasEl.addEventListener("mouseleave", () => setHover(null));
canvasEl.addEventListener("pointerdown", e => {
  if (!state) return;
  // touch: no hover exists, so preview the picked target on finger-down;
  // the click that follows on finger-up commits it
  if (e.pointerType === "touch") setHover(pickTarget(...mousePos(e)));
  panState = { sx: e.clientX, sy: e.clientY, vx: view.x, vy: view.y, moved: false };
  canvasEl.setPointerCapture(e.pointerId);
});
canvasEl.addEventListener("pointermove", e => {
  if (!panState) return;
  if (e.pointerType === "mouse" && !(e.buttons & 1)) return;
  const dx = e.clientX - panState.sx, dy = e.clientY - panState.sy;
  if (!panState.moved && Math.hypot(dx, dy) <= 5) return;
  if (!panState.moved) {
    panState.moved = true;
    setHover(null);
    canvasEl.classList.add("panning");
  }
  view.x = panState.vx + dx;
  view.y = panState.vy + dy;
  queueRender();
});
window.addEventListener("pointerup", () => {
  if (panState?.moved) {
    saveView();
    suppressClick = true;
    canvasEl.classList.remove("panning");
  }
  panState = null;
});
canvasEl.addEventListener("click", e => {
  if (suppressClick) { suppressClick = false; return; }
  if (!state) return;
  const target = pickTarget(...mousePos(e));
  if (!target) {
    if (mode === "insert_edge" && selectedVertex !== null) { selectedVertex = null; render(); }
    return;
  }
  if (target.kind === "edge") {
    doOp(mode === "insert_vertex" ? "insert_vertex" : "delete_edge", { edge: target.id });
  } else if (mode === "delete_vertex") {
    doOp("delete_vertex", { vertex: target.id });
  } else if (mode === "insert_edge") {
    if (selectedVertex === null) { selectedVertex = target.id; render(); }
    else if (selectedVertex === target.id) { selectedVertex = null; render(); }
    else doOp("insert_edge", { a: selectedVertex, b: target.id });
  }
});

function setMode(m, force) {
  const s = tutStep();
  if (s && !force && (!s.require || s.mode !== m)) {
    toast("Follow the tutorial step — or skip it below");
    return;
  }
  mode = m;
  selectedVertex = null;
  document.querySelectorAll(".mode").forEach(b =>
    b.classList.toggle("active", b.dataset.mode === m));
  if (state) render();
}

document.querySelectorAll(".mode").forEach(b =>
  b.addEventListener("click", () => setMode(b.dataset.mode)));
document.getElementById("undo").addEventListener("click", () => {
  try { game.undo(); } catch (e) { toast(e.message); return; }
  refresh();
  logEvent("undo", { moves: state.moves });
  tutCheck({ op: "undo", verts: [] });
});
document.getElementById("smoothbtn").addEventListener("click", () => {
  game.smooth(3);
  refresh();
  logEvent("smooth");
});
function gotoShape(shape, evt) {
  if (tut) tutEnd(); // defensive: a level change always exits the tutorial
  hideCard(); // and dismisses whatever card was open
  game.reset(shape);
  refresh();
  logEvent(evt);
  maybeShowNote(state.shape);
}

document.getElementById("reset").addEventListener("click", () =>
  gotoShape(document.getElementById("shape").value, "new_game"));
document.getElementById("shape").addEventListener("change", e =>
  gotoShape(e.target.value, "level_change"));
document.getElementById("next").addEventListener("click", () => {
  const shapes = state.shapes;
  gotoShape(shapes[(shapes.indexOf(state.shape) + 1) % shapes.length], "level_next");
});
document.getElementById("prev").addEventListener("click", () => {
  const shapes = state.shapes;
  gotoShape(shapes[(shapes.indexOf(state.shape) - 1 + shapes.length) % shapes.length], "level_prev");
});
document.getElementById("tut-open").addEventListener("click", tutStart);
document.getElementById("star").addEventListener("click", () => logEvent("star_click"));

// Heart: one per browser, so the counter reads "people who loved it" rather
// than "clicks". Counted as a GoatCounter event on the public build; on builds
// without analytics the button still gives feedback, it just isn't tallied.
const heartBtn = document.getElementById("heart");
let hearted = false;
try { hearted = localStorage.getItem("meshquest-hearted") === "1"; } catch {}
function paintHeart() {
  heartBtn.classList.toggle("hearted", hearted);
  heartBtn.querySelector(".glyph").innerHTML = hearted ? "&#9829;" : "&#9825;";
  heartBtn.title = hearted ? "Thanks for the love!" : "Love it?";
}
paintHeart();
heartBtn.addEventListener("click", () => {
  if (hearted) { toast("\u2665 Already loved — thank you!"); return; }
  hearted = true;
  try { localStorage.setItem("meshquest-hearted", "1"); } catch {}
  paintHeart();
  heartBtn.classList.add("pop");
  setTimeout(() => heartBtn.classList.remove("pop"), 600);
  logEvent("heart");
  if (window.goatcounter && typeof window.goatcounter.count === "function") {
    window.goatcounter.count({ path: "heart", title: "Heart", event: true });
  }
  toast("\u2665 Thank you!");
});
document.getElementById("log-export").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(activityLog, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "meshquest-log.json";
  a.click();
  URL.revokeObjectURL(a.href);
  toast(`Exported ${activityLog.length} events`);
});
window.addEventListener("resize", () => state && render());
window.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  const keys = { 1: "insert_vertex", 2: "delete_vertex", 3: "insert_edge", 4: "delete_edge" };
  if (keys[e.key]) setMode(keys[e.key]);
  else if (e.key === "u") document.getElementById("undo").click();
  else if (e.key === "s") document.getElementById("smoothbtn").click();
  else if (e.key === "ArrowLeft") document.getElementById("prev").click();
  else if (e.key === "ArrowRight") document.getElementById("next").click();
  else if (e.key === " ") {
    const nextBtn = document.getElementById("tut-next");
    if (document.getElementById("tutcard").classList.contains("show")
        && !nextBtn.classList.contains("hidden")) {
      e.preventDefault();
      nextBtn.click();
    }
  }
});

setMode("insert_vertex");
refresh();
let tutorialDone = false;
try { tutorialDone = localStorage.getItem("meshquest-tutorial") === "done"; } catch {}
if (!tutorialDone) tutStart();
