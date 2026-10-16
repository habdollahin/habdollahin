// Builds "Mathematical Models & AI for Theranostic Digital Twins" deck.
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require("/root/.claude/skills/synced/beec76db-35b5-4c67-a273-90a2bc80b41c_fd0738f3-4d4d-4a53-bfde-7c9b8ebc1b13/pptx/scripts/apply_theme.js");

const OUT = process.argv[2] || "deck.pptx";

const THEME = {
  name: "Digital Twin Models",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2430", lt1: "FFFFFF", dk2: "0B3C49", lt2: "EEF4F5",
    accent1: "F2A541", accent2: "1F7A70", accent3: "E76F51", accent4: "264653",
    accent5: "8AB17D", accent6: "5F6B76", hlink: "1F7A70", folHlink: "264653",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "Mathematical Models for Digital Twins and Virtual Trials";
pres.author = "Hamid Abdollahi";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
// Fixed hex colours (not theme references) so every viewer renders fills, e.g. Google Slides, Keynote, legacy .ppt.
const C = { text1: HEX.dk1, text2: HEX.dk2, background1: HEX.lt1, background2: HEX.lt2, accent1: HEX.accent1, accent2: HEX.accent2, accent3: HEX.accent3, accent4: HEX.accent4, accent5: HEX.accent5, accent6: HEX.accent6 };
const EQ = "+mj-lt"; // equations in the heading (serif) font
const FOOT = "Digital twins & virtual trials  ·  general model catalogue";

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.8, y: 1.55, w: 11.7, h: 0.45, fontSize: 16, bold: true, color: C.accent1, valign: "top" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.05, w: 11.7, h: 1.7, fontSize: 44, bold: true, color: C.background1, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.95, w: 10.5, h: 1.1, fontSize: 18, color: C.background2, valign: "top" }, text: "" } },
    { placeholder: { options: { name: "byline", type: "body", x: 0.8, y: 5.75, w: 11.7, h: 0.5, fontSize: 16, color: C.accent1, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.8, y: 2.25, w: 11.7, h: 0.45, fontSize: 16, bold: true, color: C.accent1, valign: "top" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.8, w: 11.7, h: 1.5, fontSize: 40, bold: true, color: C.background1, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.45, w: 10.5, h: 1.6, fontSize: 18, color: C.background2, valign: "top" }, text: "" } },
  ],
  slideNumber: { x: 12.3, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: C.background2, align: "right" },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.6, y: 0.32, w: 12.1, h: 0.36, fontSize: 13, bold: true, color: C.accent2, valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.66, w: 12.1, h: 0.8, fontSize: 36, bold: true, color: C.text2, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: 0.6, y: 6.98, w: 8, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 12.1, y: 6.98, w: 0.6, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});

// ------------------------------------------------------------------ helpers
async function icon(Comp, hex = "FFFFFF") {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

const TAGS = {
  code: { label: "IN CODE", fill: C.accent2, color: C.background1 },
  talk: { label: "IN TALKS", fill: C.accent1, color: C.text1 },
  review: { label: "IN REVIEW", fill: C.accent4, color: C.background1 },
  new: { label: "PROPOSED", fill: C.accent3, color: C.text1 },
};
function tag(slide, kind, x, y, name) {
  const t = TAGS[kind];
  slide.addText(t.label, {
    shape: pres.shapes.RECTANGLE, rectRadius: 0.06, x, y, w: 1.05, h: 0.28,
    fill: { color: t.fill }, color: t.color, fontSize: 10, bold: true, align: "center", valign: "middle",
    margin: 0, isTextBox: true, objectName: name || `tag ${t.label}`,
  });
}

function content(kicker, title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(kicker, { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  return s;
}

// Equation card: title, equation(s), description, status tag.
function eqCard(slide, o) {
  const { x, y, w, h, title, eq, desc, kind, eqSize = 16, descSize = 14, eqH } = o;
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `card ${title}`,
  });
  slide.addText(title, { x: x + 0.22, y: y + 0.15, w: w - 1.6, h: 0.38, fontSize: 16, bold: true, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: `card title ${title}` });
  if (kind) tag(slide, kind, x + w - 1.27, y + 0.18, `tag ${title}`);
  const eh = eqH || 0.8;
  const eqLines = Array.isArray(eq) ? eq : [eq];
  slide.addText(eqLines.map((t, i) => ({ text: t, options: { breakLine: i < eqLines.length - 1 } })), {
    x: x + 0.22, y: y + 0.6, w: w - 0.44, h: eh, fontSize: eqSize, fontFace: EQ, color: C.accent4, margin: 0.06,
    valign: "middle", fill: { color: C.background1 }, isTextBox: true, objectName: `equation ${title}`,
  });
  if (desc) {
    slide.addText(desc, { x: x + 0.22, y: y + 0.7 + eh, w: w - 0.44, h: h - eh - 0.82, fontSize: descSize, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: `desc ${title}` });
  }
}

function grid4(slide, cards, opts = {}) {
  const x0 = 0.6, y0 = opts.y0 || 1.7, gw = 5.9, gh = opts.h || 2.45, gap = 0.3, rgap = opts.rgap || 0.25;
  cards.forEach((c, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    eqCard(slide, { x: x0 + col * (gw + gap), y: y0 + row * (gh + rgap), w: gw, h: gh, ...c, ...(opts.card || {}) });
  });
}

function bullets(items, size = 14) {
  return items.map((t, i) => {
    if (Array.isArray(t)) return t.map((r, j) => ({ text: r.text, options: { ...r.options, fontSize: size, bullet: j === 0 ? true : undefined, breakLine: j === t.length - 1 && i < items.length - 1, paraSpaceAfter: 6 } }));
    return [{ text: t, options: { bullet: true, fontSize: size, breakLine: i < items.length - 1, paraSpaceAfter: 6 } }];
  }).flat();
}

function erf(x) { // Abramowitz-Stegun 7.1.26
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));

const CHART_TEXT = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt", catAxisTitleFontFace: "+mn-lt", valAxisTitleFontFace: "+mn-lt" };

(async () => {
  const IC = {
    users: await icon(fa.FaUsers), flask: await icon(fa.FaFlask), rad: await icon(fa.FaRadiation),
    dna: await icon(fa.FaDna), sync: await icon(fa.FaSyncAlt), brain: await icon(fa.FaBrain),
    image: await icon(fa.FaImage), chart: await icon(fa.FaChartLine), diagram: await icon(fa.FaProjectDiagram),
    robot: await icon(fa.FaRobot), dice: await icon(fa.FaDice), scale: await icon(fa.FaBalanceScale),
    cubes: await icon(fa.FaCubes), micro: await icon(fa.FaMicroscope), shield: await icon(fa.FaShieldAlt),
    target: await icon(fa.FaBullseye),
  };
  // Circles are drawn as PNG images: filled ovals and rounded rectangles lose their fill
  // in legacy .ppt exports and some viewers, while images and plain rectangles always render.
  const CIRC = {};
  for (const hx of [HEX.dk2, HEX.accent1, HEX.accent2, HEX.accent3, HEX.accent4]) {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"><circle cx="128" cy="128" r="127" fill="#${hx}"/></svg>`;
    CIRC[hx] = "image/png;base64," + (await sharp(Buffer.from(svg)).png().toBuffer()).toString("base64");
  }
  function iconCircle(slide, img, x, y, d = 0.6, fill = C.accent2) {
    slide.addImage({ data: CIRC[fill], x, y, w: d, h: d, objectName: "icon circle" });
    const p = d * 0.22;
    slide.addImage({ data: img, x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, objectName: "icon" });
  }

  // ================================================================ Title
  const SEC = (t) => { pres.addSection({ title: t }); return t; };
  function divider(sec, kicker, title, body, notes) {
    const d = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: sec });
    d.addText(kicker, { placeholder: "kicker" });
    d.addText(title, { placeholder: "title" });
    d.addText(body, { placeholder: "body" });
    d.addNotes(notes);
  }
  const CH = {
    valAxisLabelColor: HEX.accent6, catAxisLabelColor: HEX.accent6, valAxisLabelFontSize: 11, catAxisLabelFontSize: 11,
    valGridLine: { color: "E3E8EA", size: 0.75 }, catGridLine: { style: "none" }, titleFontSize: 13, titleColor: HEX.dk1,
    showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: HEX.dk1, lineDataSymbol: "none", lineSize: 2.5, ...CHART_TEXT,
  };

  let sec = SEC("Introduction");
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: sec });
  s.addText("GENERAL MODEL CATALOGUE  ·  OCTOBER 2026", { placeholder: "kicker" });
  s.addText("Mathematical Models for Digital Twins and Virtual Trials", { placeholder: "title" });
  s.addText("A field-independent reference: about 90 models across dynamics, physiology, statistics, data assimilation, uncertainty, surrogates, virtual populations, trial design, control and validation", { placeholder: "body" });
  s.addText("Hamid Abdollahi, PhD", { placeholder: "byline" });
  s.addNotes("This deck is a general reference. It is not tied to any one disease or therapy. Each slide gives the governing equation of a model family, what the symbols mean, and when to use it in a digital twin or a virtual trial.");

  // ---------------------------------------------------------------- family map
  s = content("OVERVIEW", "Twelve families of models", sec);
  const fams = [
    [IC.diagram, "1 · Digital-twin formalism", "state space, ODE, SDE, PDE, discrete"],
    [IC.flask, "2 · Pharmacokinetics", "compartments, absorption, PBPK"],
    [IC.target, "3 · Pharmacodynamics, QSP", "Emax, indirect response, kinetics"],
    [IC.dna, "4 · Disease dynamics", "growth, epidemics, progression"],
    [IC.users, "5 · Organ physiology", "heart, cells, glucose, lungs"],
    [IC.cubes, "6 · Continuum physics", "mechanics, flow, transport, heat"],
    [IC.chart, "7 · Statistical models", "mixed effects, GLM, survival"],
    [IC.sync, "8 · Data assimilation", "Kalman, particle, variational"],
    [IC.dice, "9 · Uncertainty, sensitivity", "Monte Carlo, Sobol, chaos"],
    [IC.brain, "10 · Surrogates", "POD, DMD, GP, neural"],
    [IC.scale, "11 · Virtual populations, trials", "cohorts, power, adaptive design"],
    [IC.shield, "12 · Control and validation", "MPC, RL, fit, credibility"],
  ];
  fams.forEach(([img, h, d], i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.6 + col * 4.1, y = 1.75 + row * 1.2;
    iconCircle(s, img, x, y + 0.06, 0.6);
    s.addText(h, { x: x + 0.78, y, w: 3.2, h: 0.42, fontSize: 16, bold: true, color: C.text2, margin: 0, valign: "top", isTextBox: true });
    s.addText(d, { x: x + 0.78, y: y + 0.44, w: 3.2, h: 0.5, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("The twelve families. Families 1 to 6 describe how a system behaves. Family 7 describes variability between and within individuals. Families 8 to 10 connect models to data and make them fast and honest about uncertainty. Families 11 and 12 turn twins into trials and decisions and prove they can be trusted.");

  // ---------------------------------------------------------------- loop diagram
  s = content("HOW THEY FIT TOGETHER", "The digital-twin loop and the virtual trial", sec);
  const loop = [["Physical system", "patient, organ, device"], ["Data y", "sensors, imaging, labs"], ["Assimilation", "θ, x ← y (Family 8)"], ["Virtual model", "x′ = f(x, u, θ) (1–7)"], ["Prediction", "what-if, with UQ (9–10)"], ["Decision u", "control (12)"]];
  loop.forEach(([a, b], i) => {
    const x = 0.6 + i * 2.06, y = 1.85;
    s.addText([{ text: a, options: { bold: true, fontSize: 14, color: C.background1, breakLine: true } }, { text: b, options: { fontSize: 12, color: C.background2 } }], {
      shape: pres.shapes.RECTANGLE, rectRadius: 0.1, x, y, w: 1.85, h: 1.2, fill: { color: i === 0 ? C.accent4 : C.text2 }, align: "center", valign: "middle", margin: 0.06, isTextBox: true, objectName: `loop ${a}`,
    });
    if (i < loop.length - 1) s.addShape(pres.shapes.LINE, { x: x + 1.86, y: y + 0.6, w: 0.19, h: 0, line: { color: C.accent6, width: 1.5, endArrowType: "triangle" }, objectName: "arrow" });
  });
  s.addShape(pres.shapes.LINE, { x: 11.83, y: 3.07, w: 0, h: 0.4, line: { color: C.accent1, width: 2 }, objectName: "loop down" });
  s.addShape(pres.shapes.LINE, { x: 1.52, y: 3.47, w: 10.31, h: 0, line: { color: C.accent1, width: 2 }, objectName: "loop across" });
  s.addShape(pres.shapes.LINE, { x: 1.52, y: 3.07, w: 0, h: 0.4, line: { color: C.accent1, width: 2, beginArrowType: "triangle" }, objectName: "loop up" });
  s.addText("Bidirectional link: the twin is re-synchronised with every new measurement, and its decisions act on the real system", { x: 1.7, y: 3.55, w: 10, h: 0.35, fontSize: 14, italic: true, color: C.text1, align: "center", margin: 0, isTextBox: true });
  [["Digital twin", "One individual. Model + data assimilation + prediction + decision, updated over time.", IC.sync],
   ["Virtual cohort", "Many twins sampled from population models (Families 7, 11): realistic, diverse, plausible.", IC.users],
   ["Virtual trial", "Every arm applied to every virtual subject; endpoints analysed with trial statistics.", IC.scale]].forEach(([h, d, img], i) => {
    const x = 0.6 + i * 4.1;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 4.25, w: 3.85, h: 2.4, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `card ${h}` });
    iconCircle(s, img, x + 0.22, 4.45, 0.55);
    s.addText(h, { x: x + 0.9, y: 4.5, w: 2.8, h: 0.45, fontSize: 17, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
    s.addText(d, { x: x + 0.22, y: 5.15, w: 3.45, h: 1.35, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("A digital twin is a model of one individual that is continuously synchronised with data from that individual and used to make decisions. A virtual trial runs many twins, or many sampled virtual subjects, through every arm of a study. The family numbers show where each model family sits in the loop.");

  // ================================================================ Family 1
  sec = SEC("Dynamics and physiology");
  divider(sec, "PART I", "Models of how the system behaves", "Families 1–6: dynamical formalism, pharmacology, disease, organ physiology and continuum physics", "Part I covers mechanistic models: equations that describe how a system changes over time and space.");

  s = content("FAMILY 1 · DIGITAL-TWIN FORMALISM", "Every twin is a state-space model", sec);
  grid4(s, [
    { title: "Discrete state-space model", eq: ["xₖ₊₁ = f(xₖ, uₖ, θ) + wₖ", "yₖ = h(xₖ) + vₖ"], eqH: 0.9, desc: "State x, control u, parameters θ, observations y; process noise w, measurement noise v. The general DT definition." },
    { title: "Ordinary differential equations", eq: "dx/dt = f(x(t), u(t), θ, t),   x(0) = x₀", desc: "Well-mixed systems: compartments, kinetics, organ function. Solved with Runge–Kutta or stiff (BDF) solvers." },
    { title: "Stochastic differential equations", eq: "dXₜ = f(Xₜ, θ) dt + g(Xₜ, θ) dWₜ", desc: "Adds intrinsic noise via Brownian motion W. Use when variability within an individual matters (e.g. cell counts, glucose)." },
    { title: "Partial differential equations", eq: "∂u/∂t = L(u; θ) + s(x, t)  in Ω,  + boundary conditions", eqSize: 15, desc: "Spatially resolved systems: tissue, flow, transport. Solved with finite elements, volumes or differences." },
  ]);
  s.addNotes("All digital twins can be written as a state-space model: a hidden state that evolves, and observations that depend on it. ODEs, SDEs and PDEs are the continuous versions, chosen by whether space and randomness matter.");

  s = content("FAMILY 1 · DISCRETE AND RULE-BASED MODELS", "When states are discrete or agents interact", sec);
  grid4(s, [
    { title: "Markov chain", eq: ["π(t + 1) = π(t) P,   Σⱼ Pᵢⱼ = 1", "continuous time: dπ/dt = π Q"], eqH: 0.9, desc: "Health states and transition probabilities. Basis of disease-progression and cost-effectiveness models." },
    { title: "Hidden Markov model", eq: "p(y₁:T) = Σ_z p(z₁) Πₜ p(zₜ | zₜ₋₁) p(yₜ | zₜ)", eqSize: 15, desc: "Unobserved disease stage z inferred from noisy observations y (forward–backward, Viterbi)." },
    { title: "Agent-based model", eq: "aᵢ(t + Δt) = R(aᵢ(t), neighbours, environment)", eqSize: 15, desc: "Individual cells, patients or devices follow rules; population behaviour emerges. Calibrate on summary statistics." },
    { title: "Cellular automata and hybrid systems", eq: ["sᵢⱼ(t + 1) = F(neighbourhood of sᵢⱼ(t))", "hybrid: dx/dt = f_q(x), q switches on events"], eqH: 0.9, eqSize: 15, desc: "Lattice rules for tissue; hybrid models mix continuous flow with discrete events (e.g. dosing)." },
  ]);
  s.addNotes("Not every twin is continuous. Markov models describe movement between discrete health states. Hidden Markov models infer an unobserved state. Agent-based and cellular-automaton models describe many interacting individuals. Hybrid systems combine continuous dynamics with discrete events such as a dose or a device switch.");

  // ================================================================ Family 2 PK
  s = content("FAMILY 2 · PHARMACOKINETICS", "What the body does to a drug", sec);
  grid4(s, [
    { title: "One compartment, IV bolus", eq: ["C(t) = (Dose / V) · e^(−k t),   k = CL / V", "t½ = ln2 / k,   AUC = Dose / CL"], eqH: 0.9, desc: "Volume V and clearance CL summarise distribution and elimination." },
    { title: "First-order absorption (Bateman)", eq: "C(t) = F·D·kₐ / (V(kₐ − k)) · (e^(−kt) − e^(−kₐt))", eqSize: 15, desc: "Oral dosing with bioavailability F and absorption rate kₐ. Gives Cmax and Tmax." },
    { title: "Two compartments", eq: ["dA₁/dt = −(k₁₀ + k₁₂) A₁ + k₂₁ A₂", "dA₂/dt = k₁₂ A₁ − k₂₁ A₂"], eqH: 0.9, desc: "Central and peripheral compartments; bi-exponential decline. Extends to n compartments." },
    { title: "PBPK (general)", eq: ["V_T dC_T/dt = Q_T (C_A − C_T / K_p,T) − CL_T C_T", "venous: Σ Q_T C_T / K_p,T"], eqH: 0.9, eqSize: 15, desc: "Organs linked by blood flow Q with partition coefficients K_p. Scales across species and populations." },
  ]);
  s.addNotes("Pharmacokinetic models describe concentration over time. Compartment models are empirical and fast. PBPK models are physiological, with real organ volumes and blood flows, so they extrapolate to new populations, which makes them central to virtual trials.");

  // ================================================================ Family 3 PD
  s = content("FAMILY 3 · PHARMACODYNAMICS", "What the drug does to the body", sec);
  eqCard(s, { x: 0.6, y: 1.7, w: 5.9, h: 2.45, title: "Emax / Hill model", eq: ["E = E₀ + E_max · Cⁿ / (EC₅₀ⁿ + Cⁿ)"], eqH: 0.55, desc: "Saturating concentration–effect curve. Hill coefficient n sets steepness; EC₅₀ the potency. n = 1 is the classic Emax model." });
  eqCard(s, { x: 0.6, y: 4.35, w: 5.9, h: 2.35, title: "Indirect response and effect site", eq: ["dR/dt = k_in (1 − I(C)) − k_out R", "dC_e/dt = k_e0 (C − C_e)"], eqH: 0.9, desc: "Drug alters production or loss of a response R; effect compartment C_e explains hysteresis." });
  {
    const xs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 20];
    const hill = (n) => xs.map((c) => +(Math.pow(c, n) / (1 + Math.pow(c, n))).toFixed(3));
    s.addChart(pres.charts.LINE, [1, 2, 4].map((n) => ({ name: `n = ${n}`, labels: xs.map(String), values: hill(n) })), {
      x: 6.8, y: 1.7, w: 5.9, h: 4.6, chartColors: [HEX.accent4, HEX.accent2, HEX.accent3], showTitle: true, title: "Hill curves: E / E_max vs C / EC₅₀ (log-spaced)",
      valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0%", ...CH, objectName: "Hill chart",
    });
    s.addText("All curves cross 50 % at C = EC₅₀; larger n gives a sharper switch.", { x: 6.8, y: 6.35, w: 5.9, h: 0.4, fontSize: 12, italic: true, color: C.accent6, margin: 0, isTextBox: true });
  }
  s.addNotes("The Hill or Emax model is the workhorse exposure–response model. Indirect-response models capture delayed effects when a drug changes the turnover of a biomarker. The effect-compartment model explains why effect lags behind plasma concentration.");

  s = content("FAMILY 3 · QUANTITATIVE SYSTEMS PHARMACOLOGY", "Biochemical networks as equations", sec);
  grid4(s, [
    { title: "Mass-action kinetics", eq: ["A + B ⇌ C:  dC/dt = k_f [A][B] − k_r [C]", "network: dx/dt = S · v(x, θ)"], eqH: 0.9, desc: "Stoichiometry matrix S times reaction rates v. The general QSP and systems-biology form." },
    { title: "Michaelis–Menten", eq: "v = V_max [S] / (K_m + [S])", desc: "Saturable enzyme kinetics or transport; also saturable (nonlinear) drug elimination." },
    { title: "Receptor occupancy", eq: ["L + R ⇌ LR,   K_d = k_off / k_on", "occupancy = [L] / (K_d + [L])"], eqH: 0.9, desc: "Binding drives effect; target-mediated drug disposition extends this to drug loss by binding." },
    { title: "Logical (Boolean) networks", eq: "xᵢ(t + 1) = Fᵢ(x_inputs(t)),   xᵢ ∈ {0, 1}", desc: "Signalling pathways when rate constants are unknown; finds attractors (cell fates) and drug targets." },
  ]);
  s.addNotes("Quantitative systems pharmacology models write a biological network as reaction equations. Mass action and Michaelis–Menten kinetics give continuous models; Boolean networks give a coarse logic model when kinetic data are missing.");

  // ================================================================ Family 4 disease
  s = content("FAMILY 4 · DISEASE DYNAMICS", "Growth models and treatment effect", sec);
  eqCard(s, { x: 0.6, y: 1.7, w: 5.9, h: 2.45, title: "Growth laws", eq: ["exponential: dN/dt = rN", "logistic: dN/dt = rN(1 − N/K)", "Gompertz: dN/dt = rN ln(K/N)"], eqH: 1.15, eqSize: 15, desc: "Tumours, bacteria, cell cultures; K = carrying capacity." });
  eqCard(s, { x: 0.6, y: 4.35, w: 5.9, h: 2.35, title: "Treatment effect (log-kill, TGI)", eq: ["dN/dt = g(N) − k_d · E(C(t)) · N", "Norton–Simon: kill ∝ growth rate g(N)/N"], eqH: 0.9, eqSize: 15, desc: "Drug kill proportional to exposure; tumour-growth-inhibition models (e.g. Simeoni) add delay compartments." });
  {
    const ts = []; for (let t = 0; t <= 30; t += 1) ts.push(t);
    const N0 = 0.05, K = 1, r = 0.3, b = 0.15;
    const TL = ts.map((t) => (t % 5 === 0 ? String(t) : ""));
    const ex = ts.map((t) => +(N0 * Math.exp(r * t)).toFixed(3));
    const lg = ts.map((t) => +(K / (1 + (K / N0 - 1) * Math.exp(-r * t))).toFixed(3));
    const gz = ts.map((t) => +(K * Math.exp(Math.log(N0 / K) * Math.exp(-b * t))).toFixed(3));
    s.addChart(pres.charts.LINE, [{ name: "Exponential", labels: TL, values: ex }, { name: "Logistic", labels: TL, values: lg }, { name: "Gompertz", labels: TL, values: gz }], {
      x: 6.8, y: 1.7, w: 5.9, h: 4.6, chartColors: [HEX.accent3, HEX.accent2, HEX.accent4], showTitle: true, title: "Growth from N₀ = 0.05 K (illustrative)",
      valAxisMinVal: 0, valAxisMaxVal: 1.2, valAxisMajorUnit: 0.2, catAxisLabelFrequency: 5, ...CH, objectName: "growth chart",
    });
    s.addText("Exponential growth is unbounded; logistic and Gompertz saturate at K. Fit all three and compare by AIC.", { x: 6.8, y: 6.35, w: 5.9, h: 0.4, fontSize: 12, italic: true, color: C.accent6, margin: 0, isTextBox: true });
  }
  s.addNotes("Growth laws describe untreated disease burden. Adding a kill term driven by drug exposure gives tumour-growth-inhibition and antimicrobial models. Which law fits best is an empirical question settled with model-selection criteria.");

  s = content("FAMILY 4 · POPULATION SPREAD AND PROGRESSION", "Epidemics and chronic disease over time", sec);
  grid4(s, [
    { title: "SIR / SEIR epidemic model", eq: ["dS/dt = −βSI/N,  dI/dt = βSI/N − γI", "dR/dt = γI,   R₀ = β / γ"], eqH: 0.9, desc: "Compartments of susceptible, infected, recovered; SEIR adds an exposed stage. Vaccination and contact changes as controls." },
    { title: "Disease-progression model", eq: "S(t) = S₀ + α t + E_drug(t)  (symptomatic or protective)", eqSize: 15, desc: "Natural decline of a clinical score; drug either offsets the score or slows the slope α." },
    { title: "Multistate (illness–death)", eq: "λₕⱼ(t | x) = λₕⱼ,₀(t) exp(βₕⱼᵀ x)", desc: "Transition hazards between healthy, ill and dead states; generalises survival analysis." },
    { title: "Pharmacometric time to event", eq: "S(t) = exp(−∫₀ᵗ h₀(s) e^(β·E(s)) ds)", desc: "Event hazard driven by a model-predicted, time-varying exposure or biomarker E(s)." },
  ]);
  s.addNotes("These models describe how disease spreads through a population or progresses within a person. They provide the natural-history arm of a virtual trial, against which treatment effects are simulated.");

  // ================================================================ Family 5 organ physiology
  s = content("FAMILY 5 · ORGAN PHYSIOLOGY", "Classic physiological twins", sec);
  grid4(s, [
    { title: "Windkessel (circulation)", eq: ["C dP/dt = Q(t) − P / R", "3-element: adds characteristic impedance Z"], eqH: 0.9, desc: "Arterial pressure P from inflow Q, compliance C and resistance R. Lumped 0-D cardiovascular twin." },
    { title: "Cell excitation (Hodgkin–Huxley)", eq: ["C_m dV/dt = −Σ gᵢ(V, t)(V − Eᵢ) + I", "FitzHugh–Nagumo: 2-variable reduction"], eqH: 0.9, desc: "Action potentials from ion-channel conductances; basis of cardiac and neural electrophysiology twins." },
    { title: "Glucose–insulin (Bergman minimal)", eq: ["dG/dt = −(p₁ + X) G + p₁ G_b", "dX/dt = −p₂ X + p₃ (I − I_b)"], eqH: 0.9, desc: "Insulin sensitivity S_I = p₃/p₂ from a glucose tolerance test; artificial-pancreas twins." },
    { title: "Respiratory mechanics", eq: "P_aw(t) = R · Q(t) + V(t) / C + PEEP", desc: "Airway pressure from flow, volume, resistance R and compliance C; ventilator-setting twins." },
  ]);
  s.addNotes("These are the classic lumped physiological models. Each has few parameters, which can be identified per patient from routine measurements, making them ideal first digital twins.");

  // ================================================================ Family 6 continuum
  s = content("FAMILY 6 · CONTINUUM PHYSICS", "Space-resolved mechanics, flow and transport", sec);
  grid4(s, [
    { title: "Solid mechanics (FEM)", eq: ["∇·σ + f = 0,   σ = ∂W/∂ε", "Neo-Hookean: W = μ/2 (I₁ − 3);  FEM: K u = F"], eqH: 0.9, desc: "Tissue, bone, implants and devices. Hyperelastic laws for soft tissue." },
    { title: "Fluid flow (Navier–Stokes)", eq: ["ρ(∂v/∂t + v·∇v) = −∇p + μ∇²v,  ∇·v = 0", "porous tissue (Darcy): v = −(k/μ) ∇p"], eqH: 0.9, eqSize: 15, desc: "Blood flow, airflow, CSF; fluid–structure interaction couples it to mechanics." },
    { title: "Advection–diffusion–reaction", eq: "∂c/∂t + v·∇c = ∇·(D∇c) + R(c)", desc: "Drug, oxygen or nutrient transport in tissue; reaction–diffusion for invading cells (Fisher–KPP)." },
    { title: "Heat transfer (Pennes bioheat)", eq: "ρc ∂T/∂t = ∇·(k∇T) + ω_b c_b (T_a − T) + Q", eqSize: 15, desc: "Tissue temperature with blood perfusion ω_b; ablation, hyperthermia and device-safety twins." },
  ]);
  s.addNotes("Continuum models resolve space as well as time. They are solved with finite-element or finite-volume methods and are the most expensive part of many twins, which is why surrogates matter later.");

  // ================================================================ Part II statistics
  sec = SEC("Statistics and data");
  divider(sec, "PART II", "Models of variability and of the data", "Families 7–10: population statistics, data assimilation, uncertainty and sensitivity, and surrogate models", "Part II covers the models that connect equations to data: variability between people, synchronising twins with measurements, quantifying uncertainty, and making models fast.");

  s = content("FAMILY 7 · POPULATION MODELS", "Variability between and within individuals", sec);
  grid4(s, [
    { title: "Nonlinear mixed effects (NLME)", eq: ["y_ij = f(t_ij, θᵢ) + ε_ij,   ε ~ N(0, σ²)", "θᵢ = θ_pop · exp(β·covᵢ + ηᵢ),  η ~ N(0, Ω)"], eqH: 0.9, eqSize: 15, desc: "The core population model: typical value, covariate effects, between-subject variability Ω." },
    { title: "Linear mixed model", eq: "y = Xβ + Zb + ε,   b ~ N(0, G),  ε ~ N(0, R)", desc: "Fixed effects β and random effects b for repeated measures; longitudinal biomarkers." },
    { title: "Generalised linear models", eq: ["g(E[y]) = Xβ;  logistic: p = 1 / (1 + e^(−Xβ))", "Poisson: log λ = Xβ"], eqH: 0.9, desc: "Binary, count and rate outcomes; dose–response with logit or probit links." },
    { title: "Bayesian hierarchical model", eq: ["p(θ, φ | y) ∝ Πᵢ p(yᵢ | θᵢ) p(θᵢ | φ) · p(φ)", "partial pooling between individuals"], eqH: 0.9, eqSize: 15, desc: "Borrow strength across patients, sites or studies; priors encode prior knowledge." },
  ]);
  s.addNotes("Population models separate the typical behaviour from variability between individuals and residual noise. NLME is the standard in pharmacometrics; mixed and generalised linear models are the statistical equivalents; hierarchical Bayes generalises all of them.");

  s = content("FAMILY 7 · TIME-TO-EVENT MODELS", "Survival and event endpoints", sec);
  grid4(s, [
    { title: "Kaplan–Meier estimator", eq: "Ŝ(t) = Π_{tᵢ ≤ t} (1 − dᵢ / nᵢ)", desc: "Non-parametric survival with censoring; dᵢ events among nᵢ at risk. Compared with the log-rank test." },
    { title: "Cox proportional hazards", eq: "h(t | x) = h₀(t) · exp(βᵀx),   HR = e^β", desc: "Covariate effects as hazard ratios without specifying h₀; check proportional hazards." },
    { title: "Parametric survival (Weibull, AFT)", eq: ["S(t) = exp(−(λt)ᵏ)", "AFT: log T = βᵀx + σε"], eqH: 0.9, desc: "Needed to extrapolate beyond follow-up and to simulate event times in virtual trials." },
    { title: "Joint longitudinal–survival model", eq: ["yᵢ(t) = mᵢ(t) + ε,   hᵢ(t) = h₀(t) e^(γᵀw + α mᵢ(t))"], eqH: 0.55, eqSize: 15, desc: "Links a biomarker trajectory mᵢ(t) to event risk; dynamic individual predictions for twins." },
  ]);
  s.addNotes("Most trial endpoints are times to events. Kaplan–Meier describes them, Cox explains them, parametric models extrapolate and simulate them, and joint models link them to a twin's evolving biomarkers.");

  // ================================================================ Family 8 data assimilation
  s = content("FAMILY 8 · DATA ASSIMILATION", "Keeping the twin synchronised with reality", sec);
  grid4(s, [
    { title: "Kalman filter (linear Gaussian)", eq: ["predict: x⁻ = A x̂,  P⁻ = A P Aᵀ + Q", "update: K = P⁻Hᵀ(HP⁻Hᵀ + R)⁻¹,  x̂ = x⁻ + K(y − Hx⁻)"], eqH: 0.9, eqSize: 14, desc: "Optimal state estimate; gain K weighs model against measurement uncertainty." },
    { title: "Extended / unscented / ensemble KF", eq: ["EKF: linearise f, h with Jacobians", "EnKF: P ≈ sample covariance of N members"], eqH: 0.9, eqSize: 15, desc: "Nonlinear and high-dimensional systems; also estimates parameters by augmenting the state." },
    { title: "Particle filter", eq: ["xₖ⁽ⁱ⁾ ~ p(xₖ | xₖ₋₁⁽ⁱ⁾),   wₖ⁽ⁱ⁾ ∝ wₖ₋₁⁽ⁱ⁾ p(yₖ | xₖ⁽ⁱ⁾)", "resample when N_eff = 1 / Σ w² is small"], eqH: 0.9, eqSize: 14, desc: "Sequential Monte Carlo; any nonlinearity or noise distribution." },
    { title: "Variational assimilation (4D-Var)", eq: "J(x₀) = ‖x₀ − x_b‖²_{B⁻¹} + Σₖ ‖yₖ − h(xₖ)‖²_{R⁻¹}", eqSize: 14, desc: "Fit the initial state over a time window by optimisation with adjoints; weather-forecast heritage." },
  ]);
  s.addNotes("Data assimilation is what turns a model into a twin. Filters update the state sequentially as data arrive; variational methods fit a whole window at once. Both weigh model uncertainty against measurement uncertainty.");

  s = content("FAMILY 8 · CALIBRATION AND IDENTIFIABILITY", "Estimating parameters, and knowing when you can", sec);
  grid4(s, [
    { title: "Least squares and likelihood", eq: ["θ̂ = argmin Σ (yᵢ − f(tᵢ, θ))² / σᵢ²", "MLE: θ̂ = argmax log p(y | θ)"], eqH: 0.9, desc: "Standard fitting; weighted least squares equals MLE for Gaussian noise." },
    { title: "Bayesian inference (MCMC)", eq: ["p(θ | y) ∝ p(y | θ) p(θ)", "Metropolis: accept θ′ with min(1, p(θ′|y) / p(θ|y))"], eqH: 0.9, eqSize: 15, desc: "Full posterior for parameters and predictions; Hamiltonian MC (Stan) for many parameters." },
    { title: "Fisher information, Cramér–Rao", eq: ["FIM = Jᵀ Σ⁻¹ J,   J = ∂f/∂θ", "Var(θ̂ⱼ) ≥ [FIM⁻¹]ⱼⱼ"], eqH: 0.9, desc: "Practical identifiability and optimal experimental design (D-optimal: maximise det FIM)." },
    { title: "Approximate Bayesian computation", eq: "accept θ if ρ(s(y_sim(θ)), s(y_obs)) < ε", desc: "Likelihood-free calibration with summary statistics s; for agent-based and stochastic models." },
  ]);
  s.addNotes("Calibration finds parameter values; identifiability asks whether the data can determine them at all. Fisher information answers that before collecting data and designs better experiments. ABC and simulation-based inference handle models whose likelihood cannot be written down.");

  // ================================================================ Family 9 UQ + sensitivity
  s = content("FAMILY 9 · UNCERTAINTY QUANTIFICATION", "Propagating what we do not know", sec);
  grid4(s, [
    { title: "Monte Carlo propagation", eq: ["Y⁽ᵏ⁾ = g(X⁽ᵏ⁾),  X⁽ᵏ⁾ ~ p(X),  k = 1…N", "error of the mean ∝ σ / √N"], eqH: 0.9, desc: "Simple and general; Latin hypercube sampling reduces variance for the same N." },
    { title: "Polynomial chaos expansion", eq: "Y ≈ Σ_α c_α Ψ_α(ξ),   c_α = E[Y Ψ_α] / E[Ψ_α²]", desc: "Orthogonal polynomials in random inputs ξ; mean and variance come straight from the coefficients." },
    { title: "Bayesian calibration (Kennedy–O'Hagan)", eq: "y_obs(x) = η(x, θ) + δ(x) + ε", desc: "Simulator η, model discrepancy δ (a GP) and noise ε, so the model is not assumed perfect." },
    { title: "Aleatory vs epistemic", eq: "Var(Y) = E_θ[Var(Y | θ)] + Var_θ(E[Y | θ])", desc: "Law of total variance splits natural variability from lack of knowledge about θ." },
  ]);
  s.addNotes("Uncertainty quantification makes predictions honest. Monte Carlo is the default; polynomial chaos is faster for smooth models; Kennedy–O'Hagan calibration admits that the model itself is imperfect. The total-variance split tells you whether more data would help.");

  s = content("FAMILY 9 · SENSITIVITY ANALYSIS", "Which inputs drive the output?", sec);
  grid4(s, [
    { title: "Local (normalised) sensitivity", eq: "Sⱼ = (∂y / ∂θⱼ) · (θⱼ / y)", desc: "Percent change in output per percent change in a parameter, at one point. Cheap but local." },
    { title: "Morris elementary effects", eq: "EEⱼ = [y(θ + Δeⱼ) − y(θ)] / Δ;   μ*ⱼ, σⱼ", desc: "Global screening with few runs: μ* ranks importance, σ flags nonlinearity and interactions." },
    { title: "Sobol variance indices", eq: ["Sⱼ = Var(E[Y | Xⱼ]) / Var(Y)", "S_Tj = E[Var(Y | X₋ⱼ)] / Var(Y)"], eqH: 0.9, desc: "First-order and total effects; the reference global method (Saltelli estimators)." },
    { title: "PRCC", eq: "ρ(rank Xⱼ, rank Y | other inputs)", desc: "Partial rank correlation with Latin hypercube sampling; monotonic relations, common in biology." },
  ]);
  s.addNotes("Sensitivity analysis shows which parameters matter, so you know what to measure per patient and what can be fixed at population values. Screen with Morris, then quantify with Sobol.");

  // ================================================================ Family 10 surrogates
  s = content("FAMILY 10 · SURROGATES AND REDUCED-ORDER MODELS", "Making twins fast enough to use", sec);
  grid4(s, [
    { title: "POD / PCA reduced basis", eq: ["U = Φ Σ Vᵀ (SVD of snapshots)", "u(x, t) ≈ Σᵢ₌₁ʳ aᵢ(t) φᵢ(x),  r ≪ n"], eqH: 0.9, desc: "Project PDE onto a few dominant modes; Galerkin ROMs run orders of magnitude faster." },
    { title: "Dynamic mode decomposition", eq: "xₖ₊₁ ≈ A xₖ,   A = X′ X⁺ ;  modes = eig(A)", desc: "Linear dynamics learned from data; growth rates and frequencies of each mode." },
    { title: "Gaussian-process emulator", eq: ["μ(x*) = k*ᵀ (K + σ²I)⁻¹ y", "σ²(x*) = k** − k*ᵀ (K + σ²I)⁻¹ k*"], eqH: 0.9, desc: "Surrogate with its own error bar from ~10 runs per input; drives calibration and design." },
    { title: "Neural surrogates", eq: ["PINN: L = L_data + λ ‖r(u_θ)‖²  (r = PDE residual)", "operator: G_θ : input field → solution field"], eqH: 0.9, desc: "Neural networks, physics-informed nets, neural ODEs and neural operators for large data." },
  ]);
  s.addNotes("A twin used at the bedside, or run thousands of times in a virtual trial, must be fast. Reduced-order models and emulators replace the full simulator with a cheap approximation that keeps its accuracy where it matters, and ideally reports its own error.");

  // ================================================================ Part III trials
  sec = SEC("Virtual trials and decisions");
  divider(sec, "PART III", "From twins to trials and decisions", "Families 11–12: generating virtual populations, designing and analysing virtual trials, optimal control and validation", "Part III covers how twins become virtual cohorts and trials, how they inform decisions, and how they are validated.");

  s = content("FAMILY 11 · VIRTUAL POPULATIONS", "Generating realistic virtual subjects", sec);
  grid4(s, [
    { title: "Sampling covariates with copulas", eq: ["F(x₁…x_d) = C(F₁(x₁), …, F_d(x_d))", "(Sklar's theorem)"], eqH: 0.9, desc: "Correct marginals plus correct correlations (age, weight, renal function, biomarkers)." },
    { title: "Plausible-patient selection", eq: ["θ ~ prior box;  keep if model output ∈ physiological range", "weight to match observed distribution"], eqH: 0.9, eqSize: 14, desc: "Accept–reject or prevalence weighting builds virtual populations matching real data (Allen et al. 2016)." },
    { title: "Parameter posterior sampling", eq: "θ⁽ᵏ⁾ ~ p(θ | data)  via MCMC / ABC / SBI", desc: "Each draw is one virtual subject consistent with observed data and its uncertainty." },
    { title: "Deep generative models", eq: ["VAE: max E_q[log p(x|z)] − KL(q(z|x) ‖ p(z))", "GAN / diffusion for images and records"], eqH: 0.9, eqSize: 15, desc: "Synthetic anatomies and records; check realism, coverage of rare cases and privacy." },
  ]);
  s.addNotes("A virtual trial is only as representative as its virtual population. Copulas keep correlations between covariates; plausible-patient methods ensure each subject is physiologically valid; posterior sampling and generative models scale up the cohort.");

  s = content("FAMILY 11 · VIRTUAL TRIAL DESIGN", "Estimands, power and adaptive designs", sec);
  grid4(s, [
    { title: "Potential outcomes", eq: ["ATE = E[Y(1) − Y(0)];  ITE ᵢ = Yᵢ(1) − Yᵢ(0)", "twins observe both arms for each subject"], eqH: 0.9, eqSize: 15, desc: "Virtual subjects receive every arm, so individual effects are directly computable." },
    { title: "Sample size and power", eq: ["parallel: n = 2(z₁₋α/₂ + z₁₋β)² σ² / δ²", "paired: n = (z₁₋α/₂ + z₁₋β)² σ_Δ² / δ²"], eqH: 0.9, eqSize: 15, desc: "Simulated trials estimate power directly: the fraction of replicate trials that succeed." },
    { title: "Bayesian adaptive design", eq: ["stop if P(δ > 0 | data) > 0.975", "Thompson: assign arm a with P(a is best)"], eqH: 0.9, desc: "Interim decisions and response-adaptive randomisation; operating characteristics found by simulation." },
    { title: "Dose-finding (CRM)", eq: "P(tox | dᵢ) = pᵢ^exp(a);  next dose ≈ target probability", desc: "Continual reassessment method updates the dose–toxicity curve after each cohort." },
  ]);
  s.addNotes("Virtual trials use the same statistics as real trials, but can also see both potential outcomes for each subject. Simulation is also how real adaptive designs are evaluated: run thousands of virtual trials and count how often each decision is right.");

  s = content("FAMILY 11 · TWINS INSIDE REAL TRIALS", "Augmenting, not replacing, clinical evidence", sec);
  grid4(s, [
    { title: "Prognostic covariate adjustment", eq: "Yᵢ = β₀ + τ Tᵢ + β₁ m̂ᵢ + εᵢ   (PROCOVA)", desc: "Twin-predicted control outcome m̂ᵢ as covariate; smaller variance, fewer patients for the same power." },
    { title: "Synthetic / external control arm", eq: ["e(x) = P(T = 1 | x),  weights 1/e(x), 1/(1 − e(x))", "ATE_IPTW = mean weighted difference"], eqH: 0.9, eqSize: 15, desc: "Propensity scores balance external or simulated controls against treated patients." },
    { title: "Model observers for imaging trials", eq: ["CHO: w = K⁻¹ Δs̄,   SNR² = Δs̄ᵀ K⁻¹ Δs̄", "AUC = Φ(SNR / √2)"], eqH: 0.9, desc: "Channelised Hotelling observer replaces readers in virtual imaging trials (e.g. VICTRE)." },
    { title: "Health-economic model", eq: ["ICER = (C₁ − C₀) / (QALY₁ − QALY₀)", "QALY = Σₜ uₜ · Δt · (1 + r)⁻ᵗ"], eqH: 0.9, desc: "Markov cohort over health states, with utilities u and discounting r; value of the intervention." },
  ]);
  s.addNotes("The most accepted uses of digital twins today augment real trials. Prognostic covariate adjustment, recognised by regulators, uses twin predictions to shrink trial size. Synthetic control arms and imaging model observers replace parts of the evidence. Health-economic models carry results into value decisions.");

  // ================================================================ Family 12 control
  s = content("FAMILY 12 · OPTIMISATION AND CONTROL", "Using the twin to decide", sec);
  grid4(s, [
    { title: "Optimal control", eq: ["min_u ∫₀ᵀ L(x, u) dt + Φ(x(T))  s.t. dx/dt = f(x, u)", "Pontryagin: H = L + λᵀ f,  ∂H/∂u = 0"], eqH: 0.9, eqSize: 14, desc: "Best dosing or therapy schedule over a horizon, with toxicity and resource constraints." },
    { title: "Model predictive control", eq: "min Σₖ₌₀ᴺ⁻¹ ‖xₖ − x_ref‖²_Q + ‖uₖ‖²_R  ; apply u₀, repeat", eqSize: 14, desc: "Re-optimise at every step with the updated twin (e.g. artificial pancreas)." },
    { title: "Reinforcement learning", eq: "V*(s) = max_a [ r(s, a) + γ E V*(s′) ]   (Bellman)", eqSize: 15, desc: "Learn a policy by interacting with the twin; adaptive treatment strategies. Train in silico only." },
    { title: "Bayesian and multi-objective optimisation", eq: ["x_next = argmax EI(x) on a GP surrogate", "Pareto: no objective improves without another worsening"], eqH: 0.9, eqSize: 14, desc: "Few expensive evaluations; efficacy vs toxicity vs cost trade-offs." },
  ]);
  s.addNotes("Decision models close the loop. Optimal control and MPC compute schedules from the equations; reinforcement learning learns them from simulation; Bayesian and multi-objective optimisation search designs when evaluations are expensive.");

  // ================================================================ Family 12 validation
  s = content("FAMILY 12 · MODEL FIT AND SELECTION", "Does the model describe the data?", sec);
  grid4(s, [
    { title: "Error metrics", eq: ["RMSE = √(mean (y − ŷ)²),  MAE = mean |y − ŷ|", "R² = 1 − SS_res / SS_tot"], eqH: 0.9, desc: "Accuracy of predictions; report on held-out data, not the fitting data." },
    { title: "Information criteria", eq: ["AIC = 2k − 2 ln L̂", "BIC = k ln n − 2 ln L̂"], eqH: 0.9, desc: "Trade fit against complexity k; choose between growth laws, compartments or covariates." },
    { title: "Cross-validation", eq: "CV = (1/K) Σₖ loss(y_k, f̂₋ₖ(x_k))", desc: "K-fold or leave-one-subject-out; nested CV when tuning hyper-parameters too." },
    { title: "Predictive checks", eq: ["posterior predictive: y_rep ~ p(y | θ), θ ~ p(θ | y)", "VPC: observed percentiles inside simulated bands"], eqH: 0.9, eqSize: 14, desc: "Simulate new data from the fitted model and compare to what was observed." },
  ]);
  s.addNotes("Fit and selection metrics answer whether the model describes the data, and which of several candidate models to prefer. Predictive checks are the most informative: if data simulated from the model do not look like the real data, the model is wrong somewhere.");

  s = content("FAMILY 12 · VERIFICATION, VALIDATION, CREDIBILITY", "When can a twin be trusted?", sec);
  grid4(s, [
    { title: "Verification (solve equations right)", eq: ["observed order p = ln(e₁ / e₂) / ln r", "Richardson: f_exact ≈ f₁ + (f₁ − f₂) / (rᵖ − 1)"], eqH: 0.9, desc: "Mesh and time-step convergence; code tested against analytical solutions." },
    { title: "Validation (solve right equations)", eq: "E = S − D,   compare |E| with u_val = √(u_num² + u_input² + u_D²)", eqSize: 14, desc: "Simulation S vs experiment D, judged against combined uncertainty (ASME V&V 20)." },
    { title: "Probabilistic calibration", eq: ["coverage = P(y ∈ 95 % prediction interval) ≈ 0.95", "PIT values ~ Uniform(0, 1)"], eqH: 0.9, desc: "Are the twin's uncertainty bands honest? Check on new subjects." },
    { title: "Risk-informed credibility", eq: "evidence needed ∝ model influence × decision consequence", eqSize: 15, desc: "ASME V&V 40 and FDA guidance: define context of use, then scale verification, validation and UQ." },
  ]);
  s.addNotes("Verification checks the equations are solved correctly; validation checks they are the right equations; calibration of uncertainty checks the error bars are honest. The V&V 40 framework decides how much of each is enough for a given decision.");

  // ================================================================ choosing guide
  s = content("CHOOSING A MODEL", "Match the question to the model family", sec);
  {
    const H = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 13 } });
    const R = (a, b, c) => [{ text: a, options: { bold: true, color: C.text2 } }, { text: b }, { text: c, options: { color: C.accent6 } }];
    const rows = [
      [H("Question"), H("Start with"), H("Upgrade to")],
      R("How does exposure change over time?", "Compartmental PK", "PBPK, TMDD"),
      R("How does exposure produce effect?", "Emax / Hill, indirect response", "QSP network"),
      R("How does disease evolve?", "Logistic / Gompertz, SIR, slope model", "Reaction–diffusion, agent-based"),
      R("How do individuals differ?", "Mixed effects (NLME, LMM)", "Hierarchical Bayes, copula cohorts"),
      R("How do I keep the twin current?", "Kalman / EKF", "Particle filter, 4D-Var, SBI"),
      R("How sure is the prediction?", "Monte Carlo + Morris", "Sobol, polynomial chaos, KOH"),
      R("Too slow to run thousands of times?", "GP emulator, POD", "Neural operator, PINN"),
      R("Does the treatment work in a virtual trial?", "Paired design, power by simulation", "Adaptive design, PROCOVA"),
      R("What should be done next?", "Optimal control", "MPC, reinforcement learning"),
    ];
    s.addTable(rows, { x: 0.6, y: 1.7, w: 12.1, colW: [4.4, 4.1, 3.6], fontSize: 13, color: C.text1, border: { type: "solid", pt: 0.75, color: "D5DEE0" }, fill: { color: C.background1 }, rowH: 0.47, valign: "middle", margin: [0.03, 0.08, 0.03, 0.08], objectName: "choosing table" });
  }
  s.addNotes("A practical guide: start with the simplest model that answers the question and has identifiable parameters; upgrade only when validation shows the simple model fails.");

  // ================================================================ Conclusion
  sec = SEC("Conclusion");
  s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: sec });
  s.addText("TAKEAWAYS", { placeholder: "kicker" });
  s.addText("Mechanism, data, uncertainty, decision", { placeholder: "title" });
  s.addText([
    { text: "A twin is a state-space model (Families 1–6) kept current by data assimilation (8).", options: { bullet: true, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Population, UQ and surrogate models (7, 9, 10) turn one twin into a fast, honest virtual cohort.", options: { bullet: true, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Trial, control and V&V models (11, 12) turn the cohort into credible decisions.", options: { bullet: true } },
  ], { placeholder: "body" });
  s.addNotes("Three messages: the core of every twin is a dynamical model synchronised with data; population, uncertainty and surrogate models make it scale to virtual trials; trial design, control and validation make the results usable and trustworthy.");

  s = content("KEY REFERENCES", "Classic sources for the models", sec);
  const rL = [
    "Gabrielsson J, Weiner D. Pharmacokinetic and Pharmacodynamic Data Analysis. 5th ed. 2016",
    "Owen JS, Fiedler-Kelly J. Introduction to Population PK/PD Analysis with NONMEM. 2014",
    "Keener J, Sneyd J. Mathematical Physiology. 2nd ed. Springer 2009",
    "Murray JD. Mathematical Biology I & II. Springer 2002–03",
    "Kalman RE. A new approach to linear filtering. J Basic Eng 1960",
    "Särkkä S. Bayesian Filtering and Smoothing. Cambridge 2013",
    "Kennedy MC, O'Hagan A. Bayesian calibration of computer models. JRSS B 2001",
    "Saltelli A et al. Global Sensitivity Analysis: The Primer. Wiley 2008",
  ];
  const rR = [
    "Smith RC. Uncertainty Quantification. SIAM 2014",
    "Rasmussen CE, Williams CKI. Gaussian Processes for ML. MIT Press 2006",
    "Allen RJ et al. Efficient generation of virtual populations. CPT:PSP 2016",
    "Berry SM et al. Bayesian Adaptive Methods for Clinical Trials. CRC 2010",
    "Schuler A et al. Prognostic covariate adjustment (PROCOVA). Int J Biostat 2022",
    "Badano A et al. VICTRE in silico imaging trial. JAMA Netw Open 2018",
    "ASME V&V 40-2018; FDA guidance on computational model credibility 2023",
    "NASEM. Foundational Research Gaps for Digital Twins. 2024",
  ];
  s.addText(bullets(rL, 13), { x: 0.6, y: 1.7, w: 5.95, h: 4.9, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addText(bullets(rR, 13), { x: 6.75, y: 1.7, w: 5.95, h: 4.9, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addText("Compiled from memory; verify editions and page numbers before formal citation.", { x: 0.6, y: 6.6, w: 12.1, h: 0.3, fontSize: 11, italic: true, color: C.accent6, margin: 0, isTextBox: true });
  s.addNotes("Textbooks and landmark papers for each family. Check bibliographic details before citing.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
