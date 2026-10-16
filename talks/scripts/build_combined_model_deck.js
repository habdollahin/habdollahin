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
  name: "Theranostic Twin",
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
pres.title = "The Combined Model: Digital Twins and Virtual Trials";
pres.author = "Hamid Abdollahi";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const EQ = "+mj-lt"; // equations in the heading (serif) font
const FOOT = "Digital twins & virtual trials  ·  the combined model";

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
    shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06, x, y, w: 1.05, h: 0.28,
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
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
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
  function iconCircle(slide, img, x, y, d = 0.6, fill = C.accent2) {
    slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: "icon circle" });
    const p = d * 0.22;
    slide.addImage({ data: img, x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, objectName: "icon" });
  }

  // ================================================================ 1 Title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  s.addText("FINAL DECK  ·  OCTOBER 2026", { placeholder: "kicker" });
  s.addText("The Combined Model: Digital Twins and Virtual Trials", { placeholder: "title" });
  s.addText("Every mathematical model, the best AI methods for each layer, and how they combine into one predictive twin tested in a virtual trial", { placeholder: "body" });
  s.addText("Hamid Abdollahi, PhD  ·  Medical physics · quantitative imaging · AI · radiopharmaceutical therapy", { placeholder: "byline" });
  s.addNotes("This deck collects every mathematical model in my recent work: the DoseVLM dose-prediction code, the radiomics pipeline, the virtual-trials review and the research-vision talk. Part I explains each model with its equation and use. Part II recommends the best AI methods for building each layer of a digital twin. Status tags show whether a model is already in code, discussed in talks or the review, or proposed.");

  // ================================================================ 2 Exec summary
  s = content("EXECUTIVE SUMMARY", "Eight model layers make one predictive engine", "Introduction");
  const layers = [
    [IC.users, "1 · Virtual patients", "Who is in the trial: sampled cohorts, phantoms, growth"],
    [IC.flask, "2 · Pharmacokinetics", "Where the drug goes: PBPK ODEs, TACs, SUV"],
    [IC.rad, "3 · Dosimetry", "How much energy is absorbed: MIRD, voxel, EBRT"],
    [IC.dna, "4 · Radiobiology", "What dose does: LQ, BED with G factor, TCP, NTCP"],
    [IC.sync, "5 · Updating and optimisation", "Bayesian updates per cycle; choose activity"],
    [IC.brain, "6 · Learned surrogates", "DoseVLM: seconds instead of hours of simulation"],
    [IC.micro, "7 · Imaging biomarkers", "Radiomics: features, selection, classifiers"],
    [IC.shield, "8 · Evaluation and credibility", "AUC, calibration, gamma, V&V 40, trial statistics"],
  ];
  layers.forEach(([img, h, d], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 6.2, y = 1.75 + row * 1.12;
    iconCircle(s, img, x, y + 0.08, 0.62);
    s.addText(h, { x: x + 0.82, y: y, w: 5.1, h: 0.4, fontSize: 17, bold: true, color: C.text2, margin: 0, valign: "top", isTextBox: true });
    s.addText(d, { x: x + 0.82, y: y + 0.42, w: 5.1, h: 0.5, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText("Status of each model:", { x: 0.6, y: 6.38, w: 2.1, h: 0.3, fontSize: 12, color: C.accent6, margin: 0, valign: "middle", isTextBox: true });
  tag(s, "code", 2.65, 6.39); tag(s, "talk", 3.85, 6.39); tag(s, "review", 5.05, 6.39); tag(s, "new", 6.25, 6.39);
  s.addText("code = implemented in the repo  ·  talks = research-vision slides  ·  review = virtual-trials review  ·  proposed = next step", { x: 7.45, y: 6.33, w: 5.3, h: 0.42, fontSize: 11, color: C.accent6, margin: 0, valign: "middle", isTextBox: true });
  s.addNotes("Part III combines all eight layers into one model. The engine has eight layers. Layers 1 to 4 are the mechanistic chain from patient to biological effect. Layer 5 makes it a twin: it is updated with each patient's measurements and used to choose treatment. Layer 6 replaces slow simulations with learned surrogates. Layers 7 and 8 supply imaging inputs and prove the whole thing is credible.");

  // ================================================================ 3 Pipeline
  s = content("HOW THE LAYERS CONNECT", "A virtual trial is a chain of models", "Introduction");
  const steps = [
    ["PET / SPECT / CT", "imaging input"], ["Virtual cohort", "θᵢ ~ p(θ | x)"], ["PBPK kinetics", "A(t) per organ"],
    ["Voxel dosimetry", "D = Ã ⊛ DVK"], ["Radiobiology", "BED, EQD2"], ["TCP / NTCP", "outcome"], ["Decision", "activity, cycles"],
  ];
  steps.forEach(([a, b], i) => {
    const x = 0.6 + i * 1.76, y = 1.85;
    s.addText([{ text: a, options: { bold: true, fontSize: 14, color: C.background1, breakLine: true } }, { text: b, options: { fontSize: 12, color: C.background2, fontFace: EQ } }], {
      shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x, y, w: 1.55, h: 1.25, fill: { color: i === 6 ? C.accent2 : C.text2 },
      align: "center", valign: "middle", margin: 0.05, isTextBox: true, objectName: `step ${a}`,
    });
    if (i < steps.length - 1) s.addShape(pres.shapes.LINE, { x: x + 1.56, y: y + 0.625, w: 0.19, h: 0, line: { color: C.accent6, width: 1.5, endArrowType: "triangle" }, objectName: "arrow" });
  });
  // feedback loop
  s.addShape(pres.shapes.LINE, { x: 11.5, y: 3.12, w: 0, h: 0.45, line: { color: C.accent1, width: 2 }, objectName: "loop down" });
  s.addShape(pres.shapes.LINE, { x: 4.15, y: 3.57, w: 7.35, h: 0, line: { color: C.accent1, width: 2 }, objectName: "loop across" });
  s.addShape(pres.shapes.LINE, { x: 4.15, y: 3.12, w: 0, h: 0.45, line: { color: C.accent1, width: 2, beginArrowType: "triangle" }, objectName: "loop up" });
  s.addText("Every cycle: post-therapy SPECT/PET → Bayesian update of patient parameters → re-plan the next cycle", { x: 2.6, y: 3.65, w: 10.1, h: 0.35, fontSize: 14, italic: true, color: C.text1, align: "center", margin: 0, isTextBox: true });
  // trial arms + validation
  const lower = [
    ["Virtual trial", "Run every arm (e.g. fixed 7.4 GBq vs model-guided activity) on the same virtual patients: a perfectly paired design."],
    ["Learned surrogates", "Replace slow links (Monte Carlo dose, PBPK solves) with networks such as DoseVLM, so thousands of patients run in minutes."],
    ["Credibility", "Step 1 technical fidelity · Step 2 external testing · Step 3 prospective evaluation, scaled to risk (ASME V&V 40)."],
  ];
  lower.forEach(([h, d], i) => {
    const x = 0.6 + i * 4.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.35, w: 3.85, h: 2.3, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `card ${h}` });
    iconCircle(s, [IC.scale, IC.brain, IC.shield][i], x + 0.22, 4.55, 0.55);
    s.addText(h, { x: x + 0.9, y: 4.6, w: 2.8, h: 0.45, fontSize: 17, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
    s.addText(d, { x: x + 0.22, y: 5.25, w: 3.45, h: 1.3, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("Read the chain left to right. Images build the virtual patient; PBPK turns activity into time-activity curves; dosimetry turns those into absorbed dose; radiobiology turns dose into biological effect; TCP and NTCP turn effect into outcome probabilities; the decision layer picks activity and cycles. The amber loop is what makes it a digital twin rather than a one-off model: it is updated with each new measurement.");

  // ================================================================ Part I divider
  pres.addSection({ title: "Part I · Mathematical models" });
  s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Part I · Mathematical models" });
  s.addText("PART I", { placeholder: "kicker" });
  s.addText("The mathematical models, layer by layer", { placeholder: "title" });
  s.addText("For each model: the equation, what each term means, where it is used, and whether it is already in code", { placeholder: "body" });
  s.addNotes("Part I walks through the eight layers. Each slide gives the governing equation and a one-line explanation you can use when presenting or writing.");

  const P1 = "Part I · Mathematical models";

  // ================================================================ L1 Virtual patients
  s = content("LAYER 1 · VIRTUAL PATIENTS", "Who is in the virtual trial", P1);
  grid4(s, [
    { title: "Parametric virtual population", kind: "talk", eq: "θᵢ ~ p(θ | xᵢ),   xᵢ = (age, weight, eGFR, SUV, …)", desc: "Sample N = 1,000–10,000 patients from published covariate distributions. Use it to explore heterogeneity and enrich rare subgroups." },
    { title: "Mixed-effects (population) prior", kind: "talk", eq: "θᵢ = θ_pop · exp(ηᵢ),   ηᵢ ~ N(0, Ω)", desc: "Typical value θ_pop plus log-normal between-patient variability Ω. It is also the prior for the per-patient Bayesian update (Layer 5)." },
    { title: "Computational phantom", kind: "code", eq: "(x/a)² + (y/b)² ≤ 1;   PTV: ‖(x − c)/r‖ ≤ 1", desc: "dosevlm/synthetic.py: random body, PTV ellipsoid, cord and parotids, random Rx and arcs. Known ground truth for testing the pipeline." },
    { title: "Mechanistic tumour growth", kind: "review", eq: "∂N/∂t = ∇·(D∇N) + k N (1 − N/θ) − R(D)", desc: "Reaction–diffusion: cell density N spreads (D) and grows logistically (k, capacity θ); R(D) kills cells. Calibrated to serial imaging." },
  ]);
  s.addNotes("Four ways to populate a virtual trial. The first two are statistical; the phantom is geometric and is already implemented for testing; the growth model adds time, so the twin can evolve between scans. In practice you combine them: sample parameters from the mixed-effects model, attach them to real or synthetic anatomy, and let tumours evolve.");

  // ================================================================ L2 PBPK
  s = content("LAYER 2 · PHARMACOKINETICS", "PBPK: where the radioligand goes", P1);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.7, w: 7.0, h: 4.95, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "pbpk card" });
  s.addText("Per organ i (¹⁷⁷Lu-PSMA, Cancers 2024)", { x: 0.85, y: 1.85, w: 5.2, h: 0.4, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
  tag(s, "talk", 6.3, 1.9);
  const pbpk = [
    ["Free ligand", "Vᵢ dCᵢ/dt = Qᵢ (C_art − Cᵢ/Pᵢ) − k_on Rᵢ Cᵢ + k_off Bᵢ − λ_phys Cᵢ"],
    ["Receptor-bound", "dBᵢ/dt = k_on Rᵢ Cᵢ − (k_off + λ_int + λ_phys) Bᵢ"],
    ["Internalised", "dIᵢ/dt = λ_int Bᵢ − (λ_rel + λ_phys) Iᵢ"],
    ["Free receptors", "Rᵢ = R_tot,i − Bᵢ   (hot + cold ligand share Rᵢ)"],
    ["Kidney excretion", "extra term − CL_R · C_art in the kidney balance"],
  ];
  pbpk.forEach(([h, e], i) => {
    const y = 2.4 + i * 0.83;
    s.addText(h, { x: 0.85, y, w: 6.5, h: 0.3, fontSize: 13, bold: true, color: C.accent2, margin: 0, isTextBox: true });
    s.addText(e, { x: 0.85, y: y + 0.3, w: 6.5, h: 0.45, fontSize: 15, fontFace: EQ, color: C.accent4, fill: { color: C.background1 }, margin: 0.06, valign: "middle", isTextBox: true });
  });
  s.addText("Parameters", { x: 8.0, y: 1.7, w: 4.7, h: 0.4, fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true });
  s.addText(bullets([
    "Q: organ blood flow · P: tissue/blood partition",
    "R_tot: receptor density (PSMA expression)",
    "k_on, k_off: binding and unbinding rates",
    "λ_int, λ_rel: internalisation and release",
    "λ_phys = ln2 / 6.65 d for ¹⁷⁷Lu",
    "CL_R: renal clearance (eGFR-linked)",
  ], 14), { x: 8.0, y: 2.15, w: 4.7, h: 2.7, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addText([{ text: "Tumour sink effect. ", options: { bold: true } }, { text: "Large tumour burden binds more ligand, so less reaches kidneys and salivary glands. Only a mechanistic model with shared receptors predicts this." }], {
    x: 8.0, y: 4.95, w: 4.7, h: 1.7, fontSize: 14, color: C.text1, fill: { color: C.background2 }, margin: 0.15, valign: "middle", isTextBox: true, objectName: "sink callout",
  });
  s.addNotes("PBPK is the core of the theranostic twin. Each organ has free, receptor-bound and internalised compartments, all decaying physically. Because hot and cold peptide compete for the same receptors, the model captures peptide amount effects and the tumour sink effect. Parameters come from literature plus patient imaging; patient-specific ones like receptor density are estimated from pre-therapy PET.");

  // ================================================================ L2b TAC
  s = content("LAYER 2 · FROM IMAGES TO ACTIVITY", "Time–activity curves and integrated activity", P1);
  const tacCards = [
    { title: "SUV (body weight)", kind: "code", eq: "SUV = C · BW / (A_inj · e^(−λΔt))", desc: "Decay-corrected uptake per injected activity and body weight." },
    { title: "Time–activity fit", kind: "talk", eq: ["A(t) = A₀ e^(−λ_eff t)", "λ_eff = λ_phys + λ_bio,  T_eff = ln2 / λ_eff"], eqH: 0.85, desc: "Mono- or bi-exponential fit to 3–4 scans over 4–7 days." },
    { title: "Time-integrated activity", kind: "talk", eq: ["Ã = ∫₀^∞ A(t) dt = A₀ / λ_eff", "1 time point:  Ã ≈ A(t) · 2t / ln2"], eqH: 0.85, desc: "Hänscheid single-time-point estimate: one scan near t ≈ T_eff." },
  ];
  tacCards.forEach((c, i) => eqCard(s, { x: 0.6, y: 1.7 + i * 1.75, w: 6.1, h: 1.65, eqSize: 14, descSize: 13, ...c, eqH: c.eqH ? 0.65 : 0.45 }));
  {
    const xs = []; for (let t = 0; t <= 168; t += 4) xs.push(t);
    const kid = xs.map((t) => +(4.0 * Math.exp(-Math.LN2 * t / 35)).toFixed(3));
    const tum = xs.map((t) => +(3.0 * Math.exp(-Math.LN2 * t / 70)).toFixed(3));
    s.addChart(pres.charts.SCATTER, [{ name: "Time (h)", values: xs }, { name: "Kidney (T_eff 35 h)", values: kid }, { name: "Tumour (T_eff 70 h)", values: tum }], {
      x: 7.0, y: 1.7, w: 5.7, h: 4.6, lineSize: 2.5, lineDataSymbol: "none", chartColors: [HEX.accent2, HEX.accent3],
      showTitle: true, title: "Illustrative ¹⁷⁷Lu-PSMA curves (% injected activity)", titleFontSize: 13, titleColor: HEX.dk1,
      showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: HEX.dk1,
      valAxisLabelColor: HEX.accent6, catAxisLabelColor: HEX.accent6, valAxisLabelFontSize: 11, catAxisLabelFontSize: 11,
      valGridLine: { color: "E3E8EA", size: 0.75 }, catGridLine: { style: "none" },
      showCatAxisTitle: true, catAxisTitle: "Hours after injection", catAxisTitleFontSize: 11, catAxisTitleColor: HEX.accent6,
      showValAxisTitle: true, valAxisTitle: "%IA", valAxisTitleFontSize: 11, valAxisTitleColor: HEX.accent6,
      catAxisMinVal: 0, catAxisMaxVal: 168, catAxisMajorUnit: 24, ...CHART_TEXT, objectName: "TAC chart",
    });
    s.addText("Area under each curve = Ã. Longer tumour retention gives tumour more dose per GBq than kidney.", { x: 7.0, y: 6.35, w: 5.7, h: 0.5, fontSize: 12, color: C.accent6, italic: true, margin: 0, isTextBox: true });
  }
  s.addNotes("Imaging gives activity at a few time points. Fitting an exponential gives the effective half-life, and the area under the curve is the time-integrated activity, the quantity dosimetry needs. The single-time-point method is a practical shortcut when only one post-therapy scan is available. The curves are illustrative, not patient data.");

  // ================================================================ L3 Dosimetry
  s = content("LAYER 3 · DOSIMETRY", "From integrated activity to absorbed dose", P1);
  grid4(s, [
    { title: "MIRD organ dosimetry", kind: "talk", eq: "D(r_T) = Σ_S Ã(r_S) · S(r_T ← r_S)", desc: "S-values give dose to target per decay in source, from reference phantoms. Fast, but organ-level only and assumes uniform uptake." },
    { title: "Voxel dosimetry (DVK)", kind: "talk", eq: "D(v) = (Ã ⊛ DVK)(v)", desc: "Convolve the voxel Ã map with a dose-voxel kernel. Captures heterogeneity; assumes homogeneous tissue around each voxel." },
    { title: "Local energy deposition", kind: "talk", eq: "D(v) = Ã(v) · Ē / m(v)", desc: "For β emitters whose range is below the voxel size (¹⁷⁷Lu): all energy stays in the voxel. Simple and surprisingly accurate." },
    { title: "Monte Carlo and AI surrogates", kind: "talk", eq: "D = MC(Ã, ρ_CT)  ≈  f_θ(Ã, CT)", desc: "Monte Carlo (e.g. GATE) is the reference but takes hours; a trained network f_θ approximates it in about 3 s (Layer 6)." },
  ]);
  s.addNotes("Four levels of dosimetry, from organ-level MIRD to full Monte Carlo. The trade-off is accuracy against time. For a twin that must run thousands of virtual patients, AI surrogates trained on Monte Carlo give near-reference accuracy in seconds.");

  // ================================================================ L3b EBRT
  s = content("LAYER 3 · EXTERNAL-BEAM GEOMETRY AND PLAN QUALITY", "Beam physics and dose metrics in DoseVLM", P1);
  grid4(s, [
    { title: "Divergent fluence projection", kind: "code", eq: "Φ(v) = Σ_seg MU · 𝟙[v ∈ aperture] · (SAD / (SAD − z))²", eqSize: 15, desc: "IEC 61217 couch, gantry and collimator rotations; jaw + MLC aperture test; inverse-square weight (rtplan.py)." },
    { title: "Pencil-beam dose (synthetic)", kind: "code", eq: "D = G_σ ⊛ (Φ · e^(−μ d)),   μ = 0.004 mm⁻¹", eqSize: 15, desc: "Exponential attenuation with depth d plus Gaussian scatter blur. Used only to make learnable synthetic data." },
    { title: "Dose–volume histogram metrics", kind: "code", eq: "D_x = dose to hottest x %;   V_x = % volume ≥ x Gy", eqSize: 15, desc: "PTV: D99, D95, D1, Dmean. OAR: Dmean, D0.1cc (metrics.py and the plain-Python dvh_tool.py)." },
    { title: "3-D gamma index", kind: "code", eq: "γ = min_r′ √( ΔD²/(δ·D_max)² + |r′ − r|²/DTA² )", eqSize: 15, desc: "A voxel passes if γ ≤ 1 at 3 % / 3 mm with a 10 % low-dose threshold. Pass rate compares predicted with clinical dose." },
  ]);
  s.addNotes("These are the external-beam models already implemented in DoseVLM. The fluence projection gives the network explicit beam geometry. DVH metrics and gamma are the clinical yardsticks for whether a predicted dose matches the real one.");

  // ================================================================ L4 Radiobiology
  s = content("LAYER 4 · RADIOBIOLOGY", "From physical dose to biological effect", P1);
  grid4(s, [
    { title: "Linear-quadratic survival", kind: "talk", eq: "S = exp(−αD − βD²)", desc: "Fraction of cells surviving dose D. α/β sets fractionation sensitivity: low for late-responding tissue and prostate." },
    { title: "BED for external beam", kind: "talk", eq: "BED = n d (1 + d / (α/β))", desc: "n fractions of d Gy. Lets different schedules be compared on one biological scale." },
    { title: "BED for RPT (dose-rate G factor)", kind: "talk", eq: ["BED = D (1 + G · D / (α/β))", "G = λ_eff / (λ_eff + μ)"], eqH: 0.9, desc: "Continuous, decaying dose rate allows repair (rate μ). Example kidney: T_eff 30 h, T_rep 2.8 h → G ≈ 0.085." },
    { title: "EQD2", kind: "talk", eq: "EQD2 = BED / (1 + 2 / (α/β))", desc: "Dose in 2 Gy fractions with the same effect, so RPT and external-beam limits can be added and compared." },
  ]);
  s.addNotes("Radiobiology converts absorbed dose to effect. The key RPT-specific idea is the G factor: because dose is delivered slowly and decays, cells repair during irradiation, so the quadratic term shrinks. This is the subject of my IEEE TRPMS 2025 paper. Parameter values here are illustrative.");

  // ================================================================ L4b TCP/NTCP
  s = content("LAYER 4 · OUTCOME PROBABILITY", "TCP and NTCP define the therapeutic window", P1);
  eqCard(s, { x: 0.6, y: 1.7, w: 5.6, h: 2.35, title: "Tumour control (Poisson)", kind: "talk", eq: ["TCP = exp(−N₀ · e^(−α · BED))", "N₀ = ρ · V  clonogens"], eqH: 0.9, desc: "Probability that no clonogenic cell survives. Steep sigmoid in BED." });
  eqCard(s, { x: 0.6, y: 4.3, w: 5.6, h: 2.35, title: "Normal-tissue complication (LKB)", kind: "review", eq: ["NTCP = Φ((gEUD − TD₅₀) / (m · TD₅₀))", "gEUD = (Σ vᵢ Dᵢ^(1/n))ⁿ"], eqH: 0.9, desc: "n = volume effect (n → 1 parallel, n → 0 serial organ); m = slope." });
  {
    const bed = []; for (let b = 0; b <= 120; b += 5) bed.push(b);
    const N0 = Math.LN2 * Math.exp(0.15 * 60);
    const tcp = bed.map((b) => +Math.exp(-N0 * Math.exp(-0.15 * b)).toFixed(3));
    const ntcp = bed.map((b) => +Phi((b - 90) / (0.2 * 90)).toFixed(3));
    s.addChart(pres.charts.LINE, [{ name: "TCP (tumour)", labels: bed.map(String), values: tcp }, { name: "NTCP (organ at risk)", labels: bed.map(String), values: ntcp }], {
      x: 6.5, y: 1.7, w: 6.2, h: 4.55, lineSize: 3, lineDataSymbol: "none", chartColors: [HEX.accent2, HEX.accent3],
      showTitle: true, title: "Illustrative curves vs BED (Gy)", titleFontSize: 13, titleColor: HEX.dk1,
      showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: HEX.dk1,
      valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0%",
      valAxisLabelColor: HEX.accent6, catAxisLabelColor: HEX.accent6, valAxisLabelFontSize: 11, catAxisLabelFontSize: 11,
      valGridLine: { color: "E3E8EA", size: 0.75 }, catGridLine: { style: "none" }, catAxisLabelFrequency: 2, ...CHART_TEXT, objectName: "TCP NTCP chart",
    });
    s.addText("Parameters illustrative: TCP₅₀ = 60 Gy, α = 0.15 Gy⁻¹; TD₅₀ = 90 Gy, m = 0.2. The gap between curves is the window a twin tries to exploit per patient.", { x: 6.5, y: 6.25, w: 6.2, h: 0.6, fontSize: 12, italic: true, color: C.accent6, margin: 0, isTextBox: true });
  }
  s.addNotes("TCP and NTCP turn biological dose into probabilities of cure and of harm. Their separation is the therapeutic window. A twin shifts both curves per patient, using that patient's kinetics and anatomy, which is why fixed activity can under-treat one patient and harm another.");

  // ================================================================ L5 Updating & optimisation
  s = content("LAYER 5 · TWIN UPDATING AND OPTIMISATION", "Learn each patient, then choose the treatment", P1);
  eqCard(s, { x: 0.6, y: 1.7, w: 5.9, h: 2.4, title: "Bayesian update across cycles", kind: "talk", eq: ["p(θ | y₁…ₖ) ∝ p(yₖ | θ) · p(θ | y₁…ₖ₋₁)", "θ̂_MAP = argmax_θ p(θ | y)"], eqH: 0.9, desc: "Population prior + sparse post-therapy SPECT. The prediction band narrows from cycle 1 to cycle 2." });
  eqCard(s, { x: 0.6, y: 4.3, w: 5.9, h: 2.4, title: "Activity optimisation", kind: "talk", eq: ["A* = max A  s.t.  BED_OAR(A) ≤ L", "linear case: A* = A_ref · D_lim / D(A_ref)"], eqH: 0.9, desc: "e.g. kidney limit. Tumour sink and saturation make D(A) nonlinear, so the PBPK twin is needed." });
  s.addText("Illustrative ¹⁷⁷Lu-PSMA twin: activity alone changes the outcome", { x: 6.9, y: 1.7, w: 5.8, h: 0.4, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
  [["7.4", "GBq", "Organs safe · tumour refractory", C.accent6], ["8.3", "GBq", "Organs safe · tumour responds", C.accent2], ["10", "GBq", "Tumour responds · organ toxicity", C.accent3]].forEach(([n, u, d, col], i) => {
    const y = 2.25 + i * 1.08;
    s.addText([{ text: n, options: { fontSize: 40, bold: true, color: col } }, { text: " " + u, options: { fontSize: 16, color: C.text1 } }], { x: 6.9, y, w: 2.3, h: 0.9, margin: 0, valign: "middle", fontFace: EQ, isTextBox: true });
    s.addText(d, { x: 9.3, y, w: 3.4, h: 0.9, fontSize: 15, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  });
  s.addText([{ text: "Design space to optimise: ", options: { bold: true } }, { text: "activity, number of cycles, interval, bolus vs metronomic infusion, hot/cold ligand ratio. Objective: maximise tumour BED subject to every OAR limit." }], {
    x: 6.9, y: 5.55, w: 5.8, h: 1.15, fontSize: 14, color: C.text1, fill: { color: C.background2 }, margin: 0.15, valign: "middle", isTextBox: true,
  });
  s.addNotes("This layer is what makes a model a digital twin. After each cycle the patient's parameters are re-estimated from new measurements, and the next cycle is optimised. The three activities are from my research-vision talk, after Yazdani et al. 2026, and are illustrative, not recommendations.");

  // ================================================================ L6 DoseVLM
  s = content("LAYER 6 · LEARNED SURROGATE", "DoseVLM: a vision-language dose engine", P1);
  const flow = [["CT + masks + beam projection", C.text2], ["3-D residual U-Net encoder", C.text2], ["Cross-attention to plan tokens", C.accent2], ["Decoder + FiLM", C.text2], ["Dose / Rx (softplus)", C.accent4]];
  flow.forEach(([t, col], i) => {
    const x = 0.6 + i * 2.47;
    s.addText(t, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x, y: 1.7, w: 2.2, h: 0.95, fill: { color: col }, color: C.background1, fontSize: 14, bold: true, align: "center", valign: "middle", margin: 0.06, isTextBox: true, objectName: `flow ${t}` });
    if (i < flow.length - 1) s.addShape(pres.shapes.LINE, { x: x + 2.21, y: 2.175, w: 0.25, h: 0, line: { color: C.accent6, width: 1.5, endArrowType: "triangle" }, objectName: "arrow" });
  });
  s.addText("Plan text + control-point tokens (gantry sin/cos, MU, MLC area) → plan transformer", { x: 3.07, y: 2.75, w: 7.0, h: 0.35, fontSize: 13, italic: true, color: C.accent2, align: "center", margin: 0, isTextBox: true });
  const dv = [
    { title: "Attention and conditioning", kind: "code", eq: ["Attn(Q,K,V) = softmax(QKᵀ / √d) V", "FiLM: f′ = f (1 + γ(z)) + β(z)"], eqH: 0.9, desc: "Voxels (Q) query plan tokens (K, V); pooled plan embedding z rescales decoder features." },
    { title: "Training loss", kind: "code", eq: ["L = MAE_body + MAE_PTV + 0.5 L_DVH + 0.1 L_∇", "L_DVH = mean |sort(ŷ) − sort(y)| = W₁"], eqH: 0.9, desc: "Sorted-dose L1 equals the Wasserstein-1 distance between DVHs; gradient term keeps penumbra sharp." },
  ];
  dv.forEach((c, i) => eqCard(s, { x: 0.6 + i * 6.2, y: 3.3, w: 5.9, h: 2.45, eqSize: 15, ...c }));
  s.addText("Output in Gy:  D̂ = softplus(h) · Rx · 𝟙_body    ·    AdamW, cosine learning rate with warm-up, 96³ PTV-focused crops", { x: 0.6, y: 6.0, w: 12.1, h: 0.6, fontSize: 14, color: C.text1, fill: { color: C.background2 }, margin: 0.12, valign: "middle", isTextBox: true });
  s.addNotes("DoseVLM is the working example of a learned surrogate in my repo. It predicts 3-D VMAT dose from CT, structures and the plan, reading the plan both as images and as text. In a virtual trial you change the plan text or arcs and re-predict dose in seconds. It is research use only.");

  // ================================================================ L7 Radiomics I
  s = content("LAYER 7 · IMAGING BIOMARKERS (1/2)", "Radiomics: from image to robust features", P1);
  grid4(s, [
    { title: "IBSI discretisation", kind: "code", eq: "bin(x) = ⌊x / w⌋ − ⌊min / w⌋ + 1", desc: "Fixed bin width w = 25 HU for CT, or fixed bin count 32 for MR and PET. Makes texture comparable across patients." },
    { title: "First-order and texture", kind: "code", eq: ["H = −Σ p(i) log₂ p(i)", "GLCM contrast = Σ (i − j)² p(i, j)"], eqH: 0.85, desc: "Plus shape, GLRLM, GLSZM, GLDM, NGTDM; on original, LoG and wavelet images." },
    { title: "Robustness filter ICC(3,1)", kind: "code", eq: "ICC = (MS_R − MS_E) / (MS_R + (k − 1) MS_E)", desc: "Erode and dilate the mask by one voxel; keep only features with ICC ≥ 0.75." },
    { title: "Registration (mutual information)", kind: "code", eq: "I(F;M) = Σ p(f,m) log [ p(f,m) / (p(f) p(m)) ]", desc: "Rigid → affine → B-spline, maximising Mattes MI, when a mask was drawn on another image (e.g. T2 → ADC)." },
  ], { card: { eqSize: 15 } });
  s.addNotes("Radiomics supplies imaging inputs to the twin, such as heterogeneity and endocrine-organ signatures. Before modelling, features are standardised by IBSI discretisation and filtered for robustness so that small contouring changes do not drive the result.");

  // ================================================================ L7 Radiomics II
  s = content("LAYER 7 · IMAGING BIOMARKERS (2/2)", "Selection and classifiers without data leakage", P1);
  grid4(s, [
    { title: "mRMR and LASSO selection", kind: "code", eq: ["mRMR: max_f [ I(f; y) − mean_{s∈S} I(f; s) ]", "LASSO: min_β ℓ(y, Xβ) + λ‖β‖₁"], eqH: 0.9, desc: "Also Mann–Whitney with BH-FDR, ReliefF, Boruta, RFE. All fitted inside each CV fold." },
    { title: "Classifiers", kind: "code", eq: ["Logistic: p = σ(βᵀx) = 1 / (1 + e^(−βᵀx))", "Ensembles: RF, XGBoost, LightGBM, SVM, MLP"], eqH: 0.9, desc: "Soft-voting ensemble of the top 3 by nested-CV AUC." },
    { title: "Class imbalance (SMOTE)", kind: "code", eq: "x_new = xᵢ + u (x_nn − xᵢ),   u ~ U(0, 1)", desc: "Synthetic minority samples along lines to nearest neighbours; or class weights, ADASYN, under-sampling." },
    { title: "Nested CV + Bayesian tuning", kind: "code", eq: "TPE: choose x maximising l(x) / g(x)", desc: "Outer folds score, inner folds tune (Optuna). Threshold picked on out-of-fold data; test set used once." },
  ], { h: 2.3, card: { eqSize: 15 } });
  s.addText("Worked example: endocrine-organ radiomics in PSMA-negative recurrence (n = 101): PET + CT + clinical AUC 0.758 vs 0.727 clinical alone (gain not significant)", { x: 0.6, y: 6.55, w: 12.1, h: 0.35, fontSize: 12, italic: true, color: C.accent6, margin: 0, isTextBox: true });
  s.addNotes("Every step that sees labels lives inside one pipeline that is refitted in each fold, so no information leaks from test to training. The PSMA-negative endocrine study is the applied example.");

  // ================================================================ L8 Evaluation
  s = content("LAYER 8 · EVALUATION", "Is the prediction right, calibrated and useful?", P1);
  grid4(s, [
    { title: "Discrimination", kind: "code", eq: ["AUC = P(s⁺ > s⁻) = U / (n⁺ n⁻)", "Youden J = Se + Sp − 1"], eqH: 0.9, desc: "DeLong test and bootstrap 95 % CI; PR-AUC, MCC, Cohen's κ, LR±." },
    { title: "Calibration", kind: "code", eq: ["Brier = mean (p − y)²", "ECE = Σ_b (n_b / N) |acc_b − conf_b|"], eqH: 0.9, desc: "Plus calibration slope and intercept and Hosmer–Lemeshow." },
    { title: "Clinical usefulness", kind: "code", eq: "NB(p_t) = TP/N − (FP/N) · p_t / (1 − p_t)", desc: "Decision-curve analysis: net benefit across threshold probabilities vs treat-all and treat-none." },
    { title: "Explainability", kind: "code", eq: "f(x) = φ₀ + Σⱼ φⱼ   (Shapley values)", desc: "SHAP attributes each prediction to features; permutation importance checks the ranking." },
  ]);
  s.addNotes("Three questions: does the model separate classes (AUC), are its probabilities honest (calibration), and does using it beat simple strategies (net benefit). Calibration matters most for a twin, because its probabilities feed decisions.");

  // ================================================================ L8b credibility
  s = content("LAYER 8 · CREDIBILITY AND TRIAL STATISTICS", "When can a virtual trial be trusted?", P1);
  grid4(s, [
    { title: "Paired arm comparison", kind: "review", eq: "Δᵢ = Yᵢ(arm B) − Yᵢ(arm A),   same virtual patient i", desc: "Every patient gets every arm, so between-patient noise cancels; far fewer subjects are needed." },
    { title: "Sample size for paired design", kind: "new", eq: "n = ((z₁₋α/₂ + z₁₋β) · σ_Δ / δ)²", desc: "δ = smallest difference worth detecting; σ_Δ = SD of paired differences." },
    { title: "Uncertainty propagation", kind: "review", eq: "Y⁽ᵏ⁾ = g(θ⁽ᵏ⁾),  θ⁽ᵏ⁾ ~ p(θ | data),  k = 1…K", desc: "Monte Carlo through the whole chain: phantom → kinetics → dose → NTCP. Report endpoint intervals, not points." },
    { title: "Risk-informed credibility", kind: "review", eq: "required evidence ∝ model influence × decision consequence", desc: "ASME V&V 40 and FDA 2023 guidance. Validate dose first, then response; agree with a real trial before extrapolating." },
  ], { card: { eqSize: 15 } });
  s.addNotes("A virtual trial is only as trustworthy as its weakest model. Paired designs make virtual trials statistically efficient. Uncertainty must be propagated end to end. The V&V 40 framework scales the validation effort to how much the decision depends on the model.");

  // ================================================================ Part II divider
  pres.addSection({ title: "Part II · AI for digital twins" });
  s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Part II · AI for digital twins" });
  s.addText("PART II", { placeholder: "kicker" });
  s.addText("Best AI models for digital-twin development", { placeholder: "title" });
  s.addText("Which AI family to use for each twin layer, why it fits, its caveats, and the open-source tools to start with", { placeholder: "body" });
  s.addNotes("Part II is a practical guide. There is no single best AI model for digital twins. The best choice depends on the layer, how much mechanism you know, and how much data you have.");
  const P2 = "Part II · AI for digital twins";

  // ================================================================ AI selection 2x2
  s = content("CHOOSING A METHOD", "Pick AI by mechanism knowledge and data size", P2);
  const quad = [
    ["Strong mechanism · small data", "Bayesian PBPK, PINNs, universal differential equations", "Most theranostic twins today: physics known, few scans per patient.", C.accent2, IC.flask],
    ["Strong mechanism · large data", "Neural operators (FNO, DeepONet), MC emulators", "Learn the simulator itself: Monte Carlo dose, PBPK solves in milliseconds.", C.text2, IC.cubes],
    ["Weak mechanism · small data", "Gaussian processes, TabPFN, regularised regression", "Biomarkers and outcomes in cohorts of 50–500 patients.", C.accent4, IC.chart],
    ["Weak mechanism · large data", "Deep nets, foundation models, diffusion models", "Segmentation, denoising, synthetic anatomy, multimodal fusion.", C.accent3, IC.brain],
  ];
  quad.forEach(([h, m, d, col, img], i) => {
    const col_ = i % 2, row = Math.floor(i / 2);
    const x = 1.4 + col_ * 5.75, y = 1.75 + row * 2.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 5.5, h: 2.3, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `quad ${h}` });
    iconCircle(s, img, x + 0.22, y + 0.22, 0.6, col);
    s.addText(h, { x: x + 0.95, y: y + 0.22, w: 4.4, h: 0.6, fontSize: 16, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
    s.addText(m, { x: x + 0.22, y: y + 0.95, w: 5.1, h: 0.6, fontSize: 15, bold: true, color: col === C.accent3 ? C.text1 : col, margin: 0, valign: "top", isTextBox: true });
    s.addText(d, { x: x + 0.22, y: y + 1.55, w: 5.1, h: 0.65, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText("MECHANISM", { x: 0.35, y: 1.75, w: 0.6, h: 4.8, fontSize: 12, bold: true, color: C.accent6, align: "center", valign: "middle", vert: "vert270", margin: 0, isTextBox: true });
  s.addText("DATA SIZE  →", { x: 1.4, y: 6.6, w: 11.25, h: 0.3, fontSize: 12, bold: true, color: C.accent6, align: "center", margin: 0, isTextBox: true });
  s.addNotes("Top row: you know the physics or physiology. Bottom row: you do not. Left column: little data per patient or few patients. Right column: lots of data or a simulator that can generate it. Theranostic twins mostly start top-left and use the top-right to accelerate.");

  // ================================================================ AI table
  s = content("OVERVIEW", "Best AI families for each twin layer", P2);
  {
    const H = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 13 } });
    const R = (a, b, c) => [{ text: a, options: { bold: true, color: C.text2 } }, { text: b }, { text: c, options: { color: C.accent6 } }];
    const rows = [
      [H("Twin layer"), H("Best AI models"), H("Starter tools")],
      R("Imaging: denoise, reconstruct", "U-Net / ResUNet, conditional GAN, diffusion denoisers", "MONAI, PyTorch"),
      R("Segmentation", "nnU-Net, TotalSegmentator, MedSAM / SAM-2 style promptable", "nnU-Net, MONAI Auto3DSeg"),
      R("Kinetics (PBPK)", "PINNs, neural ODEs, universal differential equations", "torchdiffeq, DeepXDE, Julia SciML"),
      R("Dosimetry", "CNN / Swin-UNETR voxel dose, neural operators, DoseVLM", "MONAI, neuraloperator, OpenGATE"),
      R("Virtual patients", "3-D latent diffusion (MAISI), GAN / VAE; copulas, CTGAN", "MONAI generative, SDV"),
      R("Calibration and UQ", "Simulation-based inference, hierarchical Bayes, conformal", "sbi, PyMC / Stan, MAPIE"),
      R("Outcome", "Cox, random survival forest, DeepSurv, XGBoost, TabPFN", "lifelines, scikit-survival"),
      R("Decision", "Bayesian optimisation, reinforcement learning, MPC", "BoTorch, Optuna, Gymnasium"),
      R("Orchestration", "Frontier LLM coding agents, biomedical text encoders", "Claude Code, BiomedBERT"),
    ];
    s.addTable(rows, { x: 0.6, y: 1.7, w: 12.1, colW: [2.9, 5.9, 3.3], fontSize: 13, color: C.text1, border: { type: "solid", pt: 0.75, color: "D5DEE0" }, fill: { color: C.background1 }, rowH: 0.47, valign: "middle", margin: [0.03, 0.08, 0.03, 0.08], objectName: "AI overview table" });
  }
  s.addNotes("One row per twin layer. The next slides go through each family in detail: what it is, how it works mathematically, where it fits in a theranostic twin, and its main caveat.");

  // ================================================================ AI-A Scientific ML
  s = content("AI FAMILY 1 · SCIENTIFIC MACHINE LEARNING", "Hybrid physics + AI: the core of a twin", P2);
  const sciml = [
    ["PINN", "L = L_data + λ ‖𝒩[u_θ]‖²", "Fit PBPK or diffusion to sparse SPECT; physics fills gaps. Caveat: stiff ODEs train poorly."],
    ["Neural ODE", "dz/dt = f_θ(z, t)", "Learn kinetics from irregular time points. Caveat: less interpretable than PBPK."],
    ["Universal DE", "dx/dt = f_PBPK(x; θ) + NN_φ(x)", "Keep known PBPK, learn only the missing term (e.g. sink saturation). Best fit for theranostics."],
    ["Neural operator", "𝒢_θ : Ã(·) ↦ D(·)   (FNO, DeepONet)", "Learn the whole Monte Carlo map; resolution-independent dose in milliseconds."],
    ["GP emulator", "y ~ GP(m(x), k(x, x′))", "Surrogate with error bars from ~100 simulator runs; Sobol sensitivity and calibration."],
  ];
  sciml.forEach(([n, e, d], i) => {
    const y = 1.7 + i * 1.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 12.1, h: 0.88, rectRadius: 0.06, fill: { color: i === 2 ? "E2F0EE" : C.background2 }, line: { color: i === 2 ? C.accent2 : C.background2, width: i === 2 ? 1.5 : 0.5 }, objectName: `row ${n}` });
    s.addText(n, { x: 0.8, y, w: 1.9, h: 0.88, fontSize: 16, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
    s.addText(e, { x: 2.7, y: y + 0.14, w: 4.3, h: 0.6, fontSize: 15, fontFace: EQ, color: C.accent4, fill: { color: C.background1 }, margin: 0.08, valign: "middle", isTextBox: true });
    s.addText(d, { x: 7.2, y, w: 5.35, h: 0.88, fontSize: 14, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  });
  s.addText("Recommended first build: a universal differential equation around the ¹⁷⁷Lu-PSMA PBPK model, calibrated with simulation-based inference.", { x: 0.6, y: 6.66, w: 12.1, h: 0.25, fontSize: 12, italic: true, color: C.accent2, margin: 0, isTextBox: true });
  s.addNotes("Scientific machine learning keeps the physics and lets a network learn what the physics misses. For theranostics, a universal differential equation is the most natural fit: the PBPK structure stays interpretable, and the network learns only a correction term. Neural operators are the best way to replace Monte Carlo dosimetry once enough simulations exist.");

  // ================================================================ AI-B Imaging AI
  s = content("AI FAMILY 2 · IMAGING AI", "Reliable inputs: segmentation and low-count imaging", P2);
  grid4(s, [
    { title: "nnU-Net", eq: "self-configuring 3-D U-Net: patch, spacing, augmentation", eqSize: 14, eqH: 0.55, desc: "Strongest out-of-the-box baseline for organ and lesion segmentation. Train per task; still the benchmark to beat." },
    { title: "TotalSegmentator", eq: "pre-trained: 100+ CT structures (MR version too)", eqSize: 14, eqH: 0.55, desc: "Instant organs at risk for dosimetry (kidneys, salivary glands, liver, marrow). Check small organs by eye." },
    { title: "MedSAM / promptable models", eq: "mask = f(image, box or point prompt)", eqSize: 14, eqH: 0.55, desc: "Foundation models for interactive lesion contouring; fast QA and correction. Weaker on PET/SPECT without fine-tuning." },
    { title: "Low-count SPECT/PET denoising", eq: "x̂ = f_θ(x_low)   (U-Net, cGAN, diffusion)", eqSize: 14, eqH: 0.55, desc: "Can cut ¹⁷⁷Lu SPECT time by 50–75 %. Always measure activity bias and small-lesion recovery on phantoms." },
  ], { card: {} });
  s.addNotes("Garbage in, garbage out: the twin is only as good as its activity maps and contours. nnU-Net and TotalSegmentator give reliable contours; promptable foundation models speed up correction. AI denoising shortens scans, but must preserve quantitative accuracy, which is the point of my Aim 2.");

  // ================================================================ AI-C Dosimetry & generative
  s = content("AI FAMILY 3 · DOSIMETRY AND VIRTUAL PATIENTS", "Fast dose engines and synthetic cohorts", P2);
  grid4(s, [
    { title: "Deep voxel dosimetry", eq: "D̂ = f_θ(SPECT/PET activity, CT density)", eqSize: 14, eqH: 0.55, desc: "U-Net or Swin-UNETR trained on Monte Carlo; learned dose-voxel kernels. Seconds per patient instead of hours." },
    { title: "Dose prediction for planning", eq: "D̂ = f_θ(CT, masks, plan)   (DoseVLM)", eqSize: 14, eqH: 0.55, desc: "OpenKBP-style cascaded U-Nets, diffusion dose models, and vision-language DoseVLM for external beam." },
    { title: "3-D generative anatomy", eq: "x₀ ← denoise(x_T),  conditioned on masks", eqSize: 14, eqH: 0.55, desc: "Latent diffusion (e.g. MONAI MAISI) makes CT with organ labels; GANs are fast but can mode-collapse." },
    { title: "Tabular virtual covariates", eq: "(age, eGFR, PSA, SUV) ~ copula / CTGAN", eqSize: 14, eqH: 0.55, desc: "Sample realistic joint covariates. Validate with MMD, downstream fidelity and membership-inference privacy tests." },
  ]);
  s.addNotes("Dosimetry AI removes the Monte Carlo bottleneck. Generative models scale the cohort, but realism must be shown, not assumed: compare distributions, check that downstream predictions behave the same, and test privacy.");

  // ================================================================ AI-D Inference, UQ, decisions
  s = content("AI FAMILY 4 · INFERENCE, UNCERTAINTY AND DECISIONS", "Calibrate the twin, then act on it safely", P2);
  grid4(s, [
    { title: "Simulation-based inference", eq: "q_φ(θ | y) ≈ p(θ | y),  trained on (θ, y) ~ sim", eqSize: 14, eqH: 0.55, desc: "Neural posterior estimation gives each patient's PBPK posterior in milliseconds once trained (sbi package)." },
    { title: "Conformal prediction", eq: "P(y ∈ C(x)) ≥ 1 − α", eqSize: 14, eqH: 0.55, desc: "Distribution-free intervals around any model (dose, NTCP). Also deep ensembles and MC dropout for UQ." },
    { title: "Bayesian optimisation", eq: "x_next = argmax EI(x)  on a GP surrogate", eqSize: 14, eqH: 0.55, desc: "Search activity, cycles and interval with few expensive twin runs (BoTorch, Optuna)." },
    { title: "Reinforcement learning / MPC", eq: "π* = argmax E[Σ rₜ],  r = BED_T − penalty(OAR)", eqSize: 14, eqH: 0.55, desc: "Cycle-by-cycle adaptation as a sequential decision. Train only in silico; keep a clinician in the loop." },
  ]);
  s.addNotes("These methods turn a predictive model into a decision tool. Simulation-based inference makes per-patient calibration instant. Conformal prediction gives honest uncertainty. Bayesian optimisation and reinforcement learning search the design space, but any recommendation stays decision support with explicit uncertainty.");

  // ================================================================ AI-E Outcome + LLMs
  s = content("AI FAMILY 5 · OUTCOME MODELS AND FOUNDATION MODELS", "Predict outcomes and accelerate the work", P2);
  grid4(s, [
    { title: "Survival models", eq: "h(t | x) = h₀(t) · exp(βᵀx)", eqSize: 14, eqH: 0.55, desc: "Cox for interpretability; random survival forests and DeepSurv for nonlinearity. Report C-index and calibration." },
    { title: "Small-data tabular", eq: "TabPFN, XGBoost / LightGBM, penalised LR", eqSize: 14, eqH: 0.55, desc: "TabPFN is a pre-trained transformer that is often strongest on small tables. Always compare against logistic regression." },
    { title: "Multimodal fusion", eq: "z = fuse(PET, CT, dose, clinical, text)", eqSize: 14, eqH: 0.55, desc: "Cross-attention fusion of image, dose and record, as in DoseVLM. Needs larger multi-site data." },
    { title: "LLMs and coding agents", eq: "Claude, other frontier LLMs; BiomedBERT, BiomedCLIP", eqSize: 14, eqH: 0.55, desc: "Write and test pipelines, map literature (e.g. the 73-report evidence map), draft protocols. Not for unsupervised clinical decisions." },
  ]);
  s.addNotes("For outcomes, start simple and earn complexity: Cox and logistic baselines first, then tree ensembles and TabPFN, then multimodal deep fusion when data allow. Large language models are best used as engineering and research accelerators, as I did to build DoseVLM and the radiomics pipeline, with every output verified.");

  // ================================================================ Recommended stack per aim
  s = content("RECOMMENDED STACK", "Best AI choice for each research aim", P2);
  const aims = [
    ["1", "Imaging biomarkers", "PyRadiomics + nested CV; logistic, XGBoost, TabPFN; SHAP"],
    ["2", "AI-enhanced imaging", "nnU-Net / TotalSegmentator; U-Net or diffusion denoising with quantitative checks"],
    ["3", "Digital twins", "UDE around PBPK + simulation-based inference; neural-operator dose emulator"],
    ["4", "Therapy integration", "DoseVLM for external beam + RPT voxel dose; Bayesian optimisation of activity"],
    ["5", "Virtual trials", "Copula / diffusion cohorts, GP emulators, conformal UQ, paired-design statistics"],
  ];
  aims.forEach(([n, h, d], i) => {
    const y = 1.75 + i * 0.98;
    s.addText(n, { shape: pres.shapes.OVAL, x: 0.6, y: y + 0.1, w: 0.65, h: 0.65, fill: { color: C.accent1 }, color: C.text1, fontSize: 20, bold: true, align: "center", valign: "middle", margin: 0, fontFace: EQ, isTextBox: true, objectName: `aim ${n}` });
    s.addText(h, { x: 1.5, y, w: 3.4, h: 0.85, fontSize: 18, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
    s.addText(d, { x: 4.9, y: y + 0.05, w: 7.8, h: 0.75, fontSize: 15, color: C.text1, fill: { color: C.background2 }, margin: 0.12, valign: "middle", isTextBox: true });
  });
  s.addText("Shared layer for all aims: MONAI + PyTorch, version-controlled pipelines, and V&V 40-style validation gates.", { x: 0.6, y: 6.6, w: 12.1, h: 0.3, fontSize: 13, italic: true, color: C.accent2, margin: 0, isTextBox: true });
  s.addNotes("This maps Part II onto the five aims of my research program. Each aim gets a concrete, open-source starting point.");

  // ================================================================ Part III divider
  const SIM = require(path.join(__dirname, "sim.js")).simulate();
  const F = SIM.sum.fixed, G = SIM.sum.guided, B = SIM.sum.bayes;
  const pct = (v) => Math.round(v * 100) + " %";
  pres.addSection({ title: "Part III · The combined model" });
  s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Part III · The combined model" });
  s.addText("PART III", { placeholder: "kicker" });
  s.addText("One combined twin: all models working together", { placeholder: "title" });
  s.addText("The full equation chain, the objective it optimises, how it learns each patient, a simulated virtual trial, and where AI plugs in", { placeholder: "body" });
  s.addNotes("Part III combines every model from Parts I and II into one working digital twin. It shows the chain as equations, the objective, the Bayesian update, an illustrative virtual trial run with the combined model, and the hybrid AI architecture.");
  const P3 = "Part III · The combined model";

  // ---------------------------------------------------------------- chain table
  s = content("THE COMBINED MODEL", "Seven equations, one chain", P3);
  {
    const H = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 13 } });
    const E = (t) => ({ text: t, options: { fontFace: EQ, color: C.accent4, fontSize: 14 } });
    const N = (t) => ({ text: t, options: { bold: true, color: C.text2 } });
    const M = (t) => ({ text: t, options: { color: C.accent6 } });
    const rows = [
      [H("Step"), H("Equation"), H("Informed by"), H("AI accelerator")],
      [N("1 Patient"), E("θᵢ ~ p(θ | xᵢ),  θᵢ = θ_pop e^(ηᵢ)"), M("covariates, PET"), M("copula / diffusion cohorts")],
      [N("2 Kinetics"), E("dx/dt = f_PBPK(x; θᵢ, A) → Ã_organ"), M("serial SPECT"), M("UDE, neural ODE")],
      [N("3 Dose"), E("D = Σ Ã · S   or   Ã ⊛ DVK"), M("CT, contours"), M("neural operator, deep dosimetry")],
      [N("4 Biology"), E("BED = D (1 + G D/(α/β)),  EQD2"), M("literature α/β, T_rep"), M("Bayesian priors")],
      [N("5 Outcome"), E("TCP = e^(−N₀ e^(−α BED)),  NTCP = Φ(·)"), M("response, toxicity"), M("survival ML, conformal")],
      [N("6 Update"), E("p(θ | y₁…ₖ) ∝ p(yₖ | θ) p(θ | y₁…ₖ₋₁)"), M("post-cycle SPECT"), M("simulation-based inference")],
      [N("7 Decide"), E("Aₖ* = argmax E[TCP (1 − NTCP)]"), M("OAR limits"), M("Bayesian optimisation, RL")],
    ];
    s.addTable(rows, { x: 0.6, y: 1.7, w: 12.1, colW: [1.6, 5.3, 2.4, 2.8], fontSize: 13, color: C.text1, border: { type: "solid", pt: 0.75, color: "D5DEE0" }, fill: { color: C.background1 }, rowH: 0.56, valign: "middle", margin: [0.03, 0.08, 0.03, 0.08], objectName: "combined chain table" });
  }
  s.addText("Each output is the next step's input; step 6 closes the loop every cycle. Swap any equation for its AI accelerator once validated.", { x: 0.6, y: 6.35, w: 12.1, h: 0.45, fontSize: 13, italic: true, color: C.accent2, margin: 0, isTextBox: true });
  s.addNotes("This is the whole twin on one slide. Read top to bottom: a patient's parameters drive kinetics, kinetics give integrated activity, that gives dose, dose gives biological dose, that gives outcome probabilities. After each cycle the parameters are updated, and the next activity is chosen to maximise uncomplicated tumour control. The last two columns say which data and which AI method improve each step.");

  // ---------------------------------------------------------------- objective
  s = content("THE COMBINED MODEL · OBJECTIVE", "What the combined twin optimises", P3);
  grid4(s, [
    { title: "Uncomplicated tumour control", kind: "new", eq: "P⁺(A) = TCP(BED_T(A)) · (1 − NTCP(BED_OAR(A)))", eqSize: 15, desc: "One number that rewards tumour control and penalises toxicity. Assumes the two are independent." },
    { title: "Constrained optimum", kind: "new", eq: ["A* = argmax_A  E_θ[ P⁺(A) ]", "s.t.  BED_kidney ≤ 28 Gy,  BED_salivary ≤ L_s"], eqH: 0.9, eqSize: 15, desc: "Expectation over the patient's posterior θ, so uncertainty is priced in." },
    { title: "Multi-cycle recursion", kind: "new", eq: ["BED_total = Σₖ Dₖ (1 + G Dₖ/(α/β))", "Aₖ* chosen after updating θ with y₁…ₖ₋₁"], eqH: 0.9, eqSize: 15, desc: "Each cycle uses everything measured so far; cycles add BED independently." },
    { title: "Combined RPT + external beam", kind: "new", eq: "EQD2_OAR = EQD2_RPT + EQD2_EBRT ≤ limit", eqSize: 15, desc: "DoseVLM predicts the external-beam dose; the PBPK twin gives the RPT dose; both sum on one EQD2 scale." },
  ]);
  s.addNotes("The objective joins the radiobiology layer to the decision layer. P plus is the probability of tumour control without complication. Optimising its expectation over the posterior means the twin is cautious when it is uncertain and bolder once measurements narrow the posterior. The last card links the external-beam work, DoseVLM, to the RPT twin.");

  // ---------------------------------------------------------------- bayes
  s = content("THE COMBINED MODEL · LEARNING THE PATIENT", "One post-therapy scan sharpens the twin", P3);
  eqCard(s, { x: 0.6, y: 1.7, w: 6.3, h: 2.45, title: "Log-normal conjugate update", kind: "new", eq: ["log d_K ~ N(μ₀, σ₀²),  log y ~ N(log d_K, σ_m²)", "σ_post = (1/σ₀² + 1/σ_m²)^(−1/2)"], eqH: 0.9, eqSize: 15, desc: "d_K = kidney dose per GBq. Prior from the population; y from cycle-1 SPECT dosimetry." });
  eqCard(s, { x: 0.6, y: 4.35, w: 6.3, h: 2.35, title: "Values used", kind: "new", eq: ["σ₀ = 0.35 (population CV),  σ_m = 0.15 (SPECT)", "σ_post = " + B.spost.toFixed(2)], eqH: 0.9, eqSize: 15, desc: "In the full twin the same update runs on every PBPK parameter (via SBI or MCMC)." });
  [["×/÷ " + B.prior.toFixed(2), "95 % range of kidney dose before any scan (population prior)", C.accent3], ["×/÷ " + B.post.toFixed(2), "95 % range after one cycle-1 SPECT measurement", C.accent2]].forEach(([n, d, col], i) => {
    const y = 1.7 + i * 2.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.2, y, w: 5.5, h: 1.95, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `bayes stat ${i}` });
    s.addText(n, { x: 7.45, y: y + 0.15, w: 5.0, h: 0.95, fontSize: 48, bold: true, color: col, fontFace: EQ, margin: 0, valign: "middle", isTextBox: true });
    s.addText(d, { x: 7.45, y: y + 1.1, w: 5.0, h: 0.7, fontSize: 15, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText("Narrower range → the twin can raise activity safely for patients whose kidneys clear well.", { x: 7.2, y: 6.05, w: 5.5, h: 0.65, fontSize: 14, italic: true, color: C.accent2, margin: 0, isTextBox: true });
  s.addNotes("A worked example of the update step. Before treatment, the population model says a patient's kidney dose per GBq is only known to within about a factor of two. One SPECT-based dosimetry measurement after cycle 1 shrinks that to about plus or minus 30 percent. That is what lets later cycles be personalised. Values are illustrative.");

  // ---------------------------------------------------------------- virtual trial
  s = content("THE COMBINED MODEL · VIRTUAL TRIAL", "Fixed vs model-guided activity, 140 virtual patients", P3);
  s.addChart(pres.charts.BAR, [
    { name: "Fixed 7.4 GBq × 4", labels: ["Over kidney limit", "Far below limit (<60 %)", "Mean TCP", "Mean P⁺"], values: [F.over, F.under, F.tcp, F.pplus].map((v) => +(v * 100).toFixed(1)) },
    { name: "Model-guided × 4", labels: ["Over kidney limit", "Far below limit (<60 %)", "Mean TCP", "Mean P⁺"], values: [G.over, G.under, G.tcp, G.pplus].map((v) => +(v * 100).toFixed(1)) },
  ], {
    x: 0.6, y: 1.7, w: 7.4, h: 4.75, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent6, HEX.accent2],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: HEX.dk1, dataLabelFormatCode: "0\"%\"",
    showTitle: true, title: "Percent of virtual patients / mean probability", titleFontSize: 13, titleColor: HEX.dk1,
    showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: HEX.dk1, valAxisMinVal: 0, valAxisMaxVal: 100, valAxisHidden: true,
    catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 12, valGridLine: { style: "none" }, catGridLine: { style: "none" }, ...CHART_TEXT, objectName: "virtual trial chart",
  });
  [[pct(F.over) + " → " + pct(G.over), "patients over the kidney BED limit"], [Math.round(F.bT) + " → " + Math.round(G.bT) + " Gy", "mean tumour BED"], [pct(F.pplus) + " → " + pct(G.pplus), "mean uncomplicated control P⁺"]].forEach(([n, d], i) => {
    const y = 1.7 + i * 1.55;
    s.addText(n, { x: 8.4, y, w: 4.3, h: 0.8, fontSize: 32, bold: true, color: C.accent2, fontFace: EQ, margin: 0, valign: "middle", isTextBox: true });
    s.addText(d, { x: 8.4, y: y + 0.8, w: 4.3, h: 0.5, fontSize: 15, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText(`Illustrative simulation, not patient data. Kidney ${SIM.P.dK_med} Gy/GBq and tumour ${SIM.P.dT_med} Gy/GBq (log-normal CV ${SIM.P.dK_sd}/${SIM.P.dT_sd}); BED limit ${SIM.P.L} Gy; guided activity capped at ${SIM.P.Amax} GBq (mean ${G.A.toFixed(1)} GBq).`, { x: 0.6, y: 6.5, w: 12.1, h: 0.4, fontSize: 11, italic: true, color: C.accent6, margin: 0, isTextBox: true });
  s.addNotes("The combined model run as a paired virtual trial: the same 140 virtual patients get both arms. With fixed activity, about one in ten exceeds the kidney limit while more than half are far below it. Model-guided activity moves everyone up to their own limit, raising tumour BED and the probability of uncomplicated control. This mirrors the schematic in my research-vision talk, now computed. All parameters are illustrative.");

  // ---------------------------------------------------------------- hybrid AI architecture
  s = content("THE COMBINED MODEL · HYBRID AI ARCHITECTURE", "Mechanistic backbone, AI where it pays", P3);
  const backbone = ["Imaging", "Cohort", "PBPK", "Dosimetry", "Radiobiology", "Outcome", "Decision"];
  const aiMods = ["nnU-Net, TotalSeg, denoising", "copulas, diffusion (MAISI)", "UDE + SBI posterior", "neural operator, DoseVLM", "Bayesian α/β, G priors", "survival ML + conformal", "Bayesian optimisation, RL"];
  backbone.forEach((b, i) => {
    const x = 0.6 + i * 1.76;
    s.addText(b, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x, y: 1.85, w: 1.55, h: 0.85, fill: { color: C.text2 }, color: C.background1, fontSize: 14, bold: true, align: "center", valign: "middle", margin: 0.04, isTextBox: true, objectName: `backbone ${b}` });
    if (i < backbone.length - 1) s.addShape(pres.shapes.LINE, { x: x + 1.56, y: 2.275, w: 0.19, h: 0, line: { color: C.accent6, width: 1.5, endArrowType: "triangle" }, objectName: "arrow" });
    s.addShape(pres.shapes.LINE, { x: x + 0.775, y: 2.72, w: 0, h: 0.5, line: { color: C.accent2, width: 1.5, dashType: "dash" }, objectName: "ai link" });
    s.addText(aiMods[i], { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x, y: 3.25, w: 1.55, h: 1.2, fill: { color: "E2F0EE" }, line: { color: C.accent2, width: 1 }, color: C.text1, fontSize: 12, align: "center", valign: "middle", margin: 0.05, isTextBox: true, objectName: `ai ${b}` });
  });
  s.addText("MECHANISTIC BACKBONE", { x: 0.6, y: 1.5, w: 6, h: 0.3, fontSize: 11, bold: true, color: C.accent6, margin: 0, isTextBox: true });
  s.addText("AI MODULES (replace or correct a step once validated)", { x: 0.6, y: 4.55, w: 8, h: 0.3, fontSize: 11, bold: true, color: C.accent6, margin: 0, isTextBox: true });
  const rules = [
    ["Keep the backbone interpretable", "Equations stay visible; AI corrects or accelerates, never hides, a step."],
    ["Validate each module alone", "Dose before response; a module enters the twin only after its own test."],
    ["Carry uncertainty through", "Every module outputs an interval; the decision uses the full posterior."],
  ];
  rules.forEach(([h, d], i) => {
    const x = 0.6 + i * 4.1;
    iconCircle(s, [IC.diagram, IC.shield, IC.dice][i], x, 5.0, 0.55);
    s.addText(h, { x: x + 0.7, y: 4.95, w: 3.25, h: 0.6, fontSize: 15, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
    s.addText(d, { x: x + 0.7, y: 5.55, w: 3.25, h: 1.1, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("This is the recommended architecture for the combined twin. The top row is the mechanistic chain. Under each step is the AI method that best accelerates or corrects it. Three rules keep it credible: the backbone stays interpretable, each module is validated on its own, and uncertainty is carried all the way to the decision.");

  // ---------------------------------------------------------------- roadmap
  s = content("THE COMBINED MODEL · BUILD PLAN", "Four modules turn the repo into the full twin", P3);
  const mods = [
    ["dt/pbpk.py", "PBPK ODEs + TAC fitting + Ã; calibrate with SBI", "Layers 1–2"],
    ["dt/radiobiology.py", "BED with G factor, EQD2, TCP, LKB-NTCP", "Layer 4"],
    ["dt/optimise.py", "Bayesian update, P⁺ objective, constrained activity search", "Layer 5"],
    ["dt/virtual_trial.py", "Cohort sampling, paired arms, uncertainty propagation", "Layers 1, 8"],
  ];
  mods.forEach(([f, d, l], i) => {
    const y = 1.75 + i * 1.12;
    s.addText(String(i + 1), { shape: pres.shapes.OVAL, x: 0.6, y: y + 0.12, w: 0.7, h: 0.7, fill: { color: C.accent1 }, color: C.text1, fontSize: 20, bold: true, align: "center", valign: "middle", margin: 0, fontFace: EQ, isTextBox: true, objectName: `module ${i + 1}` });
    s.addText(f, { x: 1.55, y, w: 3.4, h: 0.95, fontSize: 17, bold: true, color: C.text2, margin: 0, valign: "middle", fontFace: EQ, isTextBox: true });
    s.addText(d, { x: 4.95, y: y + 0.08, w: 5.6, h: 0.8, fontSize: 15, color: C.text1, fill: { color: C.background2 }, margin: 0.12, valign: "middle", isTextBox: true });
    tag(s, "new", 10.75, y + 0.33);
    s.addText(l, { x: 11.9, y, w: 0.85, h: 0.95, fontSize: 12, color: C.accent6, margin: 0, valign: "middle", isTextBox: true });
  });
  s.addText("Reuses what exists: dosevlm/ for external-beam dose and geometry, radiomics_pipeline/ for imaging biomarkers and evaluation metrics.", { x: 0.6, y: 6.35, w: 12.1, h: 0.45, fontSize: 13, italic: true, color: C.accent2, margin: 0, isTextBox: true });
  s.addNotes("Four new modules complete the combined model in code. The rest already exists in the repo. The simulation on the virtual-trial slide is a first sketch of virtual_trial.py.");

  // ================================================================ Conclusion
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Conclusion" });
  s.addText("TAKEAWAYS AND NEXT STEPS", { placeholder: "kicker" });
  s.addText("Mechanism first, AI to scale, validation throughout", { placeholder: "title" });
  s.addText([
    { text: "Already in code: dose physics, DVH, gamma, DoseVLM, radiomics and evaluation.", options: { bullet: true, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Combined: seven equations chain imaging to decision; Bayesian updates personalise every cycle.", options: { bullet: true, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Next: build pbpk, radiobiology, optimise and virtual_trial modules, then validate dose before response.", options: { bullet: true } },
  ], { placeholder: "body" });
  s.addNotes("Close with the three messages. Much of the stack already exists. The missing pieces are the kinetic and radiobiological layers, uncertainty propagation and a first pilot virtual trial.");

  // ================================================================ References
  s = content("KEY REFERENCES", "Sources for the models and AI methods", "Conclusion");
  const refsL = [
    "Abdollahi H et al. Theranostic digital twins. Theranostics 2024;14:3404–22",
    "Abdollahi H et al. PBPK in ¹⁷⁷Lu-PSMA therapy. Cancers 2024;16:3120",
    "Abdollahi H et al. Low-dose-rate radiobiology in RPT. IEEE TRPMS 2025;9:843–56",
    "Abadi E et al. Virtual clinical trials in medical imaging. J Med Imaging 2020",
    "Badano A et al. VICTRE in silico trial. JAMA Netw Open 2018",
    "Langendijk JA et al. Model-based proton selection. Radiother Oncol 2013",
    "Viceconti M et al. VV&UQ for in silico trials. Methods 2021",
    "Babier A et al. OpenKBP. Med Phys 2021",
    "Hänscheid H et al. Single-time-point dosimetry. J Nucl Med 2018",
  ];
  const refsR = [
    "Isensee F et al. nnU-Net. Nat Methods 2021",
    "Wasserthal J et al. TotalSegmentator. Radiol AI 2023",
    "Ma J et al. MedSAM. Nat Commun 2024",
    "Raissi M et al. Physics-informed neural networks. J Comput Phys 2019",
    "Chen RTQ et al. Neural ODEs. NeurIPS 2018",
    "Rackauckas C et al. Universal differential equations. arXiv 2020",
    "Li Z et al. Fourier neural operator. ICLR 2021",
    "Cranmer K et al. Simulation-based inference. PNAS 2020",
    "Hollmann N et al. TabPFN. Nature 2025",
  ];
  s.addText(bullets(refsL, 13), { x: 0.6, y: 1.7, w: 5.95, h: 4.9, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addText(bullets(refsR, 13), { x: 6.75, y: 1.7, w: 5.95, h: 4.9, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addText("Compiled from memory and from the repo review; verify volume and page numbers before formal citation.", { x: 0.6, y: 6.6, w: 12.1, h: 0.3, fontSize: 11, italic: true, color: C.accent6, margin: 0, isTextBox: true });
  s.addNotes("Key references for each model and AI family. Check bibliographic details before citing.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
