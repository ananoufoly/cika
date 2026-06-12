import { DEFAUTS, simulerPortefeuille, monteCarlo } from './moteur.js';

const fmt = x => Math.round(x).toLocaleString('fr-FR');
const fmtK = x => Math.abs(x) >= 1e6 ? (x / 1e6).toFixed(2) + 'M' : Math.abs(x) >= 1e3 ? Math.round(x / 1e3) + 'k' : Math.round(x).toString();
const pct = x => (x * 100).toFixed(0) + '%';
const $ = id => document.getElementById(id);
const meanArr = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;

let P = { ...DEFAUTS, _runs: 80 };

// scénarios = transformations de la config
const SCENARIOS = {
  nominal: p => p,
  comportemental: p => { p.comportemental_actif = true; p.choc_pd = 0.03; p.bascule_urgents = 0.30; return p; },
  macro: p => { p.macro_actif = true; p.z_choc = -2.5; p.z_persistance = 4; return p; },
  combine: p => { p.comportemental_actif = true; p.choc_pd = 0.03; p.bascule_urgents = 0.30; p.macro_actif = true; p.z_choc = -2.5; p.z_persistance = 4; return p; },
  bids_faibles: p => { p.part_urgent = 0.05; p.part_modere = 0.25; p.part_epargnant = 0.70; p.rho_mensuel = 0.01; return p; },
};
const HINTS = {
  nominal: 'Conjoncture normale, aucun stress. La cascade reste basse, la SFD ne perd presque rien.',
  comportemental: 'Choc sur les probabilités de défaut (+3 pts) et bascule de 30% vers des membres « urgents ».',
  macro: 'Choc systémique sévère (Z = −2,5) persistant 4 mois — défauts corrélés via les secteurs.',
  combine: 'Stress comportemental ET macro simultanés — le test le plus exigeant.',
  bids_faibles: 'Peu de membres enchérissent (population d\'épargnants, ρ faible) — réserve endogène minimale. Test de viabilité clé.',
};
let scenarioCourant = 'nominal';

document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active')); t.classList.add('active');
}));

// ---- config courante : défauts + scénario + sliders ----
function configCourante() {
  let p = { ...DEFAUTS };
  // appliquer sliders structure/mécanisme
  p.n_pools = P.n_pools; p.m_membres = P.m_membres; p.c = P.c; p.n_cycles = P.n_cycles;
  p.alpha = P.alpha; p.rho_mensuel = P.rho_mensuel; p.k_max = P.k_max;
  // appliquer scénario par-dessus (peut écraser rho pour bids_faibles)
  p = SCENARIOS[scenarioCourant]({ ...p });
  // ré-appliquer les sliders qui priment (sauf rho en bids_faibles)
  if (scenarioCourant !== 'bids_faibles') p.rho_mensuel = P.rho_mensuel;
  return p;
}

// ====================================================================
// 1 · ÉTAPES
// ====================================================================
function renderEtapes() {
  const pot = P.m_membres * P.c, imm = P.part_immediate * pot, dif = pot - imm;
  const etapes = [
    { n: '1', t: 'Collecte', v: fmt(pot) + ' XOF', s: 'M × c sur le séquestre SFD' },
    { n: '2', t: 'Immédiat', v: fmt(imm) + ' XOF', s: '50% au bénéficiaire (crédit SFD)' },
    { n: '3', t: 'Enchère', v: 'ρ = ' + pct(P.rho_mensuel) + '/mois', s: 'qui avance son différé' },
    { n: '4', t: 'Bid → split', v: 'α = ' + pct(P.alpha), s: 'commission + consignation' },
    { n: '5', t: 'Différé', v: fmt(dif) + ' − bid', s: 'restitué en fin de cycle' },
  ];
  $('etapes').innerHTML = etapes.map(e => `<div class="etape"><div class="n">${e.n}. ${e.t}</div><div class="v">${e.v}</div><small>${e.s}</small></div>`).join('');
}

// ====================================================================
// 2 · SIMULATEUR
// ====================================================================
const sliders = [
  ['c_npools', 'o_npools', 'n_pools', v => v],
  ['c_m', 'o_m', 'm_membres', v => v],
  ['c_c', 'o_c', 'c', v => fmt(v)],
  ['c_cyc', 'o_cyc', 'n_cycles', v => v],
  ['c_alpha', 'o_alpha', 'alpha', v => pct(v)],
  ['c_rho', 'o_rho', 'rho_mensuel', v => (v * 100).toFixed(1) + '%'],
  ['c_k', 'o_k', 'k_max', v => v],
  ['c_runs', 'o_runs', '_runs', v => v],
];
function syncSliders() {
  sliders.forEach(([cid, oid, key]) => { $(cid).value = P[key]; });
  updateOutputs();
}
function updateOutputs() {
  sliders.forEach(([cid, oid, key, f]) => { $(oid).textContent = f(P[key]); });
}
sliders.forEach(([cid, oid, key, f]) => {
  $(cid).addEventListener('input', e => {
    P[key] = +e.target.value; updateOutputs();
    if (['n_pools', 'm_membres', 'c', 'alpha', 'rho_mensuel'].includes(key)) renderEtapes();
  });
});
document.querySelectorAll('#scenarioSeg .seg-btn').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('#scenarioSeg .seg-btn').forEach(x => x.classList.remove('active'));
  b.classList.add('active'); scenarioCourant = b.dataset.scen;
  $('scenarioHint').textContent = HINTS[scenarioCourant];
  lancer();
}));
$('resetBtn').addEventListener('click', () => { P = { ...DEFAUTS, _runs: 80 }; scenarioCourant = 'nominal';
  document.querySelectorAll('#scenarioSeg .seg-btn').forEach((x, i) => x.classList.toggle('active', i === 0));
  $('scenarioHint').textContent = HINTS.nominal; syncSliders(); renderEtapes(); lancer();
});
$('runBtn').addEventListener('click', lancer);

let dernierAgg = null;
function lancer() {
  $('runBtn').textContent = 'Calcul…';
  setTimeout(() => {
    const p = configCourante();
    const agg = monteCarlo(p, P._runs, 12345);
    dernierAgg = agg;
    const rRef = simulerPortefeuille(p, 12345);  // run représentatif pour la marge/pool
    renderKPIs(agg, rRef.margePool);
    drawPertes(agg._pertes);
    drawCascade(agg);
    renderCascadeViz(agg);
    drawBreakEven(p);
    renderPnL(p);
    $('runBtn').textContent = '▶ Lancer la simulation';
  }, 20);
}

function renderKPIs(a, margePoolCourant) {
  const items = [
    { l: 'Taux complétion cycles', v: pct(a.tauxCompletion.moyenne), c: a.tauxCompletion.moyenne > 0.7 ? 'ok' : a.tauxCompletion.moyenne > 0.4 ? 'warn' : 'bad', s: 'membres non défaillants' },
    { l: 'Rétention cycle 1→2', v: pct(a.retention.moyenne), c: 'brand', s: 'restent actifs au cycle 2' },
    { l: 'Perte SFD (moy.)', v: fmtK(a.perteSFD.moyenne), c: a.perteSFD.moyenne < 300000 ? 'ok' : 'warn', s: 'bornée par la ligne contingente' },
    { l: 'Rendement net SFD', v: pct(a.rendementSFD.moyenne), c: 'ok', s: 'sur dépôts moyens' },
    { l: 'P&L SFD', v: fmtK(a.pnlSFD.moyenne), c: a.pnlSFD.moyenne > 0 ? 'ok' : 'bad', s: 'résultat net partenaire' },
    { l: 'P&L Opérateur', v: fmtK(a.pnlOp.moyenne), c: a.pnlOp.moyenne > 0 ? 'ok' : 'bad', s: 'à ' + P.n_pools + ' pools' },
    { l: 'Marge / pool', v: fmt(margePoolCourant), c: margePoolCourant > 0 ? 'ok' : 'bad', s: 'commission − coûts variables' },
    { l: 'Taux respect K', v: pct(a.tauxRespectK.moyenne), c: 'brand', s: 'diversification atteinte' },
  ];
  $('kpiGrid').innerHTML = items.map(i => `<div class="kpi"><div class="lbl">${i.l}</div><div class="val ${i.c}">${i.v}</div><small>${i.s}</small></div>`).join('');
}

// ---- distribution des pertes SFD ----
function drawPertes(pertes) {
  const cv = $('pertesCanvas'), ctx = cv.getContext('2d'); const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const data = pertes.map(x => x / 1000);
  if (!data.length) return;
  const lo = 0, hi = Math.max(...data, 1);
  const bins = 36, w = (hi - lo) / bins || 1, counts = new Array(bins).fill(0);
  data.forEach(v => { let b = Math.floor((v - lo) / w); b = Math.max(0, Math.min(bins - 1, b)); counts[b]++; });
  const maxc = Math.max(...counts), pad = 30, bw = (W - pad * 2) / bins;
  counts.forEach((c, i) => {
    const x = pad + i * bw, h = (c / maxc) * (H - 40);
    ctx.fillStyle = i / bins > 0.75 ? '#b91c1c' : '#0f4c4a';
    ctx.fillRect(x, H - 20 - h, bw - 1, h);
  });
  ctx.fillStyle = '#5b6b87'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
  ctx.fillText('0', pad, H - 4); ctx.textAlign = 'right'; ctx.fillText(fmtK(hi * 1000), W - 4, H - 4);
  ctx.textAlign = 'center'; ctx.fillText('perte SFD (milliers XOF) →', W / 2, H - 4);
}

// ---- cascade : barres par niveau ----
function drawCascade(a) {
  const cv = $('cascadeCanvas'), ctx = cv.getContext('2d'); const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const vals = [a.freq_n1.moyenne, a.freq_n2.moyenne, a.freq_n3.moyenne, a.freq_n4.moyenne];
  const labels = ['Niv 1\nRéserve', 'Niv 2\nMéta', 'Niv 3\nLigne SFD', 'Niv 4\nReport'];
  const couleurs = ['#0f4c4a', '#14706c', '#b18a3a', '#b91c1c'];
  const maxv = Math.max(...vals, 1), pad = 40, bw = (W - pad * 2) / 4;
  vals.forEach((v, i) => {
    const x = pad + i * bw, h = (v / maxv) * (H - 50);
    ctx.fillStyle = couleurs[i]; ctx.fillRect(x + 10, H - 30 - h, bw - 20, h);
    ctx.fillStyle = '#0b1220'; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(Math.round(v), x + bw / 2, H - 34 - h);
    ctx.fillStyle = '#5b6b87'; ctx.font = '10px sans-serif';
    labels[i].split('\n').forEach((ln, j) => ctx.fillText(ln, x + bw / 2, H - 14 + j * 11));
  });
}

function renderCascadeViz(a) {
  const niv = [
    { c: 'n1', t: 'Niveau 1', n: 'Réserve du pool', v: a.freq_n1.moyenne, m: a.montant ? '' : '', s: 'enchères + confiscations du pool' },
    { c: 'n2', t: 'Niveau 2', n: 'Méta-réserve', v: a.freq_n2.moyenne, s: 'mutualisation inter-pools' },
    { c: 'n3', t: 'Niveau 3', n: 'Ligne SFD', v: a.freq_n3.moyenne, s: 'plafonnée — perte SFD bornée' },
    { c: 'n4', t: 'Niveau 4', n: 'Report', v: a.freq_n4.moyenne, s: 'pro-rata + report cycle suivant' },
  ];
  $('cascadeViz').innerHTML = niv.map(x => `<div class="niv ${x.c}"><div class="num-niv">${Math.round(x.v)}</div><b>${x.t} — ${x.n}</b><small>${x.s}</small></div>`).join('');
}

// ---- break-even ----
function drawBreakEven(p) {
  const r = simulerPortefeuille(p, 12345);
  const margePool = r.margePool, fixes = r.coutsFixes;
  const cv = $('breakevenCanvas'), ctx = cv.getContext('2d'); const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const nmax = 1400, pad = 50;
  const be = margePool > 0 ? fixes / margePool : Infinity;
  const xs = n => pad + (n / nmax) * (W - pad - 10);
  const pnlAt = n => margePool * n - fixes;
  const pnlMin = pnlAt(0), pnlMax = pnlAt(nmax);
  const ys = v => H - 24 - ((v - pnlMin) / (pnlMax - pnlMin || 1)) * (H - 40);
  // zéro
  ctx.strokeStyle = '#5b6b87'; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(pad, ys(0)); ctx.lineTo(W - 10, ys(0)); ctx.stroke(); ctx.setLineDash([]);
  // courbe
  ctx.strokeStyle = '#0f4c4a'; ctx.lineWidth = 2.2; ctx.beginPath();
  for (let n = 0; n <= nmax; n += 20) { const x = xs(n), y = ys(pnlAt(n)); n === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); } ctx.stroke();
  // break-even (seulement si des frais de structure existent)
  if (fixes > 0 && isFinite(be) && be <= nmax) {
    ctx.strokeStyle = '#b18a3a'; ctx.setLineDash([5, 5]); ctx.beginPath(); ctx.moveTo(xs(be), 10); ctx.lineTo(xs(be), H - 22); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#b18a3a'; ctx.font = '12px sans-serif'; ctx.textAlign = 'center'; ctx.fillText('break-even ≈ ' + Math.round(be) + ' pools', xs(be), 22);
  } else if (fixes <= 0) {
    ctx.fillStyle = '#15803d'; ctx.font = '12px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText('marge ≈ ' + fmt(margePool) + ' XOF / pool — rentable dès le 1ᵉʳ pool', xs(50), 22);
  }
  ctx.fillStyle = '#5b6b87'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
  ctx.fillText(fmtK(pnlMax) + ' XOF', 4, 16); ctx.fillText(fmtK(pnlMin), 4, H - 24);
  ctx.textAlign = 'center'; ctx.fillText('nombre de pools →', W / 2, H - 6);
}

// ---- P&L cartes ----
function renderPnL(p) {
  const r = simulerPortefeuille(p, 12345);
  $('pnlGrid').innerHTML = `
    <div class="pnl-card op">
      <h3>Opérateur (plateforme)</h3>
      <div class="grand ${r.pnlOp >= 0 ? 'ok' : 'bad'}">${fmtK(r.pnlOp)} XOF</div>
      <div class="hint">à ${r.nPools} pools sur ${p.n_cycles * p.m_membres} mois</div>
      <ul>
        <li>Commissions fermes <b>${fmtK(r.commissionTot)}</b></li>
        <li>Coût acquisition <b>−${fmtK(r.coutAcq)}</b></li>
        <li>Coût opérationnel <b>−${fmtK(r.coutOps)}</b></li>
        ${r.coutsFixes > 0 ? `<li>Coûts fixes <b>−${fmtK(r.coutsFixes)}</b></li>` : ''}
        <li>Marge / pool <b>${fmt(r.margePool)}</b></li>
        <li>Rentabilité <b>${r.coutsFixes > 0 ? (isFinite(r.breakEven) ? 'dès ' + Math.round(r.breakEven) + ' pools' : '—') : 'dès le 1ᵉʳ pool'}</b></li>
      </ul>
    </div>
    <div class="pnl-card sfd">
      <h3>SFD (institution dépositaire)</h3>
      <div class="grand ${r.pnlSFD >= 0 ? 'ok' : 'bad'}">${fmtK(r.pnlSFD)} XOF</div>
      <div class="hint">rendement net ${pct(r.rendementSFD)} sur dépôts moyens</div>
      <ul>
        <li>Intérêts de crédit <b>${fmtK(r.interetsCredit)}</b></li>
        <li>Spread sur dépôts <b>${fmtK(r.spread)}</b></li>
        <li>Coût du risque (ligne) <b>−${fmtK(r.perteSFD)}</b></li>
        <li>Dépôts moyens <b>${fmtK(r.depotsMoyens)}</b></li>
        <li>Activation ligne SFD <b>${Math.round(r.freq.n3)}×</b></li>
      </ul>
    </div>`;
}

// ---- init ----
$('scenarioHint').textContent = HINTS.nominal;
syncSliders();
renderEtapes();
lancer();
